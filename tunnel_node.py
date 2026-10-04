#!/usr/bin/env python3
"""bitcoder tunnel node: send + receive simultaneously over SRT video.

One process, four roles:
  * local TCP SERVER  (clients connect here; their data goes out the
                      SRT-SEND channel, embedded in a live video stream)
  * SRT CALLER        (dials the peer's listen port; carries the local
                      clients' bytes as FEC-protected color-palette frames)
  * SRT LISTENER      (the peer's bytes arrive here as a live video stream)
  * local TCP CLIENT  (reconstructed peer data is sent out to local clients
                      that connect here)

Two nodes wired crosswise form a bidirectional tunnel:
      node A  --srt send-->  node B   (A's local clients reach B)
      node B  --srt send-->  node A   (B's local clients reach A)

The video codec is the file_to_video_bitcoder_color machinery: 8-corner
RGB palette, 3 bits per MxM block, 0/255 levels, 8-byte group header
(magic/seq/crc32), R=2 identical copies per group, systematic Cauchy MDS
FEC over GF(256) (k data + m parity groups per stripe; up to m lost
groups per stripe are repaired from the survivors).

Frames are fed to the send ffmpeg by a PACED writer (exactly fps
frames/second): SRT's send window is tiny (~130 ms), so bursting the
encoder overflows it and tears the link. The receive side drains the
SRT video, walks it in group windows, FEC-repairs gaps, demuxes packets
onto per-connection byte streams, and serves them as local TCP.

Usage:
  python3 tunnel_node.py --srt-listen srt://:9000 --srt-dial srt://127.0.0.1:9001 \
      [--tcp-srv-port 9010] [--tcp-cli-port 9011] \
      [--width 1280] [--height 720] [--fps 30] [--k 8] [--m 2] [--crf 23]
"""

import argparse
import logging
import os
import queue
import socket
import subprocess
import sys
import threading
import time

import numpy as np

from tunnel_core import (K, M, M_PAR, R, R_META, EOF_PAYLOAD, SeqWrapTracker,
                         cauchy_matrix, decode_group, decode_meta,
                         group_capacity, make_eof_group, parity_rows,
                         render_group, render_meta, stripe_data, stripe_of,
                         throughput_bps, verify_group)
from mux import Muxer
from demux import Demuxer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('tunnel-node')

FLUSH_IDLE_AFTER = 0.4    # s a partial send-stripe may sit (clients present)
IDLE_EOF_AFTER = 1.5      # s of idleness before the send "tunnel idle" EOF
FRAME_Q_MAX = 20          # send queue depth (~0.7 s of video)
RESYNC_LIMIT = 8          # max one-frame slides after a damaged rx window
MAX_BUFFER = 4 * 1024 * 1024   # per-connection buffering for late clients
QUIET_EXIT_GROUPS = 900   # safety net on the rx side
# SRT transport options — chosen for a hostile internet:
#   reconnect=-1     : re-dial FOREVER after any drop (outage-proof)
#   latency=400      : ARQ correction window in ms; big enough that a
#                      lossy/jittery link gets its lost packets re-requested
#                      and reordered INSTEAD of being dropped (SRT's core
#                      guarantee: a packet that arrived late-but-in-window is
#                      repaired, not lost)
#   recvbuffsize     : receive ring large enough for the ARQ window
#   passphrase       : SRT's built-in AES-CTR encryption of the whole UDP
#                      stream (added below when --srt-passphrase is set)
SRT_OPTS = ('reconnect=-1&latency=400&recvbuffsize=33554432'
            '&flowwindow=1024&inackdelay=10')


def with_srt_opts(url, mode, passphrase=None):
    """Attach an explicit SRT mode, robustness opts and (optionally) an
    AES passphrase to a URL."""
    parts = [f"mode={mode}", SRT_OPTS]
    if passphrase:
        parts.append(f"passphrase={passphrase}")
    if '?' in url or 'mode=' in url:
        sep = '&'
    else:
        sep = '?'
    return url + sep + '&'.join(parts)


class Client:
    """A local client socket on the send side."""
    __slots__ = ('sock', 'addr', 'cid')

    def __init__(self, sock, addr, cid):
        self.sock = sock
        self.addr = addr
        self.cid = cid

    def read_some(self):
        """bytes | b'' (peer closed) | None (timeout, still alive)."""
        try:
            return self.sock.recv(262144)
        except (socket.timeout, TimeoutError):
            return None
        except (ConnectionError, OSError):
            return b''


# ---------------------------------------------------------------------------
# SEND side: local TCP -> mux -> FEC stripes -> paced ffmpeg -> SRT
# ---------------------------------------------------------------------------

def stripe_frames(stripe_no, data_payloads, P, width, height):
    """One stripe: k data + m parity groups, each R copies of rgb24 frames."""
    k = len(data_payloads)
    rows = np.stack([np.frombuffer(p, dtype=np.uint8) for p in data_payloads])
    parity = parity_rows(rows, P)
    seq0 = stripe_no * (k + len(parity))
    frames = []
    for i, payload in enumerate(data_payloads + [p.tobytes() for p in parity]):
        f = render_group(payload, seq0 + i, M, width, height)
        frames.extend([f] * R)
    return frames


class SendSide:
    def __init__(self, args, B):
        self.args = args
        self.B = B
        self.P = cauchy_matrix(args.k, args.m)
        self.mux = Muxer(B)
        self.clients = {}
        self.next_cid = [1]
        self.frame_q = queue.Queue()
        self.stripe = []
        self.stripe_no = 0
        self.stripe_open_at = time.time()
        self.eof_frames = [make_eof_group(args.width, args.height)] * R
        self.idle_eof_sent = False
        self.last_idle_emit = 0.0
        self.last_log = time.time()
        self.idle_period = (args.k + args.m) * R / args.fps
        self.stop = threading.Event()
        self.alive = True
        self.ff = None
        self.srv = None
        # R_META metadata copies, written by the feed thread right after
        # each (re)dial establishes the link
        self.meta_frames = [render_meta({
            'fn': 'bitcoder-tunnel', 'w': args.width, 'h': args.height,
            'M': M, 'R': R, 'k': args.k, 'm': args.m, 'B': self.B,
            'fps': args.fps,
        }, args.width, args.height) for _ in range(R_META)]

    def _push(self, frames):
        while self.frame_q.qsize() > FRAME_Q_MAX and not self.stop.is_set():
            time.sleep(0.005)
        self.frame_q.put(frames)

    def _emit_stripe(self, stripe_data):
        self._push(stripe_frames(self.stripe_no, stripe_data, self.P,
                                 self.args.width, self.args.height))
        self.stripe_no += 1

    def _emit_eof(self, tag):
        self._push(self.eof_frames)
        log.info("[send] EOF sentinel sent (%s)", tag)

    def _flush_partial_stripe(self):
        if self.mux.pending():
            self.stripe.append(self.mux.flush_partial())
        if self.stripe:
            while len(self.stripe) < self.args.k:
                self.stripe.append(b'\x00' * self.B)
            self._emit_stripe(self.stripe)
            self.stripe = []
            self.stripe_open_at = time.time()

    def _client_gone(self, cid, reason):
        c = self.clients.pop(cid, None)
        if c is None:
            return
        try:
            c.sock.close()
        except OSError:
            pass
        self.mux.feed(cid, b'', fin=True)
        log.info("[send] client %d closed (%s)", cid, reason)

    def start(self):
        a = self.args
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((a.tcp_srv_host, a.tcp_srv_port))
        self.srv.listen(16)
        self.srv.settimeout(0.1)
        log.info("[send] local TCP server on %s:%d (clients connect here; "
                 "their data goes out the SRT send channel)",
                 a.tcp_srv_host, a.tcp_srv_port)
        # One thread owns the whole send-ffmpeg lifecycle (spawn -> header
        # -> paced pump -> re-dial on death), so there is no swap-out race
        # between a writer and a supervisor, and a first-dial that fails
        # (peer not up yet) is retried instead of crashing the node.
        threading.Thread(target=self._ffmpeg_runner, daemon=True).start()
        threading.Thread(target=self._main_loop, daemon=True).start()

    def _spawn_ffmpeg(self):
        a = self.args
        srt_output = with_srt_opts(a.srt_dial, 'caller', a.srt_passphrase)
        proc = subprocess.Popen([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{a.width}x{a.height}', '-pix_fmt', 'rgb24',
            '-r', str(a.fps), '-i', '-',
            '-map', '0:v:0',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast', '-tune', 'zerolatency',
            '-crf', str(a.crf),
            '-g', str(a.fps), '-keyint_min', str(a.fps), '-bf', '0',
            '-f', 'mpegts', srt_output,
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
           stderr=subprocess.PIPE)
        self.ff = proc
        return proc, srt_output

    def _ffmpeg_runner(self):
        """Own the send-ffmpeg lifecycle. Loop forever (until stop):
        spawn -> write R_META header -> pump frames at fps pace. A broken
        pipe (SRT caller that could not connect, or a dropped link) is
        NOT fatal: wait a moment and re-dial. The header is re-sent on
        every (re)dial so a fresh receiver always gets it first; a
        mid-stream receiver resyncs past the re-sent header frames."""
        while not self.stop.is_set():
            try:
                proc, srt_output = self._spawn_ffmpeg()
            except Exception as e:
                log.error("[send] ffmpeg spawn failed: %s; retry in 2 s", e)
                time.sleep(2.0)
                continue
            log.info("[send] ffmpeg up, SRT dial target: %s", srt_output)
            # header (non-fatal: a no-listener caller dies on the first write)
            ok = True
            try:
                for mf in self.meta_frames:
                    proc.stdin.write(mf)
            except (BrokenPipeError, OSError):
                ok = False
            if not ok:
                log.info("[send] SRT caller has no listener yet; "
                         "re-dialing in 2 s")
                time.sleep(2.0)
                continue
            # paced pump: exactly fps frames/second, never burst
            next_t = time.time()
            try:
                while not self.stop.is_set():
                    item = self.frame_q.get()
                    if item is None:
                        return
                    for f in item:
                        proc.stdin.write(f)
                        next_t += 1.0 / self.args.fps
                        now = time.time()
                        if now < next_t:
                            time.sleep(next_t - now)
                        else:
                            next_t = now
            except (BrokenPipeError, OSError):
                log.info("[send] SRT link dropped (ffmpeg pipe closed); "
                         "re-dialing in 2 s")
                time.sleep(2.0)
            except Exception as e:
                log.error("[send] pump error: %s; re-dialing in 2 s", e)
                time.sleep(2.0)

    def _main_loop(self):
        a = self.args
        try:
            while not self.stop.is_set():
                # 1. accept new local clients
                while True:
                    try:
                        sock, addr = self.srv.accept()
                    except (socket.timeout, TimeoutError, OSError):
                        break
                    if self.next_cid[0] > 254:
                        log.warning("[send] connection limit reached, "
                                    "refusing %s", addr)
                        sock.close()
                        continue
                    c = Client(sock, addr, self.next_cid[0])
                    self.next_cid[0] += 1
                    self.clients[c.cid] = c
                    sock.settimeout(0.02)
                    log.info("[send] client %d connected: %s", c.cid, addr)
                    burst = c.read_some()
                    if burst:
                        self.mux.feed(c.cid, burst)

                # 2. read every client (with backpressure: a client whose
                #    data is held back is not read again until admitted)
                for c in list(self.clients.values()):
                    if self.mux.backpressure(c.cid):
                        continue
                    data = c.read_some()
                    if data is None:
                        continue
                    if data:
                        self.mux.feed(c.cid, data)
                    else:
                        self._client_gone(c.cid, 'peer closed')

                # 2b. admit held-back data as group space frees up
                for cid in list(self.mux.holdback.keys()):
                    n = self.mux.admitted(cid)
                    if n:
                        self.mux.admit(cid, n)

                # 3-4. fill the stripe, emit when full
                while self.mux.pending() >= self.B and len(self.stripe) < a.k:
                    self.stripe.append(self.mux.next_group())
                if len(self.stripe) == a.k:
                    self._emit_stripe(self.stripe)
                    self.stripe = []
                    self.stripe_open_at = time.time()

                now = time.time()
                if self.clients:
                    if (self.stripe and self.mux.pending() == 0
                            and not self.mux.holdback
                            and now - self.stripe_open_at > FLUSH_IDLE_AFTER):
                        self._flush_partial_stripe()
                    if not self.stripe:
                        time.sleep(0.003)
                else:
                    if self.stripe or self.mux.pending():
                        self._flush_partial_stripe()
                    if now - self.last_idle_emit >= self.idle_period:
                        self._emit_stripe([b'\x00' * self.B] * a.k)
                        self.last_idle_emit = now
                    if (not self.idle_eof_sent
                            and now - self.stripe_open_at > IDLE_EOF_AFTER
                            and not self.stripe and self.mux.pending() == 0
                            and not self.mux.holdback):
                        # The idle EOF is a stream-level "nothing in flight"
                        # marker. It MUST NOT be emitted while a stripe or
                        # muxed data is still being transmitted: a receiver
                        # that sees it skips to its position in the stream,
                        # and the groups behind it (with lower seq than the
                        # new cursor) fall out of the stripe window and are
                        # dropped — the far client then gets one truncated
                        # blob plus an early FIN.
                        self._emit_eof('tunnel idle')
                        self.idle_eof_sent = True
                    time.sleep(0.005)

                if now - self.last_log > 10:
                    self.last_log = now
                    log.info("[send] %d stripes sent, %d client(s), "
                             "queue=%d", self.stripe_no, len(self.clients),
                             self.frame_q.qsize())
        except Exception as e:
            log.error("[send] main loop error: %s", e)
        finally:
            self.alive = False

    def shutdown(self):
        self.stop.set()
        try:
            self._flush_partial_stripe()
            self._emit_eof('shutdown')
            self.frame_q.put(None)   # wakes the ffmpeg_runner pump
        except Exception:
            pass
        # close ffmpeg stdin so the runner's pump unblocks and it exits
        try:
            if self.ff is not None:
                self.ff.stdin.close()
        except Exception:
            pass
        # give the runner a moment to finish the EOF, then kill ffmpeg
        try:
            if self.ff is not None:
                self.ff.wait(timeout=12)
        except Exception:
            try:
                if self.ff is not None:
                    self.ff.terminate()
                    self.ff.wait(timeout=5)
            except Exception:
                pass
        for c in list(self.clients.values()):
            try:
                c.sock.close()
            except OSError:
                pass
        if self.srv is not None:
            self.srv.close()
        log.info("[send] stopped after %d stripes", self.stripe_no)


# ---------------------------------------------------------------------------
# RECEIVE side: SRT video -> group walker + FEC repair -> demux -> local TCP
# ---------------------------------------------------------------------------

class LocalClient:
    """A local client socket on the receive side (peer data goes to it)."""
    __slots__ = ('sock', 'addr', 'cid')

    def __init__(self, sock, addr, cid):
        self.sock = sock
        self.addr = addr
        self.cid = cid


class RecvSide:
    def __init__(self, args, out_dir, dump):
        self.args = args
        self.out_dir = out_dir
        self.dump = dump
        self.socks = {}
        self.pending = {}
        self.finished = set()
        self.fin_flags = set()
        self._lock = threading.Lock()
        self.dumper = (open(os.path.join(out_dir, 'tunnel_dump.bin'), 'wb')
                       if dump else None)
        self.dm = Demuxer()
        self.seq_trk = SeqWrapTracker()
        self.ff = None
        self.stop = threading.Event()
        self.stream_ended = threading.Event()
        self.groups = 0
        self.repaired = 0
        self.crc_bad = 0
        self.resyncs = 0

    # -- local TCP serving -----------------------------------------------------
    def _accept_loop(self):
        a = self.args
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((a.tcp_cli_host, a.tcp_cli_port))
        srv.listen(16)
        srv.settimeout(0.2)
        log.info("[recv] local TCP client-port on %s:%d (peer data is "
                 "served here)", a.tcp_cli_host, a.tcp_cli_port)
        while not self.stop.is_set():
            try:
                sock, addr = srv.accept()
            except (socket.timeout, TimeoutError):
                continue
            except OSError:
                return
            with self._lock:
                # Assign a connection id with buffered data if one exists:
                # the sender numbers connections in its own order, and data
                # may arrive (and buffer) before the local client connects.
                cid = 1
                for c in sorted(self.pending) + sorted(self.fin_flags):
                    if c not in self.socks and c not in self.finished:
                        cid = c
                        break
                else:
                    while cid in self.socks or cid in self.finished:
                        cid += 1
                if cid > 254:
                    log.warning("[recv] connection id space exhausted, "
                                "dropping %s", addr)
                    sock.close()
                    continue
                self.socks[cid] = sock
                buf = self.pending.pop(cid, b'')
                had_fin = cid in self.fin_flags
                if had_fin:
                    self.fin_flags.discard(cid)
            log.info("[recv] local client %d connected: %s", cid, addr)
            if buf:
                try:
                    sock.sendall(buf)
                except OSError:
                    self._gone(cid)
                    continue
            if had_fin:
                self._fin(cid, sock)

    def _active(self):
        with self._lock:
            return len(self.socks)

    def _deliver(self, conn, data, fin):
        if conn > 254:
            return
        if self.dumper is not None:
            self.dumper.write(data)
        with self._lock:
            sock = self.socks.get(conn)
        if sock is None:
            if conn in self.finished:
                return
            p = self.pending.get(conn, b'')
            if len(p) + len(data) > MAX_BUFFER:
                log.warning("[recv] connection %d buffer overflow", conn)
                self._gone(conn)
                return
            self.pending[conn] = p + data
            if fin:
                self.fin_flags.add(conn)
            return
        try:
            sock.sendall(data)
        except OSError:
            self._gone(conn)
            log.warning("[recv] local client %d dropped", conn)
            return
        if fin:
            self._fin(conn, sock)

    def _fin(self, conn, sock):
        self._gone(conn)
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        log.info("[recv] connection %d finished", conn)

    def _gone(self, conn):
        with self._lock:
            self.finished.add(conn)
            self.socks.pop(conn, None)
            self.pending.pop(conn, None)
            self.fin_flags.discard(conn)

    # -- SRT video reading ------------------------------------------------------
    def _frame_reader(self, proc, frame_size, q):
        n = 0
        # --raw-dump: tee every raw frame (before the walker) to a file for
        # offline post-mortem. Decoding that file separately tells us
        # whether the frames arrive corrupted or the walker mangles them.
        dump = getattr(self.args, 'raw_dump', None)
        dfile = open(dump, 'wb') if dump else None
        try:
            while not self.stop.is_set():
                d = proc.stdout.read(frame_size)
                if not d or len(d) != frame_size:
                    break
                if dfile is not None:
                    dfile.write(d)
                q.put(d)
                n += 1
                if n % 300 == 0:
                    log.info("[recv] %d frames from SRT", n)
        except OSError:
            pass
        finally:
            # KEEP DRAINING until the stop flag is set, even if ffmpeg has
            # already exited: an unread pipe backpressures ffmpeg, which
            # backpressures SRT, which tears the whole link down.
            try:
                while not self.stop.is_set():
                    if not proc.stdout.read(frame_size):
                        break
            except OSError:
                pass
            try:
                proc.stdout.close()
            except OSError:
                pass
            if dfile is not None:
                dfile.close()
            log.info("[recv] frame reader done (%d frames)", n)

    def start(self):
        a = self.args
        srt_input = with_srt_opts(a.srt_listen, 'listener', a.srt_passphrase)
        self.ff = subprocess.Popen([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', srt_input,
            '-map', '0:v:0',
            '-c:v', 'rawvideo',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-',
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log.info("[recv] SRT listener up at %s; waiting for the peer...",
                 srt_input)
        threading.Thread(target=self._accept_loop, daemon=True).start()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        a = self.args
        width, height = a.width, a.height
        frame_size = width * height * 3
        q = queue.Queue(maxsize=96)
        threading.Thread(target=self._frame_reader,
                         args=(self.ff, frame_size, q), daemon=True).start()

        try:
            # ---- metadata header ----
            meta_frames = []
            for _ in range(R_META):
                meta_frames.append(q.get(timeout=120))
            meta = decode_meta(meta_frames, width, height)
            if not meta:
                raise SystemExit("[recv] metadata decode failed")
            if int(meta.get('w', -1)) != width or int(meta.get('h', -1)) != height:
                raise SystemExit(f"[recv] geometry mismatch: peer streams "
                                 f"{meta.get('w')}x{meta.get('h')}, expected "
                                 f"{width}x{height}")
            M_ = int(meta.get('M', M))
            R_ = int(meta.get('R', R))
            k = int(meta.get('k', K))
            m = int(meta.get('m', M_PAR))
            B = int(meta.get('B', group_capacity(width, height, M_)))
            P = cauchy_matrix(k, m)
            log.info("[recv] peer meta: k=%d m=%d B=%d (R=%d M=%d)",
                     k, m, B, R_, M_)

            # ---- group walker ----
            buf = []
            base = 0
            stripe_no = -1
            stripe_recv = []
            flushed = set()       # stripe numbers already delivered (a stripe
            # MUST be delivered exactly once: re-feeding a stripe's groups to
            # the demuxer re-appends its (possibly partial) packets and
            # desyncs the whole byte stream)
            cursor = 0
            idle_groups = 0
            last_activity = time.time()

            def get_frame():
                while not self.stop.is_set():
                    try:
                        return q.get(timeout=5)
                    except queue.Empty:
                        if self.ff.poll() is not None:
                            try:
                                return q.get_nowait()
                            except queue.Empty:
                                return None
                return None

            def get_upto(abs_end):
                """Ensure the local buffer holds frames up to absolute index
                abs_end. `buf[0]` is absolute `base`, so the needed LOCAL
                length is (abs_end - base) — using the absolute index here
                would over/under-fill after compaction and desync the walker."""
                nonlocal last_activity
                need = abs_end - base
                while len(buf) < need:
                    d = get_frame()
                    if d is None:
                        return False
                    buf.append(d)
                    last_activity = time.time()
                return True

            def compact(cur):
                nonlocal base
                drop = cur - base
                if drop >= R_:
                    del buf[:drop]
                    base += drop

            def try_decode(abs_off):
                # `abs_off` is an ABSOLUTE frame index; the group's frames
                # live at LOCAL index (abs_off - base) in the compacted buf.
                local = abs_off - base
                return decode_group(buf[local:local + R_], M_, width, height)

            def resync(from_off):
                for d in range(RESYNC_LIMIT):
                    off = from_off + d
                    if not get_upto(off + R_):
                        return None
                    h8, payload = try_decode(off)
                    if h8 is None:
                        continue
                    if verify_group(h8, payload)[1]:
                        return h8, payload, off
                return None

            def is_eof(h8, payload):
                seq, ok = verify_group(h8, payload)
                return ok and seq == 0xFFFF and payload[:4] == EOF_PAYLOAD

            def deliver_rows(rows):
                for p in rows:
                    for conn, data, fin in self.dm.feed(p.tobytes()):
                        self._deliver(conn, data, fin)

            def flush_stripe(no, allow_partial=False):
                # A stripe is delivered EXACTLY ONCE. Without this guard the
                # main loop double-flushes: when the last group of a stripe
                # lands, `all(...)` flushes it; then the FIRST group of the
                # next stripe sees s != stripe_no and flushes the same,
                # already-delivered stripe again. Re-feeding a stripe's
                # (possibly partial) packets re-appends them to the demuxer
                # byte stream, which desyncs every later packet (the far
                # client then gets one capped blob + an early FIN).
                if no in flushed:
                    return False
                flushed.add(no)
                recv = stripe_recv[:k + m]
                n_lost = sum(1 for x in recv if x is None)
                n_data = sum(1 for x in recv[:k] if x is not None)
                if n_lost > m:
                    # Unrecoverable loss of this stripe (the SRT link dropped
                    # more than m groups). For a LIVE tunnel the right move
                    # is to drop this stripe and keep going — exactly like a
                    # dropped packet on a real network — not to tear the
                    # whole stream down. FEC makes the common loss
                    # repairable; an unrecoverable one just costs this stripe.
                    log.warning("[recv] stripe %d: %d of %d groups lost (%d "
                                "correctable); dropping stripe, %d data "
                                "group(s) lost", no, n_lost, k + m, m,
                                k - n_data)
                    return False
                rows = stripe_data(recv, k, P)
                if n_lost:
                    self.repaired += n_lost
                    log.info("[recv] stripe %d: repaired %d lost group(s)",
                             no, n_lost)
                deliver_rows(rows)

            while not self.stop.is_set():
                h8 = payload = None
                off = None
                if get_upto(cursor + R_):
                    h8, payload = try_decode(cursor)
                    if h8 is not None:
                        seq, ok = verify_group(h8, payload)
                        if ok or is_eof(h8, payload):
                            off = cursor
                        else:
                            self.crc_bad += 1
                            h8 = payload = None
                if h8 is None:
                    r = resync(cursor)
                    self.resyncs += 1
                    if r is None:
                        log.warning("[recv] stream ended (peer shut down)")
                        break
                    h8, payload, off = r

                if is_eof(h8, payload):
                    log.info("[recv] tunnel-idle EOF after %d groups "
                             "(%d repaired)", self.groups, self.repaired)
                    idle_groups = 0
                    cursor = off + R_
                    compact(cursor)
                    continue

                seq, _ = verify_group(h8, payload)
                g = self.seq_trk.true_index(seq)
                s = stripe_of(g, k, m)
                if s != stripe_no:
                    if stripe_no >= 0:
                        flush_stripe(stripe_no)
                    stripe_no = s
                    stripe_recv = [None] * (k + m)
                idx = g - s * (k + m)
                if 0 <= idx < k + m and stripe_recv[idx] is None:
                    stripe_recv[idx] = payload
                self.groups += 1
                if self.groups <= 30 or self.groups % 100 == 0:
                    log.info("[recv] group %d (seq %d, stripe %d)",
                             self.groups, seq, s)

                if payload == b'\x00' * B:
                    idle_groups += 1
                else:
                    idle_groups = 0

                if all(x is not None for x in stripe_recv):
                    flush_stripe(stripe_no)

                cursor = off + R_
                compact(cursor)

                if (idle_groups >= QUIET_EXIT_GROUPS and not self._active()
                        and time.time() - last_activity > 30):
                    log.info("[recv] stream quiet for %d groups with no "
                             "local clients; exiting", idle_groups)
                    break
        except SystemExit:
            raise
        except Exception as e:
            log.error("[recv] error: %s", e)
            raise
        finally:
            if stripe_no >= 0:
                try:
                    flush_stripe(stripe_no, allow_partial=True)
                except SystemExit:
                    pass
            self.stream_ended.set()
            log.info("[recv] stream closed: %d groups, %d repaired, "
                     "%d bad-CRC frames, %d resyncs",
                     self.groups, self.repaired, self.crc_bad,
                     self.resyncs)

    def shutdown(self):
        self.stop.set()
        try:
            if self.ff is not None:
                self.ff.wait(timeout=8)
        except Exception:
            try:
                self.ff.terminate()
                self.ff.wait(timeout=5)
            except Exception:
                pass
        for c in list(self.socks.values()):
            try:
                c.close()
            except OSError:
                pass
        if self.dumper is not None:
            self.dumper.close()
        log.info("[recv] stopped")


# ---------------------------------------------------------------------------

def run(args):
    B = group_capacity(args.width, args.height, M)
    log.info("node: %dx%d stream, %d B/group, ~%.0f KB/s per direction "
             "at full stripes", args.width, args.height, B,
             throughput_bps(args.width, args.height, args.fps,
                            args.k, args.m) / 1024)
    os.makedirs(args.out, exist_ok=True)

    recv = RecvSide(args, args.out, args.dump)
    send = SendSide(args, B)
    recv.start()
    send.start()

    stop = threading.Event()
    try:
        while not stop.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        log.info("interrupted; shutting down node")
    finally:
        send.shutdown()
        recv.shutdown()
        log.info("node stopped")


def main():
    ap = argparse.ArgumentParser(
        description='bitcoder tunnel node: simultaneous SRT send + receive')
    ap.add_argument('--srt-listen', default='srt://:9000',
                    help='SRT URL to LISTEN on (peer data arrives here), '
                         'e.g. srt://:9000')
    ap.add_argument('--srt-dial', default='srt://127.0.0.1:9001',
                    help='SRT URL to DIAL (local client data leaves here), '
                         'e.g. srt://PEER-IP:9000')
    ap.add_argument('--srt-passphrase', default=None,
                    help='shared AES-CTR passphrase for SRT encryption '
                         '(BOTH nodes must pass the same value to talk; '
                         'leaving it unset runs the stream unencrypted)')
    ap.add_argument('--tcp-srv-host', default='0.0.0.0')
    ap.add_argument('--tcp-srv-port', type=int, default=9010,
                    help='local port: clients connect HERE, data goes to peer')
    ap.add_argument('--tcp-cli-host', default='0.0.0.0')
    ap.add_argument('--tcp-cli-port', type=int, default=9011,
                    help='local port: peer data is served HERE to clients')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--k', type=int, default=K, help='data groups per stripe')
    ap.add_argument('--m', type=int, default=M_PAR,
                    help='parity groups per stripe (repairs up to m lost)')
    ap.add_argument('--crf', type=int, default=0,
                    help='x264 CRF; 0 = lossless (REQUIRED: the 0/255 block '
                         'palette must round-trip exactly, any lossy '
                         'quantization smears block boundaries)')
    ap.add_argument('--out', default='tunnel_out',
                    help='output directory (dump, logs)')
    ap.add_argument('--dump', action='store_true',
                    help='also append every received byte to '
                         '<out>/tunnel_dump.bin')
    ap.add_argument('--raw-dump', default=None, metavar='FILE',
                    help='debug: tee every raw decoded frame to FILE '
                         '(offline post-mortem of the arriving stream)')
    args = ap.parse_args()
    if args.srt_passphrase is not None and not (10 <= len(args.srt_passphrase)
                                                <= 128):
        ap.error('--srt-passphrase must be 10..128 chars (SRT spec); got '
                 f'{len(args.srt_passphrase)} — ffmpeg would die on spawn '
                 'and both nodes would re-dial into the void forever')
    if args.width % 2 or args.height % 2:
        ap.error('width/height must be even (yuv420p)')
    if not (1 <= args.k and args.k + args.m <= 255):
        ap.error('invalid stripe size k+m')
    if 2 * args.k + args.m - 2 > 255:
        ap.error(f'k={args.k} m={args.m} exceeds the GF(256) Cauchy cap '
                 f'(2k+m-2 > 255)')
    run(args)


if __name__ == '__main__':
    main()
