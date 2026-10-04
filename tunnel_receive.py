#!/usr/bin/env python3
"""bitcoder-tunnel receiver: SRT video stream -> live TCP connections.

This is the "video in, internet out" side of the tunnel. It listens for
the SRT stream (ffmpeg), decodes the metadata header, then walks the
stream in group windows of R frames. Each group is verified (magic/seq/
crc32); gaps of <= m groups per stripe are repaired with the GF(256)
Cauchy MDS code. Reconstructed payloads are demuxed into per-connection
byte streams, which are served back out as local TCP connections.

A first EOF sentinel in the stream means "tunnel idle" (the sender keeps
streaming idle groups and accepts new clients) — the receiver keeps going.
The stream ending (the sender's shutdown EOF closes the SRT connection)
ends the receiver.

Usage:
  python3 tunnel_receive.py --srt srt://:9000 [--tcp-port 9001] [--dump]
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

from tunnel_core import (K, M, M_PAR, R, R_META, EOF_PAYLOAD,
                         SeqWrapTracker, cauchy_matrix, decode_group,
                         decode_meta, group_capacity, stripe_data,
                         stripe_of, verify_group)
from demux import Demuxer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('tunnel-recv')

RESYNC_LIMIT = 8       # max one-frame slides after a damaged window
QUIET_EXIT_GROUPS = 900  # safety net: exit after this many idle groups
                        # while there are no local clients AND the stream
                        # has stopped delivering new frames
SRT_RECV_OPTS = 'latency=0&recvbuffsize=16777216&flowwindow=1024'


class Conns:
    """Local TCP side: delivers (conn, data, fin) to sockets.

    Connection ids are unique per receiver session and are NEVER reused.
    Data that arrives before the local client connects is buffered
    (up to MAX_BUFFER per connection) and flushed on connect."""

    MAX_BUFFER = 4 * 1024 * 1024

    def __init__(self, host, port, out_file, dump):
        self.host = host
        self.port = port
        self.dump = dump
        self.dump_path = os.path.join(out_file, 'tunnel_dump.bin')
        self.socks = {}
        self.pending = {}
        self.finished = set()
        self.fin_flags = set()
        self.dumper = open(self.dump_path, 'wb') if dump else None
        self._lock = threading.Lock()

    def _accept_loop(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(16)
        srv.settimeout(0.2)
        while True:
            try:
                sock, addr = srv.accept()
            except (socket.timeout, TimeoutError):
                continue
            except OSError:
                return
            with self._lock:
                cid = 1
                while cid in self.socks or cid in self.finished:
                    cid += 1
                if cid > 254:
                    log.warning("connection id space exhausted, dropping %s",
                                addr)
                    sock.close()
                    continue
                self.socks[cid] = sock
                buf = self.pending.pop(cid, b'')
                had_fin = cid in self.fin_flags
                if had_fin:
                    self.fin_flags.discard(cid)
            log.info("local client %d connected: %s", cid, addr)
            if buf:
                try:
                    sock.sendall(buf)
                except OSError:
                    self._gone(cid)
                    continue
            if had_fin:
                self._fin(cid, sock)

    def start(self):
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def deliver(self, conn, data, fin):
        if conn > 254:
            log.warning("bad conn id %d, dropping packet", conn)
            return
        if self.dumper is not None:
            self.dumper.write(data)
        with self._lock:
            sock = self.socks.get(conn)
        if sock is None:
            if conn in self.finished:
                return
            p = self.pending.get(conn, b'')
            if len(p) + len(data) > self.MAX_BUFFER:
                log.warning("connection %d buffer overflow, dropping data", conn)
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
            log.warning("local client %d dropped", conn)
            return
        if fin:
            self._fin(conn, sock)

    def _fin(self, conn, sock):
        self._gone(conn)
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        log.info("connection %d finished", conn)

    def _gone(self, conn):
        with self._lock:
            self.finished.add(conn)
            self.socks.pop(conn, None)
            self.pending.pop(conn, None)
            self.fin_flags.discard(conn)

    @property
    def active(self):
        with self._lock:
            return len(self.socks)

    def flush(self):
        if self.dumper is not None:
            self.dumper.flush()


def frame_reader(proc, frame_size, q, stop):
    n = 0
    try:
        while not stop.is_set():
            d = proc.stdout.read(frame_size)
            if not d or len(d) != frame_size:
                log.info("[reader] EOF/short read: %d frames, got %d/%d bytes",
                         n, len(d) if d else 0, frame_size)
                break
            q.put(d)
            n += 1
            if n % 100 == 0:
                log.info("[reader] %d frames from ffmpeg", n)
    except OSError as e:
        log.info("[reader] OSError after %d frames: %s", n, e)
    finally:
        try:
            proc.stdout.close()
        except OSError:
            pass
        log.info("[reader] thread done, total %d frames", n)


def run(args):
    width, height = args.width, args.height
    frame_size = width * height * 3
    log.info("stream %dx%d (rgb24 frame %d B)", width, height, frame_size)

    # The receiver is the SRT LISTENER: it starts immediately and accepts the
    # sender's caller connection whenever it arrives. (A probe-first design
    # would deadlock: nothing listens until probed, nothing can be probed
    # until something listens.) The metadata header decodes the real
    # geometry and is checked against ours below.
    # ?mode=listener is EXPLICIT: a bare 'srt://host:port' makes ffmpeg a
    # caller, which would dial itself and fail.
    srt_input = args.srt if 'mode=' in args.srt else args.srt + '?mode=listener'
    ff = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', srt_input,
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-',
    ], stdout=subprocess.PIPE, stderr=sys.stderr)
    log.info("SRT listener up at %s; waiting for the sender...", srt_input)

    q = queue.Queue(maxsize=64)
    stop = threading.Event()
    threading.Thread(target=frame_reader, args=(ff, frame_size, q, stop),
                     daemon=True).start()

    # ---- metadata -------------------------------------------------------------
    meta_frames = []
    for _ in range(R_META):
        try:
            meta_frames.append(q.get(timeout=60))
        except queue.Empty:
            raise SystemExit("timeout reading the metadata header")
    meta = decode_meta(meta_frames, width, height)
    if not meta:
        raise SystemExit("metadata decode failed")
    M_ = int(meta.get('M', M))
    R_ = int(meta.get('R', R))
    k = int(meta.get('k', K))
    m = int(meta.get('m', M_PAR))
    B = int(meta.get('B', group_capacity(width, height, M_)))
    # geometry cross-check: the raw reader must match the sender's geometry,
    # otherwise every block grid is off
    if int(meta.get('w', -1)) != width or int(meta.get('h', -1)) != height:
        raise SystemExit(
            f"geometry mismatch: sender streams {meta.get('w')}x"
            f"{meta.get('h')}, this receiver expects {width}x{height} — "
            f"pass matching --width/--height")
    log.info("tunnel meta: k=%d m=%d B=%d (R=%d M=%d)", k, m, B, R_, M_)
    P = cauchy_matrix(k, m)

    conns = Conns(args.tcp_host, args.tcp_port, args.out, args.dump)
    conns.start()
    dm = Demuxer()
    seq_trk = SeqWrapTracker()

    def get_frame():
        while not stop.is_set():
            try:
                return q.get(timeout=5)
            except queue.Empty:
                if ff.poll() is not None:
                    try:
                        return q.get_nowait()
                    except queue.Empty:
                        return None
        return None

    # ---- group walker ---------------------------------------------------------
    buf = []          # raw frames covering [base, base+len(buf))
    base = 0
    stripe_no = -1
    stripe_recv = []  # length k+m; payload bytes or None
    repaired = 0
    total = 0
    idle_groups = 0
    last_activity = time.time()

    def get_upto(n):
        nonlocal last_activity
        while len(buf) < n:
            d = get_frame()
            if d is None:
                return False
            buf.append(d)
            last_activity = time.time()
        return True

    def compact(cursor):
        nonlocal base
        drop = cursor - base
        if drop >= R_:
            del buf[:drop]
            base += drop

    def try_decode(off):
        return decode_group(buf[off:off + R_], M_, width, height)

    def resync(from_off):
        for d in range(RESYNC_LIMIT):
            off = from_off + d
            if not get_upto(off + R_):
                return None
            h8, payload = try_decode(off)
            if h8 is None:
                continue
            seq, ok = verify_group(h8, payload)
            if ok:
                return h8, payload, off
        return None

    def is_eof(h8, payload):
        seq, ok = verify_group(h8, payload)
        return ok and seq == 0xFFFF and payload[:4] == EOF_PAYLOAD

    def deliver_rows(rows):
        for p in rows:
            for conn, data, fin in dm.feed(p.tobytes()):
                conns.deliver(conn, data, fin)
        conns.flush()

    def flush_stripe(no, allow_partial=False):
        nonlocal repaired
        recv = stripe_recv[:k + m]
        n_lost = sum(1 for x in recv if x is None)
        n_data = sum(1 for x in recv[:k] if x is not None)
        if n_lost > m:
            if allow_partial and n_data == k:
                pass  # data complete, only parity lost — no repair needed
            elif allow_partial:
                log.warning("stream end: stripe %d lost %d data group(s) "
                            "(unrecoverable)", no, k - n_data)
                return False
            else:
                raise SystemExit(f"stripe {no}: {n_lost} of {k + m} groups "
                                 f"lost (only {m} correctable) — aborting")
        data_rows = stripe_data(recv, k, P)
        if n_lost:
            repaired += n_lost
            log.info("stripe %d: repaired %d lost group(s)", no, n_lost)
        deliver_rows(data_rows)
        return True

    log.info("tunnel TCP serving on %s:%d", args.tcp_host, args.tcp_port)
    try:
        cursor = 0
        while True:
            h8 = payload = None
            off = None
            if get_upto(cursor + R_):
                h8, payload = try_decode(cursor)
                if h8 is not None:
                    seq, ok = verify_group(h8, payload)
                    if ok or is_eof(h8, payload):
                        off = cursor
                    else:
                        h8 = payload = None
            if h8 is None:
                r = resync(cursor)
                if r is None:
                    log.warning("stream ended (sender shut down)")
                    break
                h8, payload, off = r

            if is_eof(h8, payload):
                # The sender's "tunnel idle" sentinel: it keeps streaming
                # idle groups and accepts new clients, so we keep going too.
                # (A real shutdown ends the SRT stream itself.)
                log.info("tunnel idle sentinel after %d groups (%d repaired)",
                         total, repaired)
                idle_groups = 0
                cursor = off + R_
                compact(cursor)
                continue

            seq, _ = verify_group(h8, payload)
            g = seq_trk.true_index(seq)
            log.info("[walk] pos=%d seq=%d g=%d", off, seq, g)
            s = stripe_of(g, k, m)
            if s != stripe_no:
                if stripe_no >= 0:
                    flush_stripe(stripe_no)
                stripe_no = s
                stripe_recv = [None] * (k + m)
            idx = g - s * (k + m)
            if 0 <= idx < k + m and stripe_recv[idx] is None:
                stripe_recv[idx] = payload
            total += 1

            if payload == b'\x00' * B:
                idle_groups += 1
            else:
                idle_groups = 0

            if all(x is not None for x in stripe_recv):
                flush_stripe(stripe_no)

            cursor = off + R_
            compact(cursor)

            if (idle_groups >= QUIET_EXIT_GROUPS and not conns.active
                    and time.time() - last_activity > 10):
                log.info("stream went quiet (%d idle groups, no clients); "
                         "exiting", idle_groups)
                break
    except SystemExit:
        raise
    except Exception as e:
        log.error("receiver error: %s", e)
        raise
    finally:
        # flush a still-open stripe (EOF/stream end can arrive mid-stripe;
        # the data part may be complete even if parity was lost)
        if stripe_no >= 0:
            try:
                flush_stripe(stripe_no, allow_partial=True)
            except SystemExit:
                pass
        stop.set()
        try:
            ff.wait(timeout=5)
        except Exception:
            try:
                ff.terminate()
            except Exception:
                pass
        conns.flush()
        log.info("receiver stopped: %d groups, %d repaired", total, repaired)


def main():
    ap = argparse.ArgumentParser(
        description='bitcoder tunnel receiver: SRT video -> live TCP connections')
    ap.add_argument('--srt', default='srt://:9000',
                    help='SRT listen URL, e.g. srt://:9000 or srt://0.0.0.0:9000')
    ap.add_argument('--width', type=int, default=1280,
                    help='stream width (must match the sender)')
    ap.add_argument('--height', type=int, default=720,
                    help='stream height (must match the sender)')
    ap.add_argument('--tcp-host', default='0.0.0.0')
    ap.add_argument('--tcp-port', type=int, default=9001,
                    help='local TCP port tunnel clients connect to')
    ap.add_argument('--out', default='tunnel_out',
                    help='directory for --dump output')
    ap.add_argument('--dump', action='store_true',
                    help='also append every reconstructed byte to '
                         '<out>/tunnel_dump.bin (debug)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    run(args)


if __name__ == '__main__':
    main()
