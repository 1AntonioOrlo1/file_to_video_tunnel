#!/usr/bin/env python3
"""bitcoder-tunnel sender: live TCP connections -> SRT video stream.

This is the "internet in, video out" side of the tunnel. It accepts local
TCP connections, multiplexes their byte streams into fixed-size payloads,
FEC-protects them in stripes (k data + m parity groups, GF(256) Cauchy MDS),
renders every group as R identical rgb24 frames with the 8-corner color
palette (3 bits per MxM block, 0/255 levels), and encodes the whole thing
with ffmpeg (libx264, zerolatency, mpegts over SRT) in realtime.

Frames are fed to ffmpeg by a dedicated PACED writer: exactly fps frames
per second. The SRT link only tolerates a ~130 ms buffer, so feeding the
encoder faster than the declared fps overflows the send window, the
receiver's SRT queue drops packets, and the whole stream tears down —
the pacing is what keeps the stream alive. If we fall behind (a render
spike) we do NOT burst to catch up: we resume at the steady rate.

The SRT stream stays alive the whole time this process runs (idle stripes
keep it moving when no one is using the tunnel). An EOF sentinel in the
stream means "tunnel idle"; a stream end (this process shutting down)
means the real goodbye.

Usage:
  python3 tunnel_send.py --srt srt://<receiver-host>:9000 \
                         [--tcp-port 9000] [--width 1280] [--height 720] \
                         [--fps 30] [--k 8] [--m 2] [--crf 23]
"""

import argparse
import logging
import queue
import socket
import subprocess
import sys
import threading
import time

import numpy as np

from tunnel_core import (K, M, M_PAR, R, R_META, cauchy_matrix,
                         group_capacity, make_eof_group, parity_rows,
                         render_group, render_meta, throughput_bps)
from mux import Muxer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('tunnel-send')

FLUSH_IDLE_AFTER = 0.4   # s a partial stripe may sit (clients present)
IDLE_EOF_AFTER = 1.5     # s of idleness before the "tunnel idle" EOF
FRAME_Q_MAX = 20         # ~0.7 s of video in the queue; beyond that the
                         # main loop backpressures (clients' sockets buffer)


class Client:
    __slots__ = ('sock', 'addr', 'cid')

    def __init__(self, sock, addr, cid):
        self.sock = sock
        self.addr = addr
        self.cid = cid

    def read_some(self):
        """Returns bytes, b'' on clean peer close, None on timeout (alive)."""
        try:
            data = self.sock.recv(262144)
        except (socket.timeout, TimeoutError):
            return None
        except (ConnectionError, OSError):
            return b''
        return data


def stripe_frames(stripe_no, data_payloads, P, width, height):
    """Render one full stripe: k data + m parity groups, each R copies.
    Returns the list of R*(k+m) raw rgb24 frame bytes."""
    k = len(data_payloads)
    rows = np.stack([np.frombuffer(p, dtype=np.uint8) for p in data_payloads])
    parity = parity_rows(rows, P)
    seq0 = stripe_no * (k + len(parity))
    frames = []
    for i, payload in enumerate(data_payloads + [p.tobytes() for p in parity]):
        f = render_group(payload, seq0 + i, M, width, height)
        frames.extend([f] * R)
    return frames


def paced_writer(frame_q, ff, fps):
    """Write frames to ffmpeg's stdin at exactly `fps` frames/second.

    The heart of realtime stability: the SRT send window is tiny (~130 ms),
    so the encoder must never see a burst. If we fall behind we do NOT try
    to catch up (that would burst the link); we resume at the steady rate."""
    next_t = time.time()
    try:
        while True:
            item = frame_q.get()
            if item is None:
                return
            for f in item:
                ff.stdin.write(f)
                next_t += 1.0 / fps
                now = time.time()
                if now < next_t:
                    time.sleep(next_t - now)
                else:
                    next_t = now
    except (BrokenPipeError, OSError):
        log.error("ffmpeg stdin closed early")


def run(args):
    B = group_capacity(args.width, args.height, M)
    log.info("geometry %dx%d, %d B/group, ~%.0f KB/s payload at full stripes",
             args.width, args.height, B,
             throughput_bps(args.width, args.height, args.fps,
                            args.k, args.m) / 1024)
    P = cauchy_matrix(args.k, args.m)

    # ---- local TCP fan-in ---------------------------------------------------
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.tcp_host, args.tcp_port))
    srv.listen(16)
    srv.settimeout(0.1)
    log.info("tunnel TCP listening on %s:%d", args.tcp_host, args.tcp_port)

    clients = {}
    next_cid = [1]
    mux = Muxer(B)

    # ---- ffmpeg encoder to SRT ----------------------------------------------
    # SRT TRANSPORT (URL-based): in ffmpeg '-f srt' means SRT SUBTITLES, a
    # different thing. The real SRT protocol is selected by the srt:// URL
    # with an explicit container (-f mpegts). EXPLICIT caller mode: the
    # receiver listens, we dial it.
    srt_output = args.srt if 'mode=' in args.srt else args.srt + '?mode=caller'
    ff = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{args.width}x{args.height}', '-pix_fmt', 'rgb24',
        '-r', str(args.fps), '-i', '-',
        '-map', '0:v:0',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast', '-tune', 'zerolatency',
        '-crf', str(args.crf),
        '-g', str(args.fps), '-keyint_min', str(args.fps),
        '-bf', '0',
        '-f', 'mpegts', srt_output,
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
       stderr=sys.stderr)
    log.info("ffmpeg started, SRT target: %s", srt_output)

    # ---- metadata header ------------------------------------------------------
    meta_bytes = render_meta({
        'fn': 'live-tunnel', 'w': args.width, 'h': args.height,
        'M': M, 'R': R, 'k': args.k, 'm': args.m, 'B': B,
        'fps': args.fps,
    }, args.width, args.height)
    for _ in range(R_META):
        ff.stdin.write(meta_bytes)
    log.info("metadata header sent (%d copies)", R_META)

    # ---- paced frame writer ----------------------------------------------------
    frame_q = queue.Queue()
    writer_t = threading.Thread(target=paced_writer,
                                args=(frame_q, ff, args.fps), daemon=True)
    writer_t.start()

    def push(frames):
        while frame_q.qsize() > FRAME_Q_MAX:
            time.sleep(0.005)
        frame_q.put(frames)

    stripe = []          # current stripe's data payloads (< k until flushed)
    stripe_no = 0
    stripe_open_at = time.time()
    eof_frames = [make_eof_group(args.width, args.height)] * R
    idle_eof_sent = False
    last_idle_emit = 0.0
    last_log = time.time()
    # one idle stripe = (k+m)*R frames = (k+m)*R/fps seconds of video
    idle_period = (args.k + args.m) * R / args.fps

    def emit_stripe(stripe_data):
        nonlocal stripe_no
        push(stripe_frames(stripe_no, stripe_data, P,
                           args.width, args.height))
        stripe_no += 1

    def emit_idle_stripe():
        emit_stripe([b'\x00' * B] * args.k)

    def emit_eof(tag):
        push(eof_frames)
        log.info("EOF sentinel sent (%s)", tag)

    def client_gone(cid, reason):
        c = clients.pop(cid, None)
        if c is None:
            return
        try:
            c.sock.close()
        except OSError:
            pass
        # mark this connection finished so the receiver closes its side
        mux.feed(cid, b'', fin=True)
        log.info("client %d closed (%s)", cid, reason)

    def flush_partial_stripe():
        """Pad the open stripe (and any trailing mux data) to a full stripe
        and emit it. Used on stalls and before EOF sentinels so nothing is
        lost."""
        nonlocal stripe, stripe_open_at
        if mux.pending():
            stripe.append(mux.flush_partial())
        if stripe:
            while len(stripe) < args.k:
                stripe.append(b'\x00' * B)
            emit_stripe(stripe)
            stripe = []
            stripe_open_at = time.time()

    try:
        while True:
            # 1. accept new connections (drain their first burst at once)
            while True:
                try:
                    sock, addr = srv.accept()
                except (socket.timeout, TimeoutError, OSError):
                    break
                if next_cid[0] > 254:
                    log.warning("connection limit reached, refusing %s", addr)
                    sock.close()
                    continue
                c = Client(sock, addr, next_cid[0])
                next_cid[0] += 1
                clients[c.cid] = c
                sock.settimeout(0.02)
                log.info("client %d connected: %s", c.cid, addr)
                burst = c.read_some()
                if burst:
                    mux.feed(c.cid, burst)

            # 2. read from every client
            for c in list(clients.values()):
                data = c.read_some()
                if data is None:
                    continue              # timeout: still alive, no data
                if data:
                    mux.feed(c.cid, data)
                else:
                    client_gone(c.cid, 'peer closed')

            # 3. move complete groups from the mux into the stripe
            while mux.pending() >= B and len(stripe) < args.k:
                stripe.append(mux.next_group())

            # 4. flush a full stripe
            if len(stripe) == args.k:
                emit_stripe(stripe)
                stripe = []
                stripe_open_at = time.time()

            now = time.time()
            if clients:
                # 5. partial stripe with clients: pad after a short stall so
                #    the video never lags behind the data
                if (stripe and mux.pending() == 0
                        and now - stripe_open_at > FLUSH_IDLE_AFTER):
                    flush_partial_stripe()
                if not stripe:
                    time.sleep(0.003)
            else:
                # 6. no clients: keep the SRT link alive with idle stripes at
                #    the video rate (one full stripe per idle_period); an
                #    EOF sentinel tells the receiver "tunnel idle" (a stream
                #    END means real shutdown)
                if stripe or mux.pending():
                    flush_partial_stripe()
                if now - last_idle_emit >= idle_period:
                    emit_idle_stripe()
                    last_idle_emit = now
                if not idle_eof_sent and now - stripe_open_at > IDLE_EOF_AFTER:
                    emit_eof('tunnel idle')
                    idle_eof_sent = True
                time.sleep(0.005)

            if now - last_log > 5:
                last_log = now
                log.info("streaming: %d groups sent, %d client(s), "
                         "queue=%d", stripe_no, len(clients),
                         frame_q.qsize())

    except (BrokenPipeError, OSError) as e:
        log.error("error in main loop: %s", e)
    except KeyboardInterrupt:
        log.info("interrupted; shutting down")
    finally:
        try:
            flush_partial_stripe()
            emit_eof('shutdown')
            frame_q.put(None)
            writer_t.join(timeout=120)
            ff.stdin.close()
            ff.wait(timeout=10)
        except Exception:
            try:
                ff.terminate()
                ff.wait(timeout=5)
            except Exception:
                pass
        for c in clients.values():
            try:
                c.sock.close()
            except OSError:
                pass
        srv.close()
        log.info("sender stopped after %d groups", stripe_no)


def main():
    ap = argparse.ArgumentParser(
        description='bitcoder tunnel sender: live TCP connections -> SRT video')
    ap.add_argument('--tcp-host', default='0.0.0.0')
    ap.add_argument('--tcp-port', type=int, default=9000)
    ap.add_argument('--srt', default='srt://127.0.0.1:9000',
                    help='SRT target URL (the receiver listens there)')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--k', type=int, default=K, help='data groups per stripe')
    ap.add_argument('--m', type=int, default=M_PAR,
                    help='parity groups per stripe (repairs up to m lost)')
    ap.add_argument('--crf', type=int, default=23)
    args = ap.parse_args()
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
