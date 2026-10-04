#!/usr/bin/env python3
"""One-way capacity probe: client pushes N datagrams to a server that only
ACKs (no echo). Measures the true throughput of the 7030+FEC carrier, free
of bidirectional FIFO contention. This is the number the real VPN node gets
on its heavy direction (the light direction carries only ACKs)."""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vqic_video_echo import (client_cfg, server_cfg, make_cert, VQ,
                             VideoServer, W, H, FPS)
from fec_frame_channel import FecFrameChannel
from aioquic.quic.connection import QuicConnection

KB = 1024


async def main():
    mb = float(os.environ.get("PROBE_MB", "0.5"))
    n_bytes = int(mb * 1024 * 1024)
    chan = FecFrameChannel(W, H, fps=FPS)
    stop = asyncio.Event()
    stats = {'dtg': 0}
    tasks = []
    loop = asyncio.get_running_loop()

    def stream_handler(reader, writer):
        asyncio.ensure_future(sink(reader))

    async def sink(reader):
        try:
            while True:
                c = await reader.read(64 * KB)
                if not c:
                    break
        except Exception as e:
            print(f"  [sink] {type(e).__name__}: {e}", flush=True)

    cert, key = make_cert()
    server = VideoServer(chan, stats, stream_handler, cert, key)
    tasks.append(asyncio.ensure_future(server.pump(stop)))
    for s in ('a', 'b'):
        tasks.append(asyncio.ensure_future(chan.ticker(s, stop)))
        tasks.append(asyncio.ensure_future(chan.idle_flush(s, stop, idle=0.3)))

    quic = QuicConfiguration_client = client_cfg()
    quic_conn = QuicConfiguration_client and None  # placeholder to satisfy lint
    cfg = client_cfg()
    qc = QuicConnection(configuration=cfg)
    vq = VQ(qc, chan, 'a', stats)

    t_start = time.time()
    qc.connect(addr=b'peer', now=loop.time())
    vq.proto.transmit()
    tasks.append(asyncio.ensure_future(vq.pump(stop)))

    deadline = loop.time() + 30
    while not qc._handshake_complete and loop.time() < deadline:
        await asyncio.sleep(0.02)
    if not qc._handshake_complete:
        print("FAIL: handshake"); return 1
    hs = time.time() - t_start
    print(f"handshake {hs:.2f}s")

    sid = qc.get_next_available_stream_id()
    vq.proto._create_stream(sid)
    # send in 64 KB chunks to keep flow-control windows open
    off = 0
    chunk = 64 * KB
    while off < n_bytes:
        piece = os.urandom(min(chunk, n_bytes - off))
        qc.send_stream_data(sid, piece, end_stream=False)
        off += len(piece)
    qc.send_stream_data(sid, b"", end_stream=True)
    vq.proto.transmit()

    # client-side: watch how fast the server ACKs (send buffer drains)
    t0 = time.time()
    last = 0
    while len(qc._streams[sid].sender._buffer) > 0 and time.time() - t0 < 60:
        await asyncio.sleep(0.5)
        drained = n_bytes - len(qc._streams[sid].sender._buffer)
        if drained != last:
            el = time.time() - t0
            print(f"  [t={el:5.1f}s] drained {drained/KB:7.0f} KB "
                  f"= {drained/el/KB:6.1f} KB/s", flush=True)
            last = drained
    done = n_bytes - len(qc._streams[sid].sender._buffer)
    el = time.time() - t0
    print(f"one-way: {done/1024:.0f} KB in {el:.1f}s = {done/el/1024:.1f} KB/s")
    stop.set()
    for t in tasks:
        t.cancel()
    for t in tasks:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
