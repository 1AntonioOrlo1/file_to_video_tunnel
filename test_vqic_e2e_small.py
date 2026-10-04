#!/usr/bin/env python3
"""End-to-end: two vqic_tunnel nodes over real SRT video links, a local TCP
echo service at the far end, and a client pushing 1 MB through the tunnel.

        Node A (client)                     Node B (server)
  local TCP :19101  <-> QUIC over video <->  local TCP dial :19102 (echo srv)
       |  ffmpeg-out: rawvideo->x264->SRT caller :19201
       |  ffmpeg-in : SRT listener :19200 -> rawvideo
       <=======================================>
       |  ffmpeg-in : SRT listener :19201 -> rawvideo
       |  ffmpeg-out: rawvideo->x264->SRT caller :19200

SRT is UDP. The working manual repro used plain 'latency=100' URLs; a 'timeout'
on the LISTENER drops the connection a beat after the caller connects, so it
must stay out of the listener URL. The node supervises both ffmpeg processes
and retries the callers until the far listener is up (startup race).
"""
import asyncio
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
NODE = os.path.join(HERE, 'vqic_tunnel.py')
W, H, FPS = 1280, 720, 30
CRF = 0   # the proven 7030 preset: x264 crf 0 on pure-gray 0/255 frames

SRT_LISTEN = 'mode=listener&latency=100'
SRT_CALLER = 'mode=caller&latency=100'


def x264_args():
    return (f'-c:v libx264 -pix_fmt yuv420p -preset ultrafast '
            f'-tune zerolatency -crf {CRF} -g {FPS} -keyint_min {FPS} -bf 0')


A_OUT = (f'-f rawvideo -s {W}x{H} -pix_fmt rgb24 -r {FPS} -i - '
         f'{x264_args()} -f mpegts srt://127.0.0.1:19201?{SRT_CALLER}')
A_IN = (f'-i \'srt://127.0.0.1:19200?{SRT_LISTEN}\' -map 0:v:0 '
        f'-c:v rawvideo -pix_fmt rgb24 -f rawvideo -')
B_OUT = (f'-f rawvideo -s {W}x{H} -pix_fmt rgb24 -r {FPS} -i - '
         f'{x264_args()} -f mpegts srt://127.0.0.1:19200?{SRT_CALLER}')
B_IN = (f'-i \'srt://127.0.0.1:19201?{SRT_LISTEN}\' -map 0:v:0 '
        f'-c:v rawvideo -pix_fmt rgb24 -f rawvideo -')


async def echo_server(port):
    async def conn(r, w):
        try:
            while True:
                d = await r.read(65536)
                if not d:
                    break
                w.write(d)
                await w.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            w.close()
    srv = await asyncio.start_server(conn, '127.0.0.1', port)
    async with srv:
        await srv.serve_forever()


async def main():
    payload = os.urandom(32 << 10)
    t0 = time.time()
    logA, logB = open(f'{HERE}/e2e_a.log', 'w'), open(f'{HERE}/e2e_b.log', 'w')
    procs = []

    # far-end local echo service (the "remote app" at Node B)
    echo_task = asyncio.ensure_future(echo_server(19102))

    # Node B first (it owns SRT listener :19201). start_new_session puts
    # each node (and its ffmpeg children) in its own process group so the
    # cleanup can kill the whole tree, not just the node.
    procs.append((subprocess.Popen(
        [PY, NODE, '--role', 'server',
         '--ffmpeg-out', B_OUT, '--ffmpeg-in', B_IN,
         '--tcp-cli-port', '19102',
         '--width', str(W), '--height', str(H), '--fps', str(FPS)],
        stdout=logB, stderr=subprocess.STDOUT, cwd=HERE,
        start_new_session=True), 'B'))
    await asyncio.sleep(3)

    procs.append((subprocess.Popen(
        [PY, NODE, '--role', 'client',
         '--ffmpeg-out', A_OUT, '--ffmpeg-in', A_IN,
         '--tcp-srv-port', '19101',
         '--width', str(W), '--height', str(H), '--fps', str(FPS)],
        stdout=logA, stderr=subprocess.STDOUT, cwd=HERE,
        start_new_session=True), 'A'))

    # wait for Node A's local TCP to come up (handshake over video takes a bit)
    r = w = None
    for _ in range(180):
        await asyncio.sleep(1)
        try:
            r, w = await asyncio.wait_for(
                asyncio.open_connection('127.0.0.1', 19101), timeout=2)
            break
        except (OSError, TimeoutError):
            r = w = None
    if r is None:
        print("FAIL: Node A local TCP :19101 never came up")
        cleanup(procs, echo_task)
        return 1
    print(f"local TCP up after {time.time()-t0:.1f} s")

    # push the payload, read the echo
    w.write(payload)
    await w.drain()
    got = b''
    t1 = time.time()
    while len(got) < len(payload):
        chunk = await asyncio.wait_for(r.read(65536), timeout=120)
        if not chunk:
            break
        got += chunk
        if len(got) % (128 << 10) == 0:
            el = time.time() - t1
            print(f"  [t={el:6.1f}s] {len(got) >> 10} KB "
                  f"= {len(got)/el/1024:6.1f} KB/s", flush=True)

    ok = got == payload
    el = time.time() - t1
    print(f"RESULT: {'PASS' if ok else 'FAIL'} — "
          f"{len(got) >> 10} KB in {el:.1f}s = {len(got)/el/1024:.1f} KB/s")
    r.close()
    await asyncio.sleep(1)
    cleanup(procs, echo_task)
    return 0 if ok else 1


def cleanup(procs, echo_task):
    """Kill each node's whole process group (node + its ffmpeg children)."""
    for p, name in procs:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    time.sleep(2)
    for p, name in procs:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            p.wait(timeout=2)
        except Exception:
            pass
    echo_task.cancel()


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        pass
