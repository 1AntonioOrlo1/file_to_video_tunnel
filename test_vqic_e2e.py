#!/usr/bin/env python3
"""End-to-end: two vqic_tunnel nodes over real SRT video links, a local TCP
echo service at the far end, and a client pushing N KB through the tunnel.

        Node A (client)                     Node B (server)
  local TCP :19101  <-> QUIC over video <->  local TCP dial :19102 (echo srv)
       |  ffmpeg-out: rawvideo->x264->SRT caller :19201
       |  ffmpeg-in : SRT listener :19200 -> rawvideo
       <=======================================>
       |  ffmpeg-in : SRT listener :19201 -> rawvideo
       |  ffmpeg-out: rawvideo->x264->SRT caller :19200

SRT is UDP. The working manual repro used plain 'latency=100' URLs; a
'timeout' on the LISTENER drops the connection a beat after the caller
connects, so it must stay out of the listener URL. The node supervises both
ffmpeg processes and retries the callers until the far listener is up
(startup race).

Hygiene (learned the hard way):
  * preflight kills leftovers from dead runs by scanning /proc (never a
    shell pkill pattern — that matches and kills our OWN command line);
  * the echo server is a separate process (echo_srv.py);
  * a SIGTERM handler runs the same cleanup as a normal exit, so `timeout`
    or Ctrl-C never orphans nodes that pin the SRT UDP ports.

Usage: python3 test_vqic_e2e.py [payload_kb] [timeout_s]   (default 1024 / 300)
"""
import os
import select
import signal
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
NODE = os.path.join(HERE, 'vqic_tunnel.py')
ECHO = os.path.join(HERE, 'echo_srv.py')
PAYLOAD_KB = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
TIMEOUT_S = int(sys.argv[2]) if len(sys.argv) > 2 else 300
W, H, FPS = (
    (3840, 2160, 30) if (len(sys.argv) > 3 and sys.argv[3] == '4k')
    else (1920, 1080, 30) if (len(sys.argv) > 3 and sys.argv[3] == '1080p')
    else (1280, 720, 30))
COPIES = int(sys.argv[4]) if len(sys.argv) > 4 else 2
CRF = int(sys.argv[5]) if len(sys.argv) > 5 else 0

SRT_LISTEN = 'mode=listener&latency=100'
SRT_CALLER = 'mode=caller&latency=100'

# VQIC_GRID=1 (env, inherited by both nodes): the ffmpeg<->node pipe carries
# the compact block grid (W/8 x H/8) instead of the full frame; the wire
# stays true WxH via ffmpeg scale filters. See vqic_tunnel.py VideoLink.
GRID = os.environ.get('VQIC_GRID', '0') == '1'
PW, PH = (W // 8, H // 8) if GRID else (W, H)
UP_VF = f'-vf scale={W}:{H}:flags=neighbor ' if GRID else ''
DOWN_VF = f'-vf format=rgb24,scale={PW}:{PH}:flags=neighbor ' if GRID else ''

PROCS = []
CLEANED = False


def x264_args():
    # -g 1: all-I frames. SRT is UDP; a lost P-frame corrupts the whole
    # chain until the next I-frame (every 30). With all-I, each frame
    # decodes independently, so one lost frame costs exactly that frame
    # (R=2 copy averaging recovers it). 0/255 content compresses well.
    # Encoder: x264 (default, CPU) or h264_nvenc (GPU, -qp 0 = lossless).
    # NVENC needs yuv420p input; x264 takes rgb24 directly. Cards 0-2
    # (CMP 50HX) have no NVENC at all — only the 2080 SUPER (card 3) can
    # encode, and only at resolutions whose VRAM buffers fit in its free
    # ~700 MB (1080p works, 4K does not).
    if os.environ.get('VQIC_ENCODER', 'x264') == 'nvenc':
        dev = os.environ.get('VQIC_NVENC_DEV', '3')
        return (f'-pix_fmt yuv420p -c:v h264_nvenc -gpu {dev} -qp 0 '
                f'-preset p1 -tune ull -g 1 -bf 0')
    return (f'-c:v libx264 -pix_fmt yuv420p -preset ultrafast '
            f'-tune zerolatency -crf {CRF} -g 1 -bf 0')


A_OUT = (f'-f rawvideo -s {PW}x{PH} -pix_fmt rgb24 -r {FPS} -i - '
         f'{UP_VF}{x264_args()} -f mpegts srt://127.0.0.1:19201?{SRT_CALLER}')
A_IN = (f"-i 'srt://127.0.0.1:19200?{SRT_LISTEN}' -map 0:v:0 "
        f'{DOWN_VF}-c:v rawvideo -pix_fmt rgb24 -f rawvideo -')
B_OUT = (f'-f rawvideo -s {PW}x{PH} -pix_fmt rgb24 -r {FPS} -i - '
         f'{UP_VF}{x264_args()} -f mpegts srt://127.0.0.1:19200?{SRT_CALLER}')
B_IN = (f"-i 'srt://127.0.0.1:19201?{SRT_LISTEN}' -map 0:v:0 "
        f'{DOWN_VF}-c:v rawvideo -pix_fmt rgb24 -f rawvideo -')


def _our_pids():
    """Pids of leftover vqic_tunnel/echo_srv/tunnel-ffmpeg processes from
    dead runs, found by scanning /proc cmdlines (never shell pkill: a
    pattern like 'srt://127.0.0.1:192' also matches THIS test's command
    line and kills our own shell)."""
    found = []
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            raw = open(f'/proc/{d}/cmdline', 'rb').read()
        except OSError:
            continue
        parts = [c.decode('utf-8', 'ignore') for c in raw.split(b'\0') if c]
        if not parts:
            continue
        c0 = parts[0]
        # the node/echo are launched as `python3 <script>.py`, so the script
        # name is parts[1] — checking c0 (the interpreter) never matched and
        # orphaned runs survived the preflight
        is_node = any(p.endswith(('vqic_tunnel.py', 'echo_srv.py'))
                      for p in parts)
        is_ff = (os.path.basename(c0) == 'ffmpeg'
                 and any(f'srt://127.0.0.1:{p}' in c for c in parts
                         for p in (19200, 19201)))
        if is_node or is_ff:
            found.append(int(d))
    return found


def preflight():
    for pid in _our_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    t0 = time.time()
    while time.time() - t0 < 20 and _our_pids():
        time.sleep(0.5)
    left = _our_pids()
    if left:
        print(f"FAIL: leftover processes still hold the ports: {left}")
        return False
    return True


def wait_port(port, what, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = socket.socket()
        s.settimeout(0.5)
        try:
            s.connect(('127.0.0.1', port))
            s.close()
            print(f"{what} up after {time.time()-t0:.1f} s", flush=True)
            return True
        except OSError:
            time.sleep(1)
    return False


def cleanup():
    global CLEANED
    if CLEANED:
        return
    CLEANED = True
    for p, name in PROCS:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    time.sleep(2)
    for p, name in PROCS:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            p.wait(timeout=2)
        except Exception:
            pass


def on_term(signum, frame):
    print("SIGTERM received: cleaning up", flush=True)
    cleanup()
    os._exit(130)


def main():
    signal.signal(signal.SIGTERM, on_term)
    if not preflight():
        return 1

    payload = os.urandom(PAYLOAD_KB << 10)
    t0 = time.time()
    logA = open(f'{HERE}/e2e_a.log', 'w')
    logB = open(f'{HERE}/e2e_b.log', 'w')
    echo_log = open(f'{HERE}/e2e_echo.log', 'w')

    # far-end local echo service (the "remote app" at Node B), own process
    PROCS.append((subprocess.Popen(
        [PY, ECHO, '19102'], stdout=echo_log, stderr=subprocess.STDOUT,
        cwd=HERE, start_new_session=True), 'echo'))
    if not wait_port(19102, 'echo server :19102'):
        print("FAIL: echo server never came up")
        cleanup()
        return 1

    # Node B first (it owns SRT listener :19201). start_new_session puts
    # each node (and its ffmpeg children) in its own process group.
    PROCS.append((subprocess.Popen(
        [PY, NODE, '--role', 'server',
         '--ffmpeg-out', B_OUT, '--ffmpeg-in', B_IN,
         '--tcp-cli-port', '19102',
         '--width', str(W), '--height', str(H), '--fps', str(FPS),
         '--copies', str(COPIES)],
        stdout=logB, stderr=subprocess.STDOUT, cwd=HERE,
        start_new_session=True), 'B'))
    time.sleep(3)

    PROCS.append((subprocess.Popen(
        [PY, NODE, '--role', 'client',
         '--ffmpeg-out', A_OUT, '--ffmpeg-in', A_IN,
         '--tcp-srv-port', '19101',
         '--width', str(W), '--height', str(H), '--fps', str(FPS),
         '--copies', str(COPIES)],
        stdout=logA, stderr=subprocess.STDOUT, cwd=HERE,
        start_new_session=True), 'A'))

    r = w = None
    for _ in range(180):
        time.sleep(1)
        s = socket.socket()
        s.settimeout(2)
        try:
            s.connect(('127.0.0.1', 19101))
            s.settimeout(None)      # 2 s was only for the connect retry loop
            r, w = s, None
            break
        except OSError:
            s.close()
    if r is None:
        print("FAIL: Node A local TCP :19101 never came up")
        cleanup()
        return 1
    print(f"local TCP up after {time.time()-t0:.1f} s", flush=True)

    # push the payload, read the echo (blocking socket; no asyncio needed).
    # Time from BEFORE sendall: node A reads+forwards concurrently with the
    # send, so the tunnel is already draining by the time sendall returns —
    # clocking only the echo read over-states the throughput.
    t1 = time.time()
    r.sendall(payload)
    got = b''
    first_byte = None
    while len(got) < len(payload):
        ready, _, _ = select.select([r], [], [], TIMEOUT_S)
        if not ready:
            print(f"TIMEOUT: got {len(got) >> 10}/{len(payload) >> 10} KB "
                  f"in {TIMEOUT_S} s", flush=True)
            break
        chunk = r.recv(65536)
        if not chunk:
            print(f"CONN CLOSED by peer at {len(got) >> 10} KB", flush=True)
            break
        if first_byte is None and chunk:
            first_byte = time.time() - t1
        got += chunk
        if len(got) % (128 << 10) == 0:
            el = time.time() - t1
            print(f"  [t={el:6.1f}s] {len(got) >> 10} KB "
                  f"= {len(got)/el/1024:6.1f} KB/s", flush=True)

    el = time.time() - t1
    if got == payload:
        fb = f", first echo byte at {first_byte*1000:.0f} ms" if first_byte else ""
        print(f"RESULT: PASS — {len(got) >> 10} KB in {el:.1f}s "
              f"= {len(got)/el/1024:.1f} KB/s{fb}", flush=True)
        rc = 0
    else:
        print(f"RESULT: FAIL — got {len(got) >> 10} / {len(payload) >> 10} KB "
              f"in {el:.1f}s", flush=True)
        rc = 1
    r.close()
    time.sleep(1)
    cleanup()
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        cleanup()
        sys.exit(130)
