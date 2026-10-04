#!/usr/bin/env python3
"""Forced frame-loss test for the vqic tunnel.

Physically REMOVES frames from the video stream (the node's send clock skips
the write with probability VQIC_DROP_PCT) and proves QUIC retransmits the
data of the missing frames. This is the "a frame disappears from the video"
path — not a CRC-corrupted one (that path is what CRF23 already exercises
every e2e run).

Arguments: PAYLOAD_KB TIMEOUT_S [drop_pct]
  default: 512 KB, 150 s, 40% drop

Assertions (all must hold):
  1. payload arrives BYTE-EXACT through the lossy video (QUIC retransmitted
     every dropped frame's datagrams);
  2. frames were really removed from the video: drop_forced > 0;
  3. the frame count in the video changed: data_written < sent*copies
     (sent*copies = every data frame the codec produced for the video).

720p grid, R=2, CRF23 — the same carrier as the throughput matrix.
"""
import os, re, signal, socket, subprocess, sys, time, select

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
NODE = os.path.join(HERE, 'vqic_tunnel.py')
ECHO = os.path.join(HERE, 'echo_srv.py')

PAYLOAD_KB = int(sys.argv[1]) if len(sys.argv) > 1 else 512
TIMEOUT_S = int(sys.argv[2]) if len(sys.argv) > 2 else 150
DROP_PCT = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0

W, H, FPS, COPIES, CRF = 1280, 720, 30, 2, 23
os.environ['VQIC_GRID'] = '1'
os.environ['VQIC_DROP_PCT'] = f'{DROP_PCT:g}'
os.environ['VQIC_DROP_DTA'] = '1'
os.environ['VQIC_DROP_SEED'] = '7'

SRT_LISTEN = 'mode=listener&latency=100'
SRT_CALLER = 'mode=caller&latency=100'
PW, PH = W // 8, H // 8
UP_VF = f'-vf scale={W}:{H}:flags=neighbor '
DOWN_VF = f'-vf format=rgb24,scale={PW}:{PH}:flags=neighbor '
X264 = (f'-c:v libx264 -pix_fmt yuv420p -preset ultrafast -tune zerolatency '
        f'-crf {CRF} -g 1 -bf 0')
A_OUT = (f'-f rawvideo -s {PW}x{PH} -pix_fmt rgb24 -r {FPS} -i - '
         f'{UP_VF}{X264} -f mpegts srt://127.0.0.1:19201?{SRT_CALLER}')
A_IN = (f"-i 'srt://127.0.0.1:19200?{SRT_LISTEN}' -map 0:v:0 "
        f'{DOWN_VF}-c:v rawvideo -pix_fmt rgb24 -f rawvideo -')
B_OUT = (f'-f rawvideo -s {PW}x{PH} -pix_fmt rgb24 -r {FPS} -i - '
         f'{UP_VF}{X264} -f mpegts srt://127.0.0.1:19200?{SRT_CALLER}')
B_IN = (f"-i 'srt://127.0.0.1:19201?{SRT_LISTEN}' -map 0:v:0 "
        f'{DOWN_VF}-c:v rawvideo -pix_fmt rgb24 -f rawvideo -')

PROCS = []
CLEANED = False


def _our_pids():
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
    return not _our_pids()


def wait_port(port, what, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = socket.socket(); s.settimeout(0.5)
        try:
            s.connect(('127.0.0.1', port)); s.close()
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


def parse_stats(path):
    """Last 'stats:' line of a node log -> dict of its counters."""
    last = None
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if 'stats:' in line:
                last = line
    if not last:
        return None
    d = {}
    for k, v in re.findall(r'(\w+)=(\d+)', last):
        d[k] = int(v)
    return d


def on_term(signum, frame):
    cleanup()
    os._exit(130)


def main():
    signal.signal(signal.SIGTERM, on_term)
    if not preflight():
        print("FAIL: leftover processes hold the ports")
        return 1

    payload = os.urandom(PAYLOAD_KB << 10)
    t0 = time.time()
    logA = open(f'{HERE}/loss_a.log', 'w')
    logB = open(f'{HERE}/loss_b.log', 'w')
    echo_log = open(f'{HERE}/loss_echo.log', 'w')

    PROCS.append((subprocess.Popen(
        [PY, ECHO, '19102'], stdout=echo_log, stderr=subprocess.STDOUT,
        cwd=HERE, start_new_session=True), 'echo'))
    if not wait_port(19102, 'echo server :19102'):
        print("FAIL: echo server never came up"); cleanup(); return 1

    PROCS.append((subprocess.Popen(
        [PY, NODE, '--role', 'server', '--ffmpeg-out', B_OUT, '--ffmpeg-in', B_IN,
         '--tcp-cli-port', '19102', '--width', str(W), '--height', str(H),
         '--fps', str(FPS), '--copies', str(COPIES)],
        stdout=logB, stderr=subprocess.STDOUT, cwd=HERE,
        start_new_session=True), 'B'))
    time.sleep(3)

    PROCS.append((subprocess.Popen(
        [PY, NODE, '--role', 'client', '--ffmpeg-out', A_OUT, '--ffmpeg-in', A_IN,
         '--tcp-srv-port', '19101', '--width', str(W), '--height', str(H),
         '--fps', str(FPS), '--copies', str(COPIES)],
        stdout=logA, stderr=subprocess.STDOUT, cwd=HERE,
        start_new_session=True), 'A'))

    r = None
    for _ in range(180):
        time.sleep(1)
        s = socket.socket(); s.settimeout(2)
        try:
            s.connect(('127.0.0.1', 19101)); s.settimeout(None)
            r = s
            break
        except OSError:
            s.close()
    if r is None:
        print("FAIL: Node A local TCP :19101 never came up"); cleanup(); return 1
    print(f"local TCP up after {time.time()-t0:.1f} s "
          f"(drop {DROP_PCT:g}% on BOTH nodes, seed 7)", flush=True)

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
            print(f"CONN CLOSED at {len(got) >> 10} KB", flush=True)
            break
        if first_byte is None and chunk:
            first_byte = time.time() - t1
        got += chunk
        if len(got) % (64 << 10) == 0:
            el = time.time() - t1
            print(f"  [t={el:6.1f}s] {len(got) >> 10} KB "
                  f"= {len(got)/el/1024:6.1f} KB/s", flush=True)

    el = time.time() - t1
    exact = (got == payload)
    r.close()
    time.sleep(1)
    cleanup()

    # --- assertions on the frame accounting (node A = the data sender) ---
    st = parse_stats(f'{HERE}/loss_a.log')
    print(f"\ntransfer: {len(got) >> 10}/{len(payload) >> 10} KB in {el:.1f}s, "
          f"byte-exact={exact}, first echo byte "
          f"{first_byte*1000 if first_byte else float('nan'):.0f} ms",
          flush=True)
    ok = exact
    if st is None:
        print("FAIL: no stats line in loss_a.log"); return 1
    print(f"node A: data_written={st.get('data_w')} "
          f"drop_forced={st.get('drop_forced')} sent={st.get('sent')} "
          f"render_drop={st.get('render_drop')} clock_drop={st.get('clock_drop')}",
          flush=True)
    stb = parse_stats(f'{HERE}/loss_b.log')
    if stb:
        print(f"node B: data_written={stb.get('data_w')} "
              f"drop_forced={stb.get('drop_forced')} sent={stb.get('sent')}",
              flush=True)

    a_sent = st.get('sent', 0)
    a_written = st.get('data_w', 0)
    a_forced = st.get('drop_forced', 0)
    a_up = st.get('up_dtg', 0)
    # minimum distinct up-path stream datagrams for a clean (no-loss) send
    distinct_min = (len(payload) + 1349) // 1350
    retrans_factor = a_up / max(1, distinct_min)

    # 2. frames were really removed from the video
    if a_forced <= 0:
        print("FAIL: no frames were dropped — loss was not injected")
        ok = False
    # 3. the frame count in the video changed (written < produced)
    produced = a_sent * COPIES
    if a_written >= produced:
        print(f"FAIL: data_written {a_written} >= produced {produced} — "
              f"the video frame count did not shrink")
        ok = False

    b_forced = (stb or {}).get('drop_forced', 0)
    print(f"\nvideo A: {a_written}/{produced} data frames written "
          f"({a_forced} physically removed = "
          f"{100*a_forced/max(1,produced):.1f}% of produced; produced = "
          f"{a_sent} frames x R{COPIES} copies)", flush=True)
    print(f"QUIC A: {a_up} datagrams sent for ~{distinct_min} minimum "
          f"= {retrans_factor:.1f}x (retransmit factor)", flush=True)
    if ok:
        print(f"RESULT: PASS — {len(got) >> 10} KB byte-exact through "
              f"{DROP_PCT:g}% physically-removed video frames "
              f"({a_forced} dropped out, {b_forced} on the echo's way "
              f"back; QUIC retransmitted the {a_up - distinct_min:+} "
              f"extra datagrams)", flush=True)
        return 0
    print("RESULT: FAIL", flush=True)
    return 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        cleanup()
        sys.exit(130)
