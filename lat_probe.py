#!/usr/bin/env python3
"""Clean latency probe for the vqic tunnel.

The e2e client measures first-echo-byte AFTER a blocking sendall, which
contaminates the number. This probe instead:
  * launches echo + node B + node A (grid mode, 720p, CRF23, R=2) — same as
    test_vqic_e2e.py;
  * after the tunnel is up, sends N SMALL distinct probes with a quiet gap
    and times each probe's first echo byte (sendall of a few KB is instant
    over loopback, so the first-echo time IS the tunnel RTT);
  * reports min/median/mean so one warmup outlier can't masquerade as the
    steady-state figure.
Process hygiene: preflight kills leftovers by /proc scan (explicit PIDs),
never a shell pkill pattern.
"""
import os, signal, socket, subprocess, sys, time, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
NODE = os.path.join(HERE, 'vqic_tunnel.py')
ECHO = os.path.join(HERE, 'echo_srv.py')

W, H, FPS = 1280, 720, int(os.environ.get('LP_FPS', '30'))
COPIES = int(os.environ.get('LP_COPIES', '2'))
CRF = 23
SRT_LAT = os.environ.get('LP_SRT_LAT', '100')
# Low-delay flags on both mux and demux (nobuffer + low_delay) — env to A/B
EXTRA = os.environ.get('LP_FFLAGS', '')
os.environ['VQIC_GRID'] = '1'
os.environ['VQIC_TRACE'] = os.environ.get('VQIC_TRACE', '0')

SRT_LISTEN = f'mode=listener&latency={SRT_LAT}'
SRT_CALLER = f'mode=caller&latency={SRT_LAT}'
PW, PH = W // 8, H // 8
UP_VF = f'-vf scale={W}:{H}:flags=neighbor '
DOWN_VF = f'-vf format=rgb24,scale={PW}:{PH}:flags=neighbor '
X264 = (f'-c:v libx264 -pix_fmt yuv420p -preset ultrafast -tune zerolatency '
        f'-crf {CRF} -g 1 -bf 0 {EXTRA}')
A_OUT = (f'-f rawvideo -s {PW}x{PH} -pix_fmt rgb24 -r {FPS} {EXTRA} -i - '
         f'{UP_VF}{X264} -f mpegts srt://127.0.0.1:19201?{SRT_CALLER}')
A_IN = (f"{EXTRA}-i 'srt://127.0.0.1:19200?{SRT_LISTEN}' -map 0:v:0 "
        f'{DOWN_VF}-c:v rawvideo -pix_fmt rgb24 -f rawvideo -')
B_OUT = (f'-f rawvideo -s {PW}x{PH} -pix_fmt rgb24 -r {FPS} {EXTRA} -i - '
         f'{UP_VF}{X264} -f mpegts srt://127.0.0.1:19200?{SRT_CALLER}')
B_IN = (f"{EXTRA}-i 'srt://127.0.0.1:19201?{SRT_LISTEN}' -map 0:v:0 "
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
        is_node = any(p.endswith(('vqic_tunnel.py', 'echo_srv.py')) for p in parts)
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


def wait_port(port, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = socket.socket(); s.settimeout(0.5)
        try:
            s.connect(('127.0.0.1', port)); s.close(); return True
        except OSError:
            time.sleep(1)
    return False


def cleanup():
    global CLEANED
    if CLEANED:
        return
    CLEANED = True
    for p in PROCS:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            pass
    time.sleep(2)
    for p in PROCS:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass


def main():
    if not preflight():
        print('FAIL: leftovers'); return 1
    logs = [open(f'{HERE}/lat_{n}.log', 'w') for n in ('echo', 'b', 'a')]
    PROCS.append(subprocess.Popen([PY, ECHO, '19102'], stdout=logs[0],
                                  stderr=subprocess.STDOUT, cwd=HERE,
                                  start_new_session=True))
    if not wait_port(19102):
        print('FAIL: echo'); cleanup(); return 1
    PROCS.append(subprocess.Popen(
        [PY, NODE, '--role', 'server', '--ffmpeg-out', B_OUT, '--ffmpeg-in', B_IN,
         '--tcp-cli-port', '19102', '--width', str(W), '--height', str(H),
         '--fps', str(FPS), '--copies', str(COPIES)],
        stdout=logs[1], stderr=subprocess.STDOUT, cwd=HERE, start_new_session=True))
    time.sleep(3)
    PROCS.append(subprocess.Popen(
        [PY, NODE, '--role', 'client', '--ffmpeg-out', A_OUT, '--ffmpeg-in', A_IN,
         '--tcp-srv-port', '19101', '--width', str(W), '--height', str(H),
         '--fps', str(FPS), '--copies', str(COPIES)],
        stdout=logs[2], stderr=subprocess.STDOUT, cwd=HERE, start_new_session=True))
    r = None
    for _ in range(90):
        time.sleep(1)
        s = socket.socket(); s.settimeout(2)
        try:
            s.connect(('127.0.0.1', 19101)); s.settimeout(None); r = s; break
        except OSError:
            s.close()
    if r is None:
        print('FAIL: node A local TCP'); cleanup(); return 1
    print(f'tunnel up after {time.time():.0f}s (grid 720p R={COPIES} CRF{CRF})')

    # --- latency probes: small distinct payload, timed first-echo-byte ---
    N = 8
    warm = 2
    lat = []
    for i in range(N):
        probe = os.urandom(2048)
        r.sendall(probe)
        t0 = time.time()
        got = b''
        while len(got) < len(probe):
            ready, _, _ = __import__('select').select([r], [], [], 30)
            if not ready:
                print(f'probe {i}: TIMEOUT after {time.time()-t0:.2f}s')
                break
            c = r.recv(65536)
            if not c:
                print(f'probe {i}: closed'); break
            got += c
        el = time.time() - t0
        ok = (got == probe)
        lat.append(el)
        print(f'  probe {i}: {el*1000:8.1f} ms  exact={ok}', flush=True)
        time.sleep(0.3)   # quiet gap: let the pipeline settle between probes

    body = lat[warm:]
    print(f'\nsteady-state (drop {warm} warmup): '
          f'min={min(body)*1000:.0f} ms  median={statistics.median(body)*1000:.0f} ms  '
          f'mean={sum(body)/len(body)*1000:.0f} ms', flush=True)
    r.close()
    time.sleep(1)
    cleanup()
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        cleanup()
