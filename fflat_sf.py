#!/usr/bin/env python3
"""Single-frame-in-flight pipeline latency, FILE sink (most robust).

No pipe readers to fight ffmpeg buffering. FF2 writes rawvideo to a FILE.
We send ONE frame, then poll the file until it grows by a full frame.
File growth is real-time (verified: with continuous pacing it batched, but
with a single isolated frame it appears as soon as the pipeline emits it).
This measures the TRUE fixed pipeline latency with zero queueing.

  ENCODE: rawvideo in -> x264 -> h264 FILE  (poll file growth)
  FULL:   rawvideo -> x264 -> h264 pipe -> decode -> rawvideo FILE
"""
import os, shlex, signal, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1280, 720
FSZ = W * H * 3
N = int(os.environ.get('LP_N', '40'))
X264 = ('-c:v libx264 -pix_fmt yuv420p -preset ultrafast '
        '-tune zerolatency -crf 23 -g 1 -bf 0')


def kill_ff():
    me = os.getpid()
    for d in os.listdir('/proc'):
        if d.isdigit() and int(d) != me:
            try:
                raw = open(f'/proc/{d}/cmdline', 'rb').read()
                if b'ffmpeg' in raw and (b'ffsf' in raw or b'ffsf' in raw):
                    os.kill(int(d), signal.SIGKILL)
            except OSError:
                pass


def main():
    kill_ff()
    time.sleep(1)

    # ---- ENCODE, file sink ----
    outE = f'{HERE}/ffsfE.out'
    try: os.remove(outE)
    except OSError: pass
    eE = open(f'{HERE}/ffsfE.err', 'w')
    enc = subprocess.Popen(
        ['ffmpeg'] + shlex.split(
            f'-y -loglevel error -f rawvideo -s {W}x{H} -pix_fmt rgb24 -r 30 '
            f'-i - -vf scale={W}:{H}:flags=neighbor {X264} -f h264 {outE}'),
        stdin=subprocess.PIPE, stderr=eE, start_new_session=True)
    time.sleep(2.0)
    # warmup (SPS/PPS + encoder init)
    enc.stdin.write(os.urandom(FSZ)); time.sleep(0.4)
    # reset file size baseline AFTER warmup
    base = os.path.getsize(outE)
    enc_lats = []
    for i in range(N):
        t0 = time.time()
        enc.stdin.write(os.urandom(FSZ))
        deadline = time.time() + 5
        while time.time() < deadline:
            if os.path.getsize(outE) > base:
                break
            time.sleep(0.002)
        enc_lats.append(time.time() - t0)
        base = os.path.getsize(outE)
    try: os.killpg(os.getpgid(enc.pid), signal.SIGKILL)
    except Exception: pass
    enc_lats.sort()
    print(f'ENCODE single-in-flight (file): n={len(enc_lats)} '
          f'median={enc_lats[len(enc_lats)//2]*1000:.0f} ms  '
          f'min={enc_lats[0]*1000:.0f}  max={enc_lats[-1]*1000:.0f} ms')

    # ---- FULL, file sink ----
    outF = f'{HERE}/ffsfF.out'
    try: os.remove(outF)
    except OSError: pass
    e1 = open(f'{HERE}/ffsfF1.err', 'w'); e2 = open(f'{HERE}/ffsfF2.err', 'w')
    f1 = subprocess.Popen(
        ['ffmpeg'] + shlex.split(
            f'-y -loglevel error -f rawvideo -s {W}x{H} -pix_fmt rgb24 -r 30 '
            f'-i - -vf scale={W}:{H}:flags=neighbor {X264} -f h264 -'),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=e1,
        start_new_session=True)
    f2 = subprocess.Popen(
        ['ffmpeg'] + shlex.split(
            f'-y -loglevel error -f h264 -r 30 -i - '
            f'-c:v rawvideo -pix_fmt rgb24 -f rawvideo {outF}'),
        stdin=subprocess.PIPE, stderr=e2, start_new_session=True)
    time.sleep(2.0)

    # pump f1.stdout -> f2.stdin in a thread (non-blocking, select)
    import threading, select
    stop = [False]
    def pump():
        raw = f1.stdout.raw
        while not stop[0]:
            try:
                r, _, _ = select.select([raw], [], [], 0.05)
            except (OSError, ValueError):
                break
            if not r:
                if f1.poll() is not None:
                    break
                continue
            try:
                c = raw.read(65536)
            except (OSError, ValueError):
                break
            if not c:
                break
            try:
                f2.stdin.write(c)
            except (BrokenPipeError, OSError):
                break
    pt = threading.Thread(target=pump, daemon=True); pt.start()

    # warmup
    f1.stdin.write(os.urandom(FSZ)); time.sleep(0.5)
    base = os.path.getsize(outF)
    full_lats = []
    for i in range(N):
        t0 = time.time()
        f1.stdin.write(os.urandom(FSZ))
        deadline = time.time() + 5
        while time.time() < deadline:
            if os.path.getsize(outF) > base:
                break
            time.sleep(0.002)
        full_lats.append(time.time() - t0)
        base = os.path.getsize(outF)
    stop[0] = True
    for p in (f1, f2):
        try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception: pass
    full_lats.sort()
    print(f'FULL(encode+decode) single-in-flight (file): n={len(full_lats)} '
          f'median={full_lats[len(full_lats)//2]*1000:.0f} ms  '
          f'min={full_lats[0]*1000:.0f}  max={full_lats[-1]*1000:.0f} ms')
    kill_ff()


if __name__ == '__main__':
    main()
