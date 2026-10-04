#!/usr/bin/env python3
"""Continuous 30 fps one-way SRT chain latency, A/B on SRT latency value.

Recreated for the latency-units question: the node trace shows ~0.62 s per
direction in a CONTINUOUS stream, but a single-frame wire test showed 48 ms.
The difference is continuous streaming — SRT tsbpd holds each packet for the
agreed receiver latency after the stream starts. If ffmpeg's URL `latency=`
is in ms, values 100/1000/100000 should shift the offset by the difference;
if in us, 100000 (100 ms) shifts it by ~90 ms and 1000 (~1 ms) by ~0.

FF1: rawvideo pipe (30 fps paced) -> x264 zerolatency -> mpegts -> SRT caller
FF2: SRT listener -> rawvideo -> stdout pipe; reader thread timestamps
     each full frame (real-time sink, as in the node's _read_in).
Absolute pacing (write ref = t_start + i/fps, independent of blocking).
"""
import os, shlex, signal, subprocess, threading, time, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1280, 720, 30
FSZ = W * H * 3
LAT = os.environ.get('LP_SRT_LAT', '100')
N = int(os.environ.get('LP_N', '300'))
WARM = 30

FF1 = shlex.split(
    f'-y -loglevel error -f rawvideo -s {W}x{H} -pix_fmt rgb24 -r 30 -i - '
    f'-vf scale={W}:{H}:flags=neighbor '
    f'-c:v libx264 -pix_fmt yuv420p -preset ultrafast -tune zerolatency '
    f'-crf 23 -g 1 -bf 0 -f mpegts '
    f'srt://127.0.0.1:19200?mode=caller&latency={LAT}')
FF2 = shlex.split(
    f'-y -loglevel error -i '
    f'srt://127.0.0.1:19200?mode=listener&latency={LAT} -map 0:v:0 '
    f'-c:v rawvideo -pix_fmt rgb24 -f rawvideo -')


def kill_leftovers():
    me = os.getpid()
    for d in os.listdir('/proc'):
        if d.isdigit() and int(d) != me:
            try:
                raw = open(f'/proc/{d}/cmdline', 'rb').read()
                if b'srt://127.0.0.1:1920' in raw and b'ffmpeg' in raw:
                    os.kill(int(d), signal.SIGKILL)
            except OSError:
                pass


def main():
    kill_leftovers()
    time.sleep(1)
    e1 = open(f'{HERE}/fflatrt1.err', 'w'); e2 = open(f'{HERE}/fflatrt2.err', 'w')
    ff2 = subprocess.Popen(['ffmpeg'] + FF2, stdout=subprocess.PIPE,
                           stderr=e2, start_new_session=True)
    time.sleep(2.0)
    ff1 = subprocess.Popen(['ffmpeg'] + FF1, stdin=subprocess.PIPE,
                           stderr=e1, start_new_session=True)
    time.sleep(3.0)

    arrivals = []
    stop = False

    def reader():
        raw = ff2.stdout.raw
        buf = b''
        while not stop:
            try:
                import select
                r, _, _ = select.select([raw], [], [], 0.05)
                if not r:
                    if ff2.poll() is not None:
                        break
                    continue
                c = raw.read(65536)
            except (OSError, ValueError):
                break
            if not c:
                break
            buf += c
            while len(buf) >= FSZ:
                buf = buf[FSZ:]
                arrivals.append(time.time())

    rt = threading.Thread(target=reader, daemon=True)
    rt.start()

    frames = [os.urandom(FSZ) for _ in range(N)]
    t_start = time.time()
    writes = []
    blocks = []
    for i in range(N):
        t0 = time.time()
        try:
            ff1.stdin.write(frames[i])
        except (BrokenPipeError, OSError):
            break
        blocks.append(time.time() - t0)
        writes.append(t_start + i / FPS)
        sl = t_start + (i + 1) / FPS - time.time()
        if sl > 0:
            time.sleep(sl)
        if ff1.poll() is not None:
            print('FF1 died'); break

    time.sleep(3.0)
    stop = True
    rt.join(timeout=5)
    ff1.stdin.close()
    for p in (ff1, ff2):
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass

    m = min(len(writes), len(arrivals))
    delays = [arrivals[i] - writes[i] for i in range(WARM, m)]
    bs = blocks[WARM:m]
    if bs:
        print(f'lat={LAT}: writes blocked max={max(bs)*1000:.0f}ms '
              f'frames_in={len(arrivals)}')
    if len(delays) >= 5:
        delays.sort()
        print(f'one-way continuous (lat={LAT}, n={len(delays)}): '
              f'min={delays[0]*1000:.0f}  p10={delays[len(delays)//10]*1000:.0f}  '
              f'median={delays[len(delays)//2]*1000:.0f}  '
              f'p90={delays[9*len(delays)//10]*1000:.0f}  '
              f'max={delays[-1]*1000:.0f} ms')
    else:
        print(f'FAIL: pairs={m} -> {len(delays)}; '
              f'writes={len(writes)} arrivals={len(arrivals)}')
        print('FF1 err:', open(f'{HERE}/fflatrt1.err').read()[:300])
        print('FF2 err:', open(f'{HERE}/fflatrt2.err').read()[:300])


if __name__ == '__main__':
    main()
