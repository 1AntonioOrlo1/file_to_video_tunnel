#!/usr/bin/env python3
"""Clean single-frame-in-flight WIRE chain latency (file sinks, no live pipe).

Reuses the proven file-sink pattern (the ENCODE half measured 4 ms). This
measures the mpegts+SRT+decode half with ZERO queueing:
  1) pre-encode ONE noise frame -> single.h264
  2) feed single.h264 to an SRT caller (mpegts mux), receiver SRT -> decode
     -> rawvideo FILE; poll the file for one full frame.
File sinks don't buffer like pipes, and one frame in flight means no queue.
This is the per-direction wire cost the tunnel RTT pays twice.
"""
import os, shlex, signal, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1280, 720, 30
FSZ = W * H * 3
LAT = os.environ.get('LP_SRT_LAT', '100')
H264 = f'{HERE}/ffwire.h264'
OUT = f'{HERE}/ffwire.out'
X264 = ('-c:v libx264 -pix_fmt yuv420p -preset ultrafast '
        '-tune zerolatency -crf 23 -g 1 -bf 0')


def kill_ff():
    me = os.getpid()
    for d in os.listdir('/proc'):
        if d.isdigit() and int(d) != me:
            try:
                raw = open(f'/proc/{d}/cmdline', 'rb').read()
                if b'ffmpeg' in raw and b'ffwire' in raw:
                    os.kill(int(d), signal.SIGKILL)
            except OSError:
                pass


def main():
    kill_ff()
    time.sleep(1)
    for f in (H264, OUT):
        try:
            os.remove(f)
        except OSError:
            pass

    # 1) pre-encode one noise frame -> h264 file
    eE = open(f'{HERE}/ffwireE.err', 'w')
    enc = subprocess.Popen(
        ['ffmpeg'] + shlex.split(
            f'-y -loglevel error -f lavfi -i '
            f"nullsrc=s={W}x{H}:r={FPS},geq=lum='random(1)*255':"
            f"cb='random(2)*255':cr='random(3)*255' -frames:v 1 "
            f'-vf scale={W}:{H}:flags=neighbor {X264} -f h264 {H264}'),
        stdout=subprocess.DEVNULL, stderr=eE, start_new_session=True)
    rc = enc.wait(timeout=20)
    if rc != 0 or not os.path.exists(H264):
        print('pre-encode FAIL:', open(f'{HERE}/ffwireE.err').read()[:300])
        return
    hsz = os.path.getsize(H264)
    print(f'pre-encoded 1 frame -> {hsz} bytes h264')

    # 2) wire: h264 file -> SRT caller; receiver SRT -> decode -> rawvideo file
    e1 = open(f'{HERE}/ffwire1.err', 'w'); e2 = open(f'{HERE}/ffwire2.err', 'w')
    recv = subprocess.Popen(
        ['ffmpeg'] + shlex.split(
            f'-y -loglevel error -i '
            f'srt://127.0.0.1:19200?mode=listener&latency={LAT} -map 0:v:0 '
            f'-c:v rawvideo -pix_fmt rgb24 -f rawvideo {OUT}'),
        stderr=e2, start_new_session=True)
    time.sleep(2.0)
    t0 = time.time()
    send = subprocess.Popen(
        ['ffmpeg'] + shlex.split(
            f'-y -loglevel error -i {H264} -c copy -f mpegts '
            f'srt://127.0.0.1:19200?mode=caller&latency={LAT}'),
        stderr=e1, start_new_session=True)
    deadline = time.time() + 8
    ok = False
    while time.time() < deadline:
        if os.path.exists(OUT) and os.path.getsize(OUT) >= FSZ:
            ok = True
            break
        time.sleep(0.002)
    el = time.time() - t0
    send.wait(timeout=10)
    for p in (send, recv):
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
    if ok:
        print(f'WIRE (mpegts+SRT+decode) single-in-flight, latency={LAT}: '
              f'{el*1000:.0f} ms  (per direction; RTT pays it ~2x)')
    else:
        print(f'FAIL: no frame in {el:.1f}s')
        print('send err:', open(f'{HERE}/ffwire1.err').read()[:300])
        print('recv err:', open(f'{HERE}/ffwire2.err').read()[:300])
    kill_ff()


if __name__ == '__main__':
    main()
