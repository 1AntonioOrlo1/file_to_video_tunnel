#!/usr/bin/env python3
"""Offline codec round-trip: render a full stripe sequence, encode with the
EXACT sender x264 params to an mp4, decode back, and check the seq values.
This isolates the video-codec layer from SRT."""
import os
import subprocess
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tunnel_core import (M, R, K, M_PAR, R_META, cauchy_matrix,
                         render_group, render_meta, decode_group,
                         verify_group, group_capacity, parity_rows,
                         make_eof_group, EOF_PAYLOAD)
from bitcoder_fec import crc_for_group

W, H, FPS = 1280, 720, 30
B = group_capacity(W, H, M)
P = cauchy_matrix(K, M_PAR)
rng = np.random.default_rng(7)

# Build a sequence: 1 full data stripe (k data + m parity) + 1 idle stripe + EOF
data = [rng.integers(0, 256, B, dtype=np.uint8).tobytes() for _ in range(K)]
par = [p.tobytes() for p in parity_rows(np.stack([np.frombuffer(s, 'u1') for s in data]), P)]
stripe0 = data + par                      # seq 0..9
stripe1 = [b'\x00' * B] * (K + M_PAR)     # seq 10..19 idle

frames = []
for i, payload in enumerate(stripe0 + stripe1):
    f = render_group(payload, i, M, W, H)
    frames.extend([f] * R)
# EOF sentinel
eof = make_eof_group(W, H)
frames.extend([eof] * R)

# meta header (R_META copies)
meta_bytes = render_meta({'fn': 'live-tunnel', 'w': W, 'h': H, 'M': M,
                          'R': R, 'k': K, 'm': M_PAR, 'B': B, 'fps': FPS}, W, H)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'e2e', 'codec_rt.mp4')
os.makedirs(os.path.dirname(out), exist_ok=True)
frame_bytes = W * H * 3
t0 = time.time()
with open(out, 'wb') as fh:
    # encode via ffmpeg stdin (exact sender params)
    ff = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{W}x{H}', '-pix_fmt', 'rgb24', '-r', str(FPS), '-i', '-',
        '-map', '0:v:0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast', '-tune', 'zerolatency', '-crf', '23',
        '-g', str(FPS), '-keyint_min', str(FPS), '-bf', '0',
        out,
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for _ in range(R_META):
        ff.stdin.write(meta_bytes)
    for f in frames:
        ff.stdin.write(f)
    ff.stdin.close()
    err = ff.stderr.read()
    ff.wait()
enc_t = time.time() - t0
print(f"encoded {len(frames)+R_META} frames -> {out} in {enc_t:.2f}s "
      f"({os.path.getsize(out)} B)")
if err.strip():
    print("ffmpeg stderr:", err.decode()[:500])

# decode back: probe, read raw, skip meta, walk groups
probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                        '-show_entries', 'stream=width,height', '-of', 'csv=p=0',
                        out], capture_output=True, text=True)
w, h = (int(x) for x in probe.stdout.strip().split(','))
print(f"decoded stream is {w}x{h}")
ff = subprocess.run([
    'ffmpeg', '-y', '-loglevel', 'error', '-i', out,
    '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
    capture_output=True)
raw = ff.stdout
print(f"read back {len(raw)//frame_bytes} frames "
      f"(expected {len(frames)+R_META})")
if len(raw) // frame_bytes < len(frames) + R_META:
    print("MISMATCH in frame count — x264 dropped frames!")

# skip R_META meta frames
off = R_META * frame_bytes
expected_seq = list(range(K + M_PAR)) + list(range(K + M_PAR, 2 * (K + M_PAR)))
got = []
n_groups = K + M_PAR + (K + M_PAR) + 1  # stripe0 + stripe1 + eof
for gi in range(n_groups):
    win = raw[off + gi * R * frame_bytes: off + (gi + 1) * R * frame_bytes]
    frs = [win[i * frame_bytes:(i + 1) * frame_bytes] for i in range(R)]
    h8, payload = decode_group(frs, M, w, h)
    if h8 is None:
        got.append((gi, 'DECODE-FAIL'))
        continue
    seq, ok = verify_group(h8, payload)
    is_eof = (seq == 0xFFFF and payload[:4] == EOF_PAYLOAD)
    got.append((gi, f'seq={seq} crc={"OK" if ok else "BAD"} eof={is_eof} '
                    f'exp={expected_seq[gi] if gi < len(expected_seq) else "EOF"}'))

print("\n  gi  result")
for gi, res in got:
    exp = expected_seq[gi] if gi < len(expected_seq) else 'EOF'
    print(f"  {gi:2d}  {res}")
ok_count = sum(1 for _, r in got if 'crc=OK' in r)
print(f"\nCRC-OK: {ok_count}/{n_groups}")
