#!/usr/bin/env python3
"""Offline decode of a captured raw SRT stream: walk groups, check seq
pattern (are odd seqs missing?), CRCs, payload sanity."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tunnel_core import (M, R, K, M_PAR, R_META, decode_group,
                         verify_group, group_capacity, EOF_PAYLOAD,
                         stripe_of)

W = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
H = int(sys.argv[3]) if len(sys.argv) > 3 else 720
B = group_capacity(W, H, M)
path = sys.argv[1] if len(sys.argv) > 1 else 'e2e/srt_capture.raw'
raw = open(path, 'rb').read()
frame_size = W * H * 3
n_frames = len(raw) // frame_size
print(f"file: {path}, {n_frames} frames, {n_frames - R_META} data frames "
      f"= {(n_frames - R_META) // R} groups")

off = R_META * frame_size
seqs = []
bad_crc = 0
eof_count = 0
idle = 0
for gi in range((n_frames - R_META) // R):
    win = raw[off + gi * R * frame_size: off + (gi + 1) * R * frame_size]
    frs = [win[i * frame_size:(i + 1) * frame_size] for i in range(R)]
    h8, payload = decode_group(frs, M, W, H)
    if h8 is None:
        seqs.append('DECODE-FAIL')
        continue
    seq, ok = verify_group(h8, payload)
    if not ok:
        bad_crc += 1
        seqs.append(f'{seq}!BAD!')
        continue
    if seq == 0xFFFF and payload[:4] == EOF_PAYLOAD:
        eof_count += 1
        seqs.append(f'{seq} EOF')
    else:
        seqs.append(seq)
        if payload == b'\x00' * B:
            idle += 1

print(f"bad CRC: {bad_crc}, EOF sentinels: {eof_count}, idle groups: {idle}")
print("first 40 seqs:", seqs[:40])
# gaps: expected consecutive run
nums = [s for s in seqs if isinstance(s, int) and s != 0xFFFF]
gaps = [(a, b) for a, b in zip(nums, nums[1:]) if b - a > 1]
print(f"gaps (a->b where b-a>1): {gaps[:20]} ... total {len(gaps)} gaps")
dupes = len(nums) - len(set(nums))
print(f"duplicate seqs: {dupes}")
