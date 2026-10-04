#!/usr/bin/env python3
"""Generic 7030-style frame generator: W H FRAMES [SEED].

Writes RGB24 frames to stdout. Each frame is pure-gray 0/255 8x8 blocks
(the 7030 color-fec workload) so x264/NVENC crf/qp 0 round-trip cleanly.
"""
import sys
import numpy as np

W, H, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 1
m = 8
rng = np.random.default_rng(SEED)


def block_frame():
    flat = rng.integers(0, 2, (H // m, W // m, 3), dtype=np.uint8) * 255
    img = np.repeat(np.repeat(flat, m, 0), m, 1)
    return img.tobytes()


out = sys.stdout.buffer
for _i in range(N):
    out.write(block_frame())
    out.flush()
