#!/usr/bin/env python3
"""Unit tests for the tunnel core (no ffmpeg, no network)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from tunnel_core import (K, M, M_PAR, R, EOF_PAYLOAD, group_capacity,
                         make_eof_group, parity_rows, render_group,
                         render_meta, decode_meta, decode_group,
                         stripe_data, verify_group, SeqWrapTracker)
from mux import Muxer
from demux import Demuxer

W, H = 1280, 720
B = group_capacity(W, H, M)
print(f"capacity: {B} B/group, ~{B * K * 30 // (K + M_PAR) / 1024:.0f} KB/s at 30fps k=8 m=2")

fails = 0

def check(name, cond, extra=''):
    global fails
    status = 'OK' if cond else 'FAIL'
    if not cond:
        fails += 1
    print(f"  [{status}] {name} {extra}")

print("== 1. group render/decode roundtrip (random payloads) ==")
rng = np.random.default_rng(42)
for i in range(4):
    payload = rng.integers(0, 256, B, dtype=np.uint8).tobytes()
    frames = [render_group(payload, i, M, W, H)] * R
    h8, decoded = decode_group(frames, M, W, H)
    seq, ok = verify_group(h8, decoded)
    check(f"group {i}: CRC valid", ok, f"seq={seq}")
    check(f"group {i}: seq={i}", seq == i)
    check(f"group {i}: payload identical", decoded == payload,
          f"({len(decoded)} B)")

print("== 2. all-zero payload (idle group) ==")
z = b'\x00' * B
h8, dec = decode_group([render_group(z, 7, M, W, H)] * R, M, W, H)
check("idle payload identical", dec == z)
check("idle CRC valid", verify_group(h8, dec)[1])

print("== 3. metadata roundtrip ==")
meta = render_meta({'fn': 'live-tunnel', 'w': W, 'h': H, 'M': M, 'R': R,
                    'k': K, 'm': M_PAR, 'B': B, 'fps': 30}, W, H)
got = decode_meta([meta] * 3, W, H)
check("meta parsed", got is not None)
check("meta fields", got and got['k'] == K and got['m'] == M_PAR and
      got['B'] == B and got['M'] == M, str(got) if got else '')

print("== 4. mux/demux roundtrip (multi-connection, >group-sized data) ==")
mux = Muxer(B)
dm = Demuxer()
c1 = os.urandom(3 * B)
c2 = os.urandom(100)
mux.feed(1, c1)
mux.feed(2, c2, fin=True)
groups = []
while mux.pending() >= B:
    groups.append(mux.next_group())
groups.append(mux.flush_partial())
out1, out2 = b'', b''
fin2 = False
for g in groups:
    for conn, data, fin in dm.feed(g):
        if conn == 1:
            out1 += data
        elif conn == 2:
            out2 += data
            fin2 = fin
check("conn1 data identical", out1 == c1, f"({len(out1)} B)")
check("conn2 data identical", out2 == c2, f"({len(out2)} B)")
check("conn2 FIN seen", fin2)

print("== 5. FEC: stripe with 2 lost groups repaired (k=8 m=2) ==")
from tunnel_core import cauchy_matrix
P = cauchy_matrix(K, M_PAR)
stripe = [rng.integers(0, 256, B, dtype=np.uint8).tobytes() for _ in range(K)]
parity = parity_rows(np.stack([np.frombuffer(s, dtype=np.uint8) for s in stripe]), P)
allrows = [np.frombuffer(s, dtype=np.uint8) for s in stripe] + parity
# lose group 2 (data) and group 9 (first parity)
recv = list(allrows)
recv[2] = None
recv[9] = None
data = stripe_data(recv, K, P)
check("stripe repaired",
      all(data[i].tobytes() == stripe[i] for i in range(K)),
      "" if all(data[i].tobytes() == stripe[i] for i in range(K)) else "mismatch")
# also: all data present, parity lost
recv2 = [np.frombuffer(s, dtype=np.uint8) for s in stripe] + [None, None]
data2 = stripe_data(recv2, K, P)
check("parity-only loss ok", all(data2[i].tobytes() == stripe[i] for i in range(K)))

print("== 6. EOF sentinel ==")
eof = make_eof_group(W, H, M)
h8, dec = decode_group([eof] * R, M, W, H)
seq, ok = verify_group(h8, dec)
check("EOF verified", ok and seq == 0xFFFF and dec[:4] == EOF_PAYLOAD)

print("== 7. seq wrap tracker ==")
t = SeqWrapTracker()
idxs = [t.true_index(65535), t.true_index(0), t.true_index(1)]
check("wrap: 65535 -> 0 -> 1 maps to 65535,65536,65537",
      idxs == [65535, 65536, 65537], str(idxs))

print()
if fails:
    print(f"FAILED: {fails} check(s) failed")
    sys.exit(1)
print("ALL CORE TESTS PASSED")
