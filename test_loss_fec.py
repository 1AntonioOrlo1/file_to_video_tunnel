#!/usr/bin/env python3
"""Loss-resilience test WITHOUT root/netem: exercises the REAL pipeline
(x264 CRF0 encode -> decode -> group walk -> Cauchy FEC repair -> demux)
with injected group losses, exactly the failure mode the live tunnel faces
when SRT's ARQ window is exceeded.

Model:
  * each group occupies R=2 consecutive video frames;
  * "group lost" = both frames of that group are gone (the walker sees a
    hole and FEC must repair the stripe from the survivors);
  * FEC (k=8, m=2) recovers any stripe with <=2 lost groups;
  * a stripe with 3+ lost groups is DROPPED (live-tunnel policy) and the
    stream must stay in sync afterwards — no cascade corruption.

Proven: (a) 0..2 losses per stripe -> 100% byte recovery;
        (b) 3+ losses per stripe -> only that stripe's payload is lost,
            all following stripes stay byte-exact.
"""
import os
import subprocess
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tunnel_core import (M, R, K, M_PAR, R_META, cauchy_matrix,
                         render_group, render_meta, decode_group,
                         verify_group, group_capacity, parity_rows,
                         make_eof_group, EOF_PAYLOAD, stripe_of)
from mux import Muxer
from demux import Demuxer
from bitcoder_fec import fec_decode

W, H, FPS = 640, 360, 30
B = group_capacity(W, H, M)
K, mP = K, M_PAR
P = cauchy_matrix(K, mP)
rng = np.random.default_rng(42)

N_STRIPES = 30
NS = K + mP                      # groups per stripe (10)
frame_bytes = W * H * 3

# ---- build the payload: each stripe carries K*B bytes of random data ----
stripe_payloads = [[rng.integers(0, 256, B, dtype=np.uint8)
                    .tobytes() for _ in range(K)] for _ in range(N_STRIPES)]
par_cache = [parity_rows(np.stack([np.frombuffer(s, 'u1') for s in sp]), P)
             for sp in stripe_payloads]

# ---- render the whole stream and push it through REAL x264 CRF0 ----
frames = []
for s in range(N_STRIPES):
    for i, payload in enumerate(stripe_payloads[s] + [p.tobytes() for p in par_cache[s]]):
        f = render_group(payload, s * NS + i, M, W, H)
        frames.extend([f] * R)
eof = make_eof_group(W, H)
frames.extend([eof] * R)
meta = render_meta({'fn': 'loss-test', 'w': W, 'h': H, 'M': M, 'R': R,
                    'k': K, 'm': mP, 'B': B, 'fps': FPS}, W, H)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'e2e2_loss', 'loss_rt.mp4')
os.makedirs(os.path.dirname(out), exist_ok=True)
t0 = time.time()
ff = subprocess.Popen([
    'ffmpeg', '-y', '-loglevel', 'error',
    '-f', 'rawvideo', '-vcodec', 'rawvideo',
    '-s', f'{W}x{H}', '-pix_fmt', 'rgb24', '-r', str(FPS), '-i', '-',
    '-map', '0:v:0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    '-preset', 'ultrafast', '-tune', 'zerolatency', '-crf', '0',
    '-g', str(FPS), '-keyint_min', str(FPS), '-bf', '0', out,
], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
for _ in range(R_META):
    ff.stdin.write(meta)
for f in frames:
    ff.stdin.write(f)
ff.stdin.close()
err = ff.stderr.read(); ff.wait()
enc_t = time.time() - t0
assert ff.returncode == 0, f"encode failed: {err.decode()[:300]}"
print(f"[1] encoded {len(frames)+R_META} frames (x264 CRF0) in {enc_t:.1f}s "
      f"({os.path.getsize(out)} B)")

# ---- decode back to raw ----
ff = subprocess.run([
    'ffmpeg', '-y', '-loglevel', 'error', '-i', out,
    '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True)
raw = ff.stdout
n_total = len(frames) // R
assert len(raw) // frame_bytes >= n_total + R_META, "frame count mismatch!"
off = R_META * frame_bytes


def read_group(gi):
    """gi = absolute group index; returns (seq, payload) or (None, None)."""
    win = raw[off + gi * R * frame_bytes: off + (gi + 1) * R * frame_bytes]
    frs = [win[i * frame_bytes:(i + 1) * frame_bytes] for i in range(R)]
    h8, payload = decode_group(frs, M, W, H)
    if h8 is None:
        return None, None
    seq, ok = verify_group(h8, payload)
    if not ok:
        return None, None
    return seq, payload


# ---- baseline: no loss, everything must be perfect ----
bad = 0
for gi in range(n_total):
    seq, payload = read_group(gi)
    if seq is None:
        bad += 1
print(f"[2] baseline (no loss): {n_total - bad}/{n_total} groups CRC-OK"
      + (f"  <-- {bad} BAD" if bad else "  (perfect)"))
assert bad == 0, "baseline codec path is not clean — stop"

# ---- inject group losses per stripe, run FEC repair ----
def simulate(loss_plan):
    """loss_plan: stripe -> set of lost group-indices-within-stripe.
    Returns per-stripe recovered data rows (or None if dropped)."""
    results = {}
    for s in range(N_STRIPES):
        recv = [None] * NS
        lost = loss_plan.get(s, set())
        for i in range(NS):
            if i in lost:
                continue                     # both frames of this group gone
            seq, payload = read_group(s * NS + i)
            if seq is not None:
                recv[i] = payload
        n_lost = sum(1 for x in recv if x is None)
        if n_lost > mP:
            results[s] = None                # live-tunnel policy: drop stripe
            continue
        # FEC repair (works with any K survivors incl. parity)
        from bitcoder_fec import fec_decode
        rec = fec_decode(list(recv), K, P)
        results[s] = [r.tobytes() for r in rec]
    return results


def check(name, loss_plan, allow_drops):
    res = simulate(loss_plan)
    ok_st = bad_st = drop_st = 0
    first_bad = None
    for s in range(N_STRIPES):
        if res[s] is None:
            drop_st += 1
            if not allow_drops:
                first_bad = first_bad or s
            continue
        if all(res[s][i] == stripe_payloads[s][i] for i in range(K)):
            ok_st += 1
        else:
            bad_st += 1
            first_bad = first_bad or s
    tag = 'OK ' if (bad_st == 0 and drop_st == (N_STRIPES - ok_st)
                    and (allow_drops or drop_st == 0)) else 'FAIL'
    print(f"[{tag}] {name}: {ok_st} stripes recovered byte-exact, "
          f"{bad_st} corrupted, {drop_st} dropped"
          + (f" (first problem at stripe {first_bad})" if first_bad else ""))
    return bad_st == 0


# case 1: no loss
ok1 = check("no loss", {}, allow_drops=False)

# case 2: every stripe loses exactly 2 random groups (max correctable)
plan2 = {}
for s in range(N_STRIPES):
    plan2[s] = set(rng.choice(NS, size=mP, replace=False).tolist())
ok2 = check(f"all stripes lose {mP} groups (max correctable)", plan2,
            allow_drops=False)

# case 3: every stripe loses 1 random group
plan3 = {s: {int(rng.integers(0, NS))} for s in range(N_STRIPES)}
ok3 = check("all stripes lose 1 group", plan3, allow_drops=False)

# case 4: 6 stripes lose m+1=3 groups -> must be dropped, the REST stays exact
plan4 = {}
victims = sorted(rng.choice(N_STRIPES, size=6, replace=False).tolist())
for s in victims:
    plan4[s] = set(rng.choice(NS, size=mP + 1, replace=False).tolist())
ok4 = check("6 stripes lose 3 groups (uncorrectable -> drop)", plan4,
            allow_drops=True)

# case 5: verify the dropped-stripe policy does NOT cascade: with case-4
# losses, every NON-victim stripe must be byte-exact
res4 = simulate(plan4)
cascade_bad = [s for s in range(N_STRIPES)
               if s not in victims and res4[s] is not None
               and not all(res4[s][i] == stripe_payloads[s][i] for i in range(K))]
print(f"[{'OK ' if not cascade_bad else 'FAIL'}] no cascade: "
      f"non-victim stripes corrupted after drops: {len(cascade_bad)}")

# case 6: TRUE end-to-end demux over a repaired stream. A known blob is
# packed into the EXACT mux packet format the live sender produces
# (conn|len|flags|data back-to-back, one B-byte packet per group payload),
# then split into FEC stripes with parity; per-stripe losses are injected,
# FEC-repaired, and the demuxed output must be the original blob.
known_blob = bytes(rng.integers(0, 256, 60000, dtype=np.uint8))
from mux import pack_packet
PKT_DATA = B - 4
packed = b''
off = 0
while off < len(known_blob):
    chunk = known_blob[off:off + PKT_DATA]
    off += len(chunk)
    packed += pack_packet(1, chunk, fin=(off >= len(known_blob)))
while len(packed) % B:
    packed += b'\x00' * 4        # idle filler packet: conn=0 len=0 flags=0
payload_groups6 = [packed[i * B:(i + 1) * B] for i in range(len(packed) // B)]
n_g6 = len(payload_groups6)
gper = K
st6 = []
for s in range(0, n_g6, gper):
    chunk = list(payload_groups6[s:s + gper])
    while len(chunk) < gper:
        chunk.append(b'\x00' * B)
    st6.append(chunk)
par6 = [parity_rows(np.stack([np.frombuffer(x, 'u1') for x in st]), P)
        for st in st6]
# 1 lost group in each of the first (len-1) stripes: max common loss pattern
loss6 = {s: {int(rng.integers(0, gper))} for s in range(len(st6) - 1)}
dm6 = Demuxer()
recovered6 = b''
for s in range(len(st6)):
    full = [np.frombuffer(x, 'u1') for x in st6[s]]
    par = par6[s]
    recv = [None] * (gper + len(par))
    for i in range(gper):
        if i not in loss6.get(s, set()):
            recv[i] = full[i]
    for j in range(len(par)):
        recv[gper + j] = par[j]
    if sum(1 for x in recv if x is None) > mP:
        continue
    rec = fec_decode(list(recv), gper, P)
    for r in rec:
        for conn, data, fin in dm6.feed(r.tobytes()):
            recovered6 += data
ok6 = (recovered6 == known_blob)
print(f"[{'OK ' if ok6 else 'FAIL'}] end-to-end demux over repaired stream: "
      f"{len(recovered6)}/{len(known_blob)} B, byte-exact: {ok6} "
      f"({len(st6)} stripes, 1 loss/stripe)")

print("\nRESULT:", "PASS" if (ok1 and ok2 and ok3 and ok4 and ok6
                             and not cascade_bad) else "FAIL")
sys.exit(0 if (ok1 and ok2 and ok3 and ok4 and ok6
               and not cascade_bad) else 1)
