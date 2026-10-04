#!/usr/bin/env python3
"""Unit test: FEC-protected datagram channel over 7030 frames.

  * byte-exact round trip, no loss
  * controlled frame loss: m groups' worth of frames dropped in a stripe
    (FEC must repair, datagrams still byte-exact)
  * datagram sizes 1..max_dtg, including multi-frame spans

Loss model: drop whole FRAMES (a video service can drop/duplicate a frame).
Dropping <= m groups' worth of frames per stripe is repairable by Cauchy.

The ticker (30 fps video clock) is started only AFTER the loss test has
excised its frames from staging, so the drops are deterministic.
"""
import asyncio
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fec_frame_channel import FecFrameChannel


async def run(width, height, n_dtg, datagrams, drop_groups=()):
    chan = FecFrameChannel(width, height, fps=30)
    stop = asyncio.Event()
    got = []
    pt = asyncio.ensure_future(chan.pump('b', lambda d: got.extend(d), stop))
    tasks = [pt]

    for d in datagrams:
        chan.send('a', d)
    chan.flush('a')

    # frames now sit in chan._staging['b'] in order:
    #   stripe s, group g -> frames [ (s*G+g)*R : (s*G+g+1)*R )
    if drop_groups:
        G = chan.groups_per_stripe
        r = chan.r
        drop_idx = set()
        for s, gi in drop_groups:
            base = (s * G + gi) * r
            drop_idx.update(range(base, base + r))
        staging = chan._staging['b']
        chan._staging['b'] = [f for i, f in enumerate(staging)
                              if i not in drop_idx]
        dropped = len(staging) - len(chan._staging['b'])
    else:
        dropped = 0

    # now let the video clock run
    tt = asyncio.ensure_future(chan.ticker('b', stop))
    tasks.append(tt)

    loop = asyncio.get_running_loop()
    t0 = loop.time()
    while len(got) < n_dtg and loop.time() - t0 < 45:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.1)
    stop.set()
    for t in tasks:
        t.cancel()
    for t in tasks:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass

    ok = got == list(datagrams)
    st = chan._recv.get('b')
    stats = (f"groups={st.groups} repaired={st.repaired} "
             f"dropped_stripes={st.dropped_stripes} bad={st.bad} "
             f"resyncs={chan.resyncs} frames_dropped={dropped}")
    print(f"{width}x{height} {n_dtg} dtg {'loss' if drop_groups else 'no-loss'}: "
          f"{'PASS' if ok else 'FAIL'} ({stats})")
    if not ok:
        for i, (a, b) in enumerate(zip(got, datagrams)):
            if a != b:
                print(f"  first diff at {i}: len {len(a)}/{len(b)}")
                break
    return ok


async def main():
    allok = True

    # 1) no loss, 1350-B datagrams (40 groups -> 5 full stripes)
    allok &= await run(1280, 720, 100, [os.urandom(1350) for _ in range(100)])

    # 2) no loss, mixed sizes
    rng = random.Random(7)
    allok &= await run(1280, 720, 60,
                       [os.urandom(rng.randint(1, 1350)) for _ in range(60)])

    # 3) controlled loss: 40 dtg (~11 groups -> stripes 0..1), drop 2 groups
    #    (m=2) from stripe 0 -> FEC must repair, bytes still exact.
    allok &= await run(1280, 720, 40, [os.urandom(1350) for _ in range(40)],
                       drop_groups=[(0, 2), (0, 5)])

    # 4) heavier loss: drop m groups in TWO different stripes at once.
    allok &= await run(1280, 720, 100, [os.urandom(1350) for _ in range(100)],
                       drop_groups=[(0, 1), (0, 4), (2, 3), (2, 7)])

    print("RESULT:", "PASS" if allok else "FAIL")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
