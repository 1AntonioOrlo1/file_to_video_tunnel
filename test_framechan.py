#!/usr/bin/env python3
"""Pure unit test of FrameChannel: no QUIC, no aioquic.

Feed a burst of random datagrams of varying sizes, let the ticker pace
them into frames, receive the other side, and require byte-exact match.
Also tests: idle periods, datagrams larger than B, mixed streams.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from videoframe_channel import FrameChannel


async def test(width, height, n_datagrams, sizes, with_idle=False):
    chan = FrameChannel(width, height, fps=120)  # fast clock for the test
    stop = asyncio.Event()
    ta = asyncio.ensure_future(chan.ticker('a', stop))
    tb = asyncio.ensure_future(chan.ticker('b', stop))

    expected = [os.urandom(s) for s in sizes]
    assert len(expected) == n_datagrams

    # burst: send everything, interleaved with occasional idle gaps
    for i, d in enumerate(expected):
        chan.send('a', d)
        if with_idle and i % 20 == 0:
            # force a few empty ticks
            for _ in range(3):
                chan.tick('a')
                await asyncio.sleep(0)

    async def receiver():
        got = []
        while len(got) < n_datagrams:
            dtgs = await chan.recv('b')
            got.extend(dtgs)
        return got

    got = await asyncio.wait_for(receiver(), timeout=60)

    ok = got == expected
    print(f"{width}x{height}: {n_datagrams} datagrams, "
          f"frames_sent_a={chan.frames_sent['a']}", end=" ")
    if ok:
        print("PASS")
    else:
        # find first mismatch
        for i, (e, g) in enumerate(zip(expected, got)):
            if e != g:
                print(f"FAIL: mismatch at datagram {i} "
                      f"(expected {len(e)}B, got {len(g)}B)")
                break
        else:
            print(f"FAIL: length mismatch ({len(expected)} vs {len(got)})")
    stop.set()
    ta.cancel(); tb.cancel()
    for t in (ta, tb):
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    return ok


async def main():
    from tunnel_core import group_capacity
    allok = True
    B = group_capacity(1280, 720)
    print(f"B@720p = {B}")
    # 1) all datagrams exactly 1350 B (the QUIC size)
    allok &= await test(1280, 720, 300, [1350] * 300)
    # 2) mixed sizes incl. tiny
    import random
    random.seed(42)
    allok &= await test(1280, 720, 200, [random.randint(1, 1350) for _ in range(200)])
    # 3) with idle gaps between bursts
    allok &= await test(1280, 720, 100, [1350] * 100, with_idle=True)
    # 4) 1080p
    allok &= await test(1920, 1080, 100, [1350] * 100)
    # 5) datagrams spanning MULTIPLE frames (size > B)
    allok &= await test(1280, 720, 50, [B * 3 + 1234] * 50)
    print("RESULT:", "PASS" if allok else "FAIL")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
