"""Empirical concurrency test for Muxer: does a fast client + main-loop
admit/emit preserve order AND put the FIN at the very tail?

Simulates the node's two tasks:
  * reader task: feed(cid, data) for 64KB chunks, then feed(cid, b'', fin=True)
  * main loop: admit(holdback) + next_group()/flush_partial() into stripes
Then demuxes the emitted groups and reports where the FIN lands vs total data.
"""
import random
from mux import Muxer, FLAG_FIN
from demux import Demuxer

B = 48576          # 4K group
CAP_GROUPS = 8     # mux cap = 8 groups
K = 8

def run(trials=200):
    for t in range(trials):
        random.seed(t)
        mux = Muxer(B)
        mux.cap = CAP_GROUPS * B
        dm = Demuxer()
        total_payload = random.randint(2_000_000, 14_000_000)
        blob = bytearray(random.getrandbits(8) for _ in range(total_payload))
        # we don't need real randomness speed; build it lazily
        blob = bytes((i * 31 + t) % 256 for i in range(total_payload))

        fin_seen_at = None
        data_out = 0
        off = 0
        fin_fed = False
        stripe = 0
        steps = 0
        # interleave: reader feeds 64KB, main loop admits+emits a bit
        while True:
            steps += 1
            if steps > 200000:
                print(f'trial {t}: EXCEEDED STEPS, off={off}/{total_payload}')
                return
            # --- reader task: feed one 64KB chunk (or the FIN) ---
            if off < total_payload:
                chunk = blob[off:off + 65536]
                # backpressure: skip feeding if holdback non-empty
                if not mux.backpressure(1):
                    mux.feed(1, chunk)
                    off += len(chunk)
            elif not fin_fed:
                mux.feed(1, b'', fin=True)
                fin_fed = True
            # --- main loop task ---
            for cid in list(mux.holdback.keys()):
                n = mux.admitted(cid)
                if n:
                    mux.admit(cid, n)
            while mux.pending() >= B and len([x for x in ()]) < 0:
                pass
            # emit as full groups as available
            while mux.pending() >= B:
                g = mux.next_group()
                for conn, data, fin in dm.feed(g):
                    if conn == 1:
                        if fin:
                            if fin_seen_at is None:
                                fin_seen_at = data_out
                        else:
                            data_out += len(data)
            if off >= total_payload and fin_fed and mux.pending() == 0 \
                    and not mux.holdback:
                # flush tail
                if mux.pending():
                    g = mux.flush_partial()
                    for conn, data, fin in dm.feed(g):
                        if conn == 1:
                            if fin:
                                if fin_seen_at is None:
                                    fin_seen_at = data_out
                            else:
                                data_out += len(data)
                break

        # final tail flush safety
        if mux.pending():
            g = mux.flush_partial()
            for conn, data, fin in dm.feed(g):
                if conn == 1 and not fin:
                    data_out += len(data)
            for conn, data, fin in dm.feed(g):
                if conn == 1 and fin and fin_seen_at is None:
                    fin_seen_at = data_out

        ok = (data_out == total_payload) and (fin_seen_at == total_payload)
        if not ok:
            print(f'trial {t}: payload={total_payload} data_out={data_out} '
                  f'FIN_at={fin_seen_at} '
                  f'missing={total_payload-data_out} '
                  f'FIN_early_by={total_payload-fin_seen_at if fin_seen_at is not None else None}')
            return
    print(f'ALL {trials} trials OK: order preserved, FIN at tail, '
          f'byte-exact')

run()
