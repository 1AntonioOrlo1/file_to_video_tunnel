#!/usr/bin/env python3
"""FEC-protected datagram <-> 7030-frame channel (the proven preset).

This is the carrier for "a QUIC/VPN tunnel through a video service", built
on the 7030 color + Cauchy-FEC machinery that already round-trips over a
real lossy video link (tunnel_node.py). Layering, bottom to top:

  video frames (7030, R identical copies per group)
  -> FEC stripe: k data groups + m parity groups (any k of k+m recover)
  -> B-byte group payloads (B = group_capacity(w,h))
  -> LINEAR datagram stream: [len16][data] records packed back-to-back
  -> QUIC datagrams (the unit QUIC tracks & retransmits)

SEND (side):  datagram -> [len16] pack into a linear buffer -> cut into
  B-byte groups as they fill -> every k groups make a stripe with m parity
  -> render all (k+m) groups, R copies each -> push the frames.

RECV (side):  pull R frames -> decode_group (averages the R copies, so a
  single smeared frame is still correct) -> place the group in its stripe
  by seq -> when the stripe resolves, either FEC-recover the k data payloads
  (>= k groups survived) or zero-pad k*B bytes (unrecoverable: the affected
  datagrams were simply never ACKed, so QUIC retransmits them). Either way
  the linear stream advances by exactly k*B, so it stays ALIGNED — zeros
  read as len16==0 records the reassembler skips. A bounded-length resync
  (skip bytes while the read length is implausible) is a safety net for
  unexpected corruption.

Because QUIC already retransmits un-ACKed datagrams, FEC's job here is
latency (repair a video-dropped group locally instead of waiting a full
RTT); an unrecoverable stripe just costs its datagrams one retransmit.
"""
import asyncio
import numpy as np

from tunnel_core import (group_capacity, render_group, decode_group, R, K,
                         M_PAR, parity_rows, stripe_data)
from bitcoder_fec import (cauchy_matrix, crc_for_group, unpack_header)


class _RecvState:
    def __init__(self, chan):
        self.chan = chan
        self.frames = []
        self.stripe_no = None
        self.stripe_recv = {}
        self.buf = bytearray()
        self.groups = 0
        self.repaired = 0
        self.dropped_stripes = 0
        self.bad = 0
        self._last = None
        self._wraps = 0

    def _true_idx(self, seq):
        if self._last is not None and seq < self._last:
            self._wraps += 1
        self._last = seq
        return self._wraps * 65536 + seq

    def feed_frame(self, frame):
        """Add one raw frame; returns a list of complete datagrams (may be
        empty)."""
        self.frames.append(frame)
        out = []
        while len(self.frames) >= self.chan.r:
            rf = self.frames[:self.chan.r]
            del self.frames[:self.chan.r]
            h8, payload = decode_group(rf, self.chan.m, self.chan.w,
                                       self.chan.h)
            if h8 is None or payload is None:
                self.bad += 1
                continue
            g, ok = unpack_header(h8)
            if not ok:
                self.bad += 1
                continue
            if crc_for_group(g, payload) != int.from_bytes(h8[3:7], 'big'):
                self.bad += 1
                continue
            self.groups += 1
            true_idx = self._true_idx(g)
            s = true_idx // self.chan.groups_per_stripe
            idx = true_idx % self.chan.groups_per_stripe
            if self.stripe_no is None:
                self.stripe_no = s
                self.stripe_recv = {}
            elif s != self.stripe_no:
                out.extend(self._flush())
                self.stripe_no = s
                self.stripe_recv = {}
            if idx < self.chan.groups_per_stripe and idx not in self.stripe_recv:
                self.stripe_recv[idx] = payload
            if len(self.stripe_recv) >= self.chan.groups_per_stripe:
                out.extend(self._flush())
                self.stripe_no = None
                self.stripe_recv = {}
        return out

    def _flush(self):
        recv = [self.stripe_recv.get(i) for i in range(self.chan.groups_per_stripe)]
        n_present = sum(1 for x in recv if x is not None)
        n_lost = self.chan.groups_per_stripe - n_present
        if n_present >= self.chan.k:
            rows = stripe_data(recv, self.chan.k, self.chan.P)
            data = [r.tobytes() for r in rows]
            if n_lost:
                self.repaired += n_lost
        else:
            # Unrecoverable stripe: zero-pad k*B to keep the linear stream
            # aligned. The datagrams that lived in the lost groups are simply
            # never ACKed, so QUIC retransmits them.
            self.dropped_stripes += 1
            data = [b'\x00' * self.chan.B for _ in range(self.chan.k)]
        for p in data:
            self.buf += p
        return self._extract()

    def _extract(self):
        out = []
        resyncs = 0
        while len(self.buf) >= 2:
            ln = int.from_bytes(self.buf[:2], 'big')
            if ln == 0:
                del self.buf[:2]
                continue
            if ln > self.chan.max_dtg:
                # desynced: advance one byte (safety net; zero-padding keeps
                # the stream aligned so this is rare)
                del self.buf[:1]
                resyncs += 1
                continue
            if len(self.buf) < 2 + ln:
                break
            out.append(bytes(self.buf[2:2 + ln]))
            del self.buf[:2 + ln]
        if resyncs:
            self.chan.resyncs += resyncs
        return out


class FecFrameChannel:
    def __init__(self, width, height, m=8, k=K, mpar=M_PAR, fps=30,
                 r=None, max_dtg=1350):
        self.w, self.h, self.m = width, height, m
        self.k, self.mpar, self.fps = k, mpar, fps
        self.r = r if r is not None else R
        self.max_dtg = max_dtg
        self.B = group_capacity(width, height, m)
        self.P = cauchy_matrix(k, mpar)
        self.groups_per_stripe = k + mpar
        self._linear = {'a': bytearray(), 'b': bytearray()}
        self._pending = {'a': [], 'b': []}
        self._staging = {'a': [], 'b': []}
        self._frames = {'a': asyncio.Queue(), 'b': asyncio.Queue()}
        self._seq = {'a': 0, 'b': 0}
        self._seq_last = {'a': None, 'b': None}
        self._seq_wraps = {'a': 0, 'b': 0}
        self._recv = {}
        self.frames_sent = {'a': 0, 'b': 0}
        self.stripe_sent = {'a': 0, 'b': 0}
        self.resyncs = 0

    def seq_true(self, seq):
        # wrap-aware true index (shared per channel; both directions are
        # monotonic in their own seq space, and the test uses one channel
        # for both, so track per the last-seen). For the real node each
        # direction is a separate channel, so a single tracker is exact.
        return seq

    def _recv_for(self, side):
        if side not in self._recv:
            self._recv[side] = _RecvState(self)
        return self._recv[side]

    # ---- send ----
    def send(self, side, datagram):
        self._linear[side] += len(datagram).to_bytes(2, 'big') + datagram
        lin = self._linear[side]
        while len(lin) >= self.B:
            g = bytes(lin[:self.B])
            del lin[:self.B]
            self._pending[side].append(g)
            if len(self._pending[side]) >= self.k:
                self._emit_stripe(side)

    def _emit_stripe(self, side):
        pg = self._pending[side]
        data = pg[:self.k]          # list of B-byte payloads (bytes)
        del pg[:self.k]
        rows = np.stack([np.frombuffer(p, dtype=np.uint8) for p in data])
        parity = parity_rows(rows, self.P)   # list of (k,) uint8 arrays
        all_payloads = list(data) + [p.tobytes() for p in parity]
        seq0 = self._seq[side]
        self._seq[side] += self.groups_per_stripe
        other = 'b' if side == 'a' else 'a'
        for i, payload in enumerate(all_payloads):
            f = render_group(payload, (seq0 + i) & 0xFFFF, self.m,
                             self.w, self.h)
            for _ in range(self.r):
                self._staging[other].append(f)
        self.stripe_sent[side] += 1

    async def ticker(self, side, stop):
        """The video clock: release exactly one rendered frame every
        1/fps seconds — the cadence a real video service imposes. Without
        it, an in-process queue delivers the whole stripe instantly and
        QUIC's loss detection misbehaves; with it, the channel behaves
        like the real 30 fps pipe."""
        interval = 1.0 / self.fps
        while not stop.is_set():
            st = self._staging[side]
            if st:
                f = st.pop(0)
                self._frames[side].put_nowait(f)
                self.frames_sent[side] += 1
            await asyncio.sleep(interval)

    def flush(self, side):
        """Force-emit everything pending (shutdown / idle): cut the linear
        tail into a zero-padded group, then emit the final partial stripe."""
        lin = self._linear[side]
        if lin:
            lin += b'\x00' * (self.B - len(lin) % self.B)
            while len(lin) >= self.B:
                self._pending[side].append(bytes(lin[:self.B]))
                del lin[:self.B]
        pg = self._pending[side]
        if not pg:
            return
        while len(pg) < self.k:
            pg.append(b'\x00' * self.B)
        while pg:
            self._emit_stripe(side)

    # ---- idle flush ----
    async def idle_flush(self, side, stop, idle=0.1):
        """Emit a pending partial stripe after `idle` seconds without new
        data, so the last datagram (e.g. a stream FIN) is not left stranded
        in the partial buffer. The video clock paces the frames; this just
        pads the tail when the sender goes quiet."""
        import time as _t
        last_n = -1
        since = None
        while not stop.is_set():
            await asyncio.sleep(0.05)
            n = len(self._pending[side]) + len(self._linear[side])
            if n:
                if n == last_n:
                    if since is None:
                        since = _t.monotonic()
                    elif _t.monotonic() - since > idle:
                        self.flush(side)
                        last_n, since = -1, None
                else:
                    last_n, since = n, None
            else:
                last_n, since = -1, None

    # ---- recv ----
    async def recv(self, to_side):
        """Pull one frame addressed to `to_side`, feed it, return the list of
        complete datagrams it produced (may be empty)."""
        frame = await self._frames[to_side].get()
        return self._recv_for(to_side).feed_frame(frame)

    async def pump(self, to_side, on_datagrams, stop, timeout=0.05):
        """Consume frames addressed to `to_side` until `stop`; call
        on_datagrams(list) for each non-empty batch. Cancel-safe (a single
        persistent pending get, never cancelled on timeout)."""
        recv = self._recv_for(to_side)
        get_task = None
        n = 0
        while not stop.is_set():
            q = self._frames[to_side]
            if get_task is None:
                get_task = asyncio.ensure_future(q.get())
            stop_task = asyncio.ensure_future(stop.wait())
            done, _ = await asyncio.wait({get_task, stop_task}, timeout=timeout,
                                         return_when=asyncio.FIRST_COMPLETED)
            stop_task.cancel()
            try:
                await stop_task
            except (asyncio.CancelledError, Exception):
                pass
            if get_task in done:
                frame = get_task.result()
                get_task = None
                dtgs = recv.feed_frame(frame)
                if dtgs:
                    n += len(dtgs)
                    on_datagrams(dtgs)
        return n
