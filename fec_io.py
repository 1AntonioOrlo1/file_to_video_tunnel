"""FEC codec split into two independent ends for the video-tunnel node.

The node talks to ffmpeg in rawvideo rgb24 frames over stdin/stdout. One
direction of the tunnel needs each half separately:

  FecEncoder  : datagram in  -> raw frame bytes out   (local -> service)
  FecDecoder  : raw frame in -> datagram out          (service -> local)

They share the 7030 + Cauchy-FEC preset (k=8 m=2, R=2 temporal copies) but
are decoupled from asyncio and from each other, so the node can drive them
directly with ffmpeg pipes.

Frame layout (one video slot):
  W x H rgb24 (W*H*3 bytes) carrying one FEC group (header 8B + payload).
  Each group is drawn in R CONSECUTIVE slots (temporal redundancy).
  A full stripe = (k+m) groups = (k+m)*R consecutive slots.

Idle slots (no data pending) are emitted as a pure-black frame; the decoder
rejects any frame whose 7030 header fails the magic/CRC check, so black
frames are naturally skipped. Idle is emitted in runs of R slots to keep the
decoder's R-copy grouping aligned.
"""
import numpy as np

from tunnel_core import (group_capacity, render_group, decode_group, R, K,
                         M_PAR, parity_rows, stripe_data)
from bitcoder_fec import cauchy_matrix, crc_for_group, unpack_header

IDLE_SEQ = 0xFFFF


def idle_frame(width, height):
    """A frame the decoder will always reject (pure black -> no valid
    7030 header). One video slot's worth of raw rgb24 bytes."""
    return b'\x00' * (width * height * 3)


class FecEncoder:
    """datagrams in -> raw 7030 frames out (paced by the caller)."""

    def __init__(self, width, height, m=8, k=K, mpar=M_PAR, r=None,
                 max_dtg=1350):
        self.w, self.h, self.m = width, height, m
        self.k, self.mpar = k, mpar
        self.r = r if r is not None else R
        self.max_dtg = max_dtg
        self.B = group_capacity(width, height, m)
        self.P = cauchy_matrix(k, mpar)
        self.G = k + mpar
        self._linear = bytearray()
        self._pending = []
        self._seq = 0
        self.stripe_count = 0

    def feed_datagram(self, datagram):
        """Pack one datagram; return the list of raw frames now ready to
        emit (a full stripe's worth when one completes, else empty)."""
        out = []
        self._linear += len(datagram).to_bytes(2, 'big') + datagram
        lin = self._linear
        while len(lin) >= self.B:
            self._pending.append(bytes(lin[:self.B]))
            del lin[:self.B]
            if len(self._pending) >= self.k:
                out.extend(self._emit_stripe())
        return out

    def _emit_stripe(self):
        data = self._pending[:self.k]
        del self._pending[:self.k]
        rows = np.stack([np.frombuffer(p, dtype=np.uint8) for p in data])
        parity = parity_rows(rows, self.P)
        all_payloads = list(data) + [p.tobytes() for p in parity]
        seq0 = self._seq
        self._seq += self.G
        frames = []
        for i, payload in enumerate(all_payloads):
            f = render_group(payload, (seq0 + i) & 0xFFFF, self.m,
                             self.w, self.h)
            for _ in range(self.r):
                frames.append(f)
        self.stripe_count += 1
        return frames

    def flush(self):
        """Emit everything pending (stream end): pad the linear tail into a
        group, then emit the final partial stripe (zero groups pad to k)."""
        lin = self._linear
        if lin:
            lin += b'\x00' * (self.B - len(lin) % self.B)
            while len(lin) >= self.B:
                self._pending.append(bytes(lin[:self.B]))
                del lin[:self.B]
        out = []
        while self._pending:
            while len(self._pending) < self.k:
                self._pending.append(b'\x00' * self.B)
            out.extend(self._emit_stripe())
        return out


class FecDecoder:
    """raw 7030 frames in -> complete datagrams out (in stream order)."""

    def __init__(self, width, height, m=8, k=K, mpar=M_PAR, r=None,
                 max_dtg=1350):
        self.w, self.h, self.m = width, height, m
        self.k, self.mpar = k, mpar
        self.r = r if r is not None else R
        self.max_dtg = max_dtg
        self.B = group_capacity(width, height, m)
        self.P = cauchy_matrix(k, mpar)
        self.G = k + mpar
        self._frames = []
        self._stripe_no = None
        self._stripe_recv = {}
        self._buf = bytearray()
        self._last = None
        self._wraps = 0
        # Idempotence against QUIC retransmissions: a retransmitted packet
        # re-renders the SAME FEC stripe, and a lost-group stripe that was
        # zero-flushed may later be completed by the retransmission. Both
        # must NOT re-emit bytes into the linear stream (that would
        # duplicate/shift it). Remember every finished stripe.
        self._emitted_stripes = set()
        self.dups = 0
        self.last_seq = None
        self.last_stripe = None
        self.groups = self.repaired = self.dropped = self.bad = 0
        self.resyncs = 0

    def _true_idx(self, seq):
        if self._last is not None and seq < self._last:
            self._wraps += 1
        self._last = seq
        return self._wraps * 65536 + seq

    def feed_frame(self, frame):
        """Feed one raw frame; return the list of complete datagrams it
        produced (may be empty). Idle/black frames are rejected here."""
        self._frames.append(frame)
        out = []
        while len(self._frames) >= self.r:
            rf = self._frames[:self.r]
            del self._frames[:self.r]
            h8, payload = decode_group(rf, self.m, self.w, self.h)
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
            self.last_seq = g
            ti = self._true_idx(g)
            self.last_stripe = ti // self.G
            s = ti // self.G
            idx = ti % self.G
            if self._stripe_no is None:
                self._stripe_no = s
                self._stripe_recv = {}
            elif s != self._stripe_no:
                out.extend(self._flush())
                self._stripe_no = s
                self._stripe_recv = {}
            if idx < self.G and idx not in self._stripe_recv:
                self._stripe_recv[idx] = payload
            if len(self._stripe_recv) >= self.G:
                out.extend(self._flush())
                self._stripe_no = None
                self._stripe_recv = {}
        return out

    def _flush(self):
        recv = [self._stripe_recv.get(i) for i in range(self.G)]
        n_present = sum(1 for x in recv if x is not None)
        n_lost = self.G - n_present
        if n_present >= self.k:
            rows = stripe_data(recv, self.k, self.P)
            data = [r.tobytes() for r in rows]
            if n_lost:
                self.repaired += n_lost
        else:
            self.dropped += 1
            data = [b'\x00' * self.B for _ in range(self.k)]
        for p in data:
            self._buf += p
        return self._extract()

    def _extract(self):
        out = []
        resyncs = 0
        while len(self._buf) >= 2:
            ln = int.from_bytes(self._buf[:2], 'big')
            if ln == 0:
                del self._buf[:2]
                continue
            if ln > self.max_dtg:
                del self._buf[:1]
                resyncs += 1
                continue
            if len(self._buf) < 2 + ln:
                break
            out.append(bytes(self._buf[2:2 + ln]))
            del self._buf[:2 + ln]
        if resyncs:
            self.resyncs += resyncs
        return out
