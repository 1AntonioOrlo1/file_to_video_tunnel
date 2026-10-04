#!/usr/bin/env python3
"""Datagram <-> 7030-frame channel (the carrier for "QUIC inside a video").

QUIC datagrams are packed LINEARLY as [len16][data] records into a per-side
byte buffer. A TICKER (paced at `fps`, the video clock) emits one 7030 color
frame per tick. Each frame's B-byte group payload is laid out as:

    [n_valid: 2B BE] [real bytes: n_valid] [zero padding: B - 2 - n_valid]

where `real bytes` is the head of the send buffer (up to B-2 bytes, which may
cut a record — the reassembler carries the tail). The receiver decodes the
frame, verifies the group CRC, reads n_valid, and appends exactly those real
bytes to its reassembly buffer. Padding is NEVER parsed, so there is no
phase/parity hazard: the stream is a pure concatenation of [len16][data]
records, and a datagram may span any number of frames.

The rendered frames ARE the video stream (the transport). A frame is a
self-contained boundary (own seq + CRC), so a lost frame is a bounded
erasure — FEC/MDS on top of the video repairs it; without FEC the datagram
is an erasure, not a stream-wide resync failure.

Constraint for the no-waste steady state:
  B = group_capacity(w,h) >= 2 + max QUIC datagram size,
so 720p (B=5376) comfortably carries 1350-byte datagrams.
"""
import asyncio

from tunnel_core import group_capacity, render_group, decode_group_fast
from bitcoder_fec import crc_for_group


class FrameChannel:
    def __init__(self, width, height, m=8, fps=30):
        self.w, self.h, self.m, self.fps = width, height, m, fps
        self.B = group_capacity(width, height, m)
        assert self.B >= 3, "geometry too small for the [n_valid] header"
        self._send_buf = {'a': bytearray(), 'b': bytearray()}
        self._frames = {'a': asyncio.Queue(), 'b': asyncio.Queue()}
        self._recv_buf = {'a': bytearray(), 'b': bytearray()}
        self._seq = {'a': 0, 'b': 0}
        self.frames_sent = {'a': 0, 'b': 0}
        self.bytes_in = {'a': 0, 'b': 0}
        self.bytes_out = {'a': 0, 'b': 0}

    def send(self, side, datagram):
        """Append one complete outgoing datagram to the send buffer. Frame
        emission is driven by the ticker (tick), not here."""
        self.bytes_in[side] += len(datagram)
        self._send_buf[side] += len(datagram).to_bytes(2, 'big') + datagram

    def tick(self, side):
        """Emit one frame from `side`'s buffer (the paced video writer)."""
        buf = self._send_buf[side]
        take = min(self.B - 2, len(buf))
        real = bytes(buf[:take])
        if take:
            del buf[:take]
        # group layout: [n_valid 2B][real bytes: take][zero padding]
        payload = (take.to_bytes(2, 'big') + real
                   + b'\x00' * ((self.B - 2) - take))
        seq = self._seq[side] & 0xFFFF
        self._seq[side] += 1
        frame = render_group(payload, seq, m=self.m,
                             width=self.w, height=self.h)
        other = 'b' if side == 'a' else 'a'
        self._frames[other].put_nowait(frame)
        self.frames_sent[side] += 1

    async def ticker(self, side, stop):
        """Paced frame emission at `fps` (the video clock)."""
        interval = 1.0 / self.fps
        while not stop.is_set():
            self.tick(side)
            await asyncio.sleep(interval)

    async def recv(self, to_side):
        """Pull one frame addressed to `to_side`, decode it, and return the
        list of complete datagrams reassembled from its real bytes (plus any
        carried tail). Empty list = idle frame or a partial record.

        WARNING: this awaits queue.get() directly; wrap it with a
        stop-aware reader (SideReader), NOT asyncio.wait_for — wait_for
        cancels the get() on timeout and a cancelled get() can drop the
        frame it was about to deliver (that is how the stream's FIN got
        lost in the echo test)."""
        frame = await self._frames[to_side].get()
        return await self._decode_frame(frame, to_side)

    async def _decode_frame(self, frame, to_side):
        header8, payload = decode_group_fast(frame, self.m, self.w, self.h)
        if header8 is None or payload is None:
            return []
        seq = int.from_bytes(header8[1:3], 'big')
        if crc_for_group(seq, payload) != int.from_bytes(header8[3:7], 'big'):
            return []
        n_valid = int.from_bytes(payload[:2], 'big')
        if n_valid > self.B - 2:      # corrupted frame, CRC should have caught it
            return []
        real = payload[2:2 + n_valid]
        self.bytes_out[to_side] += n_valid
        buf = self._recv_buf[to_side]
        buf += real
        out = []
        while len(buf) >= 2:
            ln = int.from_bytes(buf[:2], 'big')
            if ln == 0:               # safety: never a real datagram
                del buf[:2]
                continue
            if len(buf) < 2 + ln:     # record spans into the next frame
                break
            out.append(bytes(buf[2:2 + ln]))
            del buf[:2 + ln]
        return out


class SideReader:
    """Stop-aware frame reader for ONE direction. Keeps a SINGLE persistent
    pending queue.get() per side and NEVER cancels it on a timeout — only a
    shutdown does. This is what eliminates the wait_for cancel-drop hazard
    that ate the stream's FIN datagram."""

    def __init__(self, chan, to_side):
        self.chan = chan
        self.to_side = to_side
        self._get_task = None

    async def next(self, stop, timeout=0.05):
        """Return a list of datagrams (possibly empty) or None on timeout /
        stop."""
        if stop.is_set():
            return None
        q = self.chan._frames[self.to_side]
        if self._get_task is None:
            self._get_task = asyncio.ensure_future(q.get())
        stop_task = asyncio.ensure_future(stop.wait())
        done, _ = await asyncio.wait(
            {self._get_task, stop_task}, timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED)
        stop_task.cancel()
        try:
            await stop_task
        except (asyncio.CancelledError, Exception):
            pass
        if self._get_task in done:
            frame = self._get_task.result()
            self._get_task = None
            return await self.chan._decode_frame(frame, self.to_side)
        # timed out: keep self._get_task pending for the next call
        return None

    async def pump(self, on_datagrams, stop, timeout=0.05):
        """Consume frames until `stop`, calling on_datagrams(list) for each
        non-empty batch. Returns the total number of datagrams delivered."""
        n = 0
        while not stop.is_set():
            dtgs = await self.next(stop, timeout)
            if dtgs is None:
                continue
            if dtgs:
                n += len(dtgs)
                on_datagrams(dtgs)
        return n
