#!/usr/bin/env python3
"""Video-tunnel node: a QUIC/VPN tunnel carried by a video stream.

Layers, bottom to top:

  video service (SRT / RTMP / OBS / file)          <- your --ffmpeg-in/out
  7030 color frames, R=2 temporal copies, FEC k=8 m=2 (Cauchy)
  QUIC datagrams (the unit QUIC tracks & retransmits)
  QUIC (aioquic: TLS 1.3, streams, flow control, 0-RTT)
  mux byte stream (conn | len16 | flags | data packets)
  local TCP connections (the VPN side)

Run two instances, one per end. Each owns:
  * two ffmpeg points:
      --ffmpeg-out "ARGS"  ffmpeg reads rawvideo rgb24 WxH@fps from our
        stdout->its stdin and encodes to a destination (SRT listener,
        RTMP, file...). These are the video-service IN points.
      --ffmpeg-in  "ARGS"  ffmpeg decodes a source (SRT caller, RTMP,
        file...) into rawvideo rgb24 WxH on stdout, which we read.
  * a local TCP side:
      --tcp-srv-port  listen for local connections (data goes through
                      the tunnel to the other end)
      --tcp-cli-host/--tcp-cli-port  dial a local destination; tunnel
                      data arrives here
  * the QUIC middle (client or server, --role): it rides the video
    channel, no ports of its own.

The rawvideo spec must match exactly: -f rawvideo -s WxH -pix_fmt rgb24 -r fps.
"""
import argparse
import asyncio
import ipaddress
import os
import shlex
import sys
import time
from datetime import datetime, timedelta, timezone
from functools import partial

from aioquic.quic.configuration import QuicConfiguration, SMALLEST_MAX_DATAGRAM_SIZE
from aioquic.quic.connection import QuicConnection
from aioquic.quic.packet import QuicPacketType, pull_quic_header
from aioquic.buffer import Buffer
from aioquic.asyncio.protocol import QuicConnectionProtocol

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from tunnel_core import (group_capacity, render_group, decode_group,
                         decode_group_fast, decode_group_grid, R)
from bitcoder_fec import unpack_header, crc_for_group
from mux import pack_packet, PKT_MAX
from demux import Demuxer
import video_cc  # noqa: F401  registers the 'video-tunnel' CC

CID_LEN = 8
SERVER_NAME = 'bitcoder-video-tunnel'
DTG = 1350
STREAM_ID = 0
SEND_BUF_PAUSE = 4 << 20   # pause local readers above this QUIC send buffer
CHUNK = 64 << 10


def make_cert():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, SERVER_NAME)])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=30))
        .add_extension(x509.SubjectAlternativeName([
            x509.DNSName(SERVER_NAME),
            x509.IPAddress(ipaddress.ip_address('127.0.0.1')),
        ]), critical=False)
        .sign(key, hashes.SHA256())
    )
    return cert, key


def client_cfg():
    c = QuicConfiguration(is_client=True, alpn_protocols=["bitcoder/1"],
                          server_name=SERVER_NAME)
    c.verify_mode = 0
    c.max_datagram_size = DTG
    c.initial_rtt = float(os.environ.get('VQIC_INITIAL_RTT', '2.0'))
    c.max_data = 16 << 20
    c.max_stream_data = 64 << 20
    c.congestion_control_algorithm = "video-tunnel"
    return c


def server_cfg(cert, key):
    c = QuicConfiguration(is_client=False, alpn_protocols=["bitcoder/1"])
    c.certificate = cert
    c.private_key = key
    c.max_datagram_size = DTG
    c.initial_rtt = float(os.environ.get('VQIC_INITIAL_RTT', '2.0'))
    c.max_data = 16 << 20
    c.max_stream_data = 64 << 20
    c.congestion_control_algorithm = "video-tunnel"
    return c


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# gated per-stage latency trace (VQIC_TRACE=1): one line per pipeline event
# with a monotonic offset, so the gaps between stages are readable. Capped
# per pid so a long run can't flood the log.
_TRACE_ON = os.environ.get('VQIC_TRACE', '') == '1'
_trace_t0 = time.monotonic()
_trace_n = 0
_TRACE_CAP = int(os.environ.get('VQIC_TRACE_CAP', '300'))


def tr(tag):
    global _trace_n
    if not _TRACE_ON or _trace_n >= _TRACE_CAP:
        return
    _trace_n += 1
    wc = time.strftime('%H:%M:%S') + f'.{int((time.time() % 1) * 1000):03d}'
    print(f"[tr {time.monotonic() - _trace_t0:9.3f} {wc}] {os.getpid() % 100000} {tag}",
          flush=True)


# --------------------------------------------------------------------------
# the video side: ffmpeg in/out + 7030 codec + 30 fps clock
# --------------------------------------------------------------------------

class FrameCodec:
    """7030 frame <-> QUIC-packet codec, one frame at a time.

    Each 7030 frame carries a self-validating header (magic + seq + CRC) and
    a set of COMPLETE QUIC datagrams packed as [len16][data] records. A
    datagram never spans two frames: when the next one does not fit in the
    remaining room the current frame is closed (zero-padded) first, so every
    frame is atomic. That removes the whole sync problem — there is no stream
    phase to keep. A frame is either fully valid or discarded; garbage costs
    exactly itself.

    * The sender renders every logical frame in R consecutive copies
      (the proven 7030 preset, chosen so copies survive x264 smearing;
      R is a CLI knob, default 2).
    * The receiver uses the header: the R copies (same seq) are collected
      and threshold-averaged (decode_group) before decoding; if the next
      seq arrives first, the partial group is emitted with the copies on
      hand — a lone 7030 copy survives CRF 0 with the CRC intact, so a
      lost copy costs nothing.
    * Reliability is QUIC's: if both copies of a frame are lost, its
      datagrams simply never reach the far end, are never ACKed, and QUIC
      retransmits them (carried by later frames). The video layer is a plain
      lossy, self-framing pipe."""

    def __init__(self, width, height, m=8, r=R, grid=False):
        self.w, self.h, self.m, self.r = width, height, m, r
        self.grid = grid        # compact (blocks_y, blocks_x, 3) pipe frames
        self.bx, self.by = width // m, height // m
        self.B = group_capacity(width, height, m)
        self._lin = bytearray()
        self._seq = 1             # seq 0 is reserved for the empty idle group
        self.frames_sent = 0
        self._idle = None         # cached empty group frame
        # receive side
        self._pending = []            # groups awaiting copies, oldest
                                      # first: [(unwrapped_seq, payload, frame)]
        self._pending_since = None
        self._last = -1               # last emitted seq (unwrapped)
        self.groups = self.bad = self.dups = self.averaged = self.idle = 0
        self.resyncs = 0
        self.drops_render = 0   # datagrams dropped on a full render queue
                                # (video layer overloaded; QUIC retransmits)
        self.drops_clock = 0    # data frames dropped by the clock when the
                                # send pipe is backpressured (QUIC retransmits)

    # ---- send ----
    SLOTF = 64   # fixed record slot: [len16][data][zero pad to slot]
    SLOTA = 62   # data bytes per slot (B need not be a multiple of SLOTF;
                 # the trailing partial slot is padding and is dropped by
                 # _extract_frame)

    def _pack_datagram(self, data):
        """One datagram as a run of fixed 64-byte slots. Slot 0 carries
        [total_len 16][first 62B]; continuation slots carry [0][next 62B]
        until the total is reached. A datagram always fits in one frame
        (max 1350 B = 22 slots <= 84)."""
        n = len(data)
        slots = []
        first = data[:self.SLOTA]
        rest = data[self.SLOTA:]
        s = bytearray(self.SLOTF)
        s[0:2] = n.to_bytes(2, 'big')
        s[2:2 + len(first)] = first
        slots.append(bytes(s))
        while len(rest):
            c = rest[:self.SLOTA]
            rest = rest[self.SLOTA:]
            s = bytearray(self.SLOTF)
            s[2:2 + len(c)] = c
            slots.append(bytes(s))
        return b''.join(slots)

    def feed_datagram(self, data):
        """Pack one QUIC datagram into fixed slots; return the B-byte
        payloads now ready (0 or 1). A datagram never spans two frames:
        when its slots do not fit in the remaining room, the current frame
        is closed (zero slots) first. Slots are fixed-width, so a frame is
        a pure array of records — zero padding can never be mistaken for a
        length field."""
        rec = self._pack_datagram(data)
        out = []
        if len(self._lin) + len(rec) > self.B:
            out.append(self._close())          # close the partial frame
        self._lin += rec
        while len(self._lin) >= self.B:
            out.append(self._close())
        return out

    def _close(self):
        """Zero-pad the linear tail to B (whole slots); one frame payload."""
        self._lin += b'\x00' * (self.B - len(self._lin))
        payload = bytes(self._lin)
        del self._lin[:]
        return payload

    def flush(self):
        """Emit the pending partial payload (shutdown/idle): so the last
        datagram (e.g. a stream FIN) is never stranded in the buffer."""
        return [self._close()] if self._lin else []

    def next_seq(self):
        s = self._seq
        self._seq += 1
        self.frames_sent += 1
        # masked header seq: 1..0xFFFF, never 0 (0 is the idle group's
        # header); wraps 0xFFFF -> 1, so the receiver's _unwrap stays valid
        return (s % 0xFFFF) + 1

    def idle_frame(self):
        """A VALID empty 7030 frame: real header (magic + seq 0 + CRC)
        over a zero payload, no datagrams. The video stream is then 100%
        self-validating frames with no black filler at all — the receiver
        sees the seq-0 header, counts it, and moves on."""
        if self._idle is None:
            self._idle = render_group(b'\x00' * self.B, 0, self.m,
                                      self.w, self.h, grid=self.grid)
        return self._idle

    # ---- receive ----
    def _unwrap(self, g):
        """16-bit header seq -> monotonic integer. The sender increments
        forever and masks at render time; the CRC covers the MASKED value,
        so a wrapped seq still verifies. At 30 fps at most one wrap can be
        in flight, so anchor to the last emitted seq and fold the distance."""
        if self._last < 0:
            return g
        last16 = self._last & 0xFFFF
        d = (g - last16) & 0xFFFF
        if d >= 0x8000:
            d -= 0x10000
        return self._last + d

    def feed_frame(self, frame):
        """One raw frame in; complete QUIC datagrams out (in seq order).

        The header makes the frame self-validating: no stream sync needed.
        Up to R copies (same seq) are collected and threshold-averaged
        (the proven 7030 preset — survives x264 smearing at any R); if the
        next seq arrives first, the partial group is emitted with the
        copies on hand — a lone 7030 copy survives CRF 0 with the CRC
        intact, so a lost copy costs nothing. Late copies and decoder CFR
        duplicates are counted and dropped."""
        h8, pl = (decode_group_grid(frame, self.bx, self.by) if self.grid
                  else decode_group_fast(frame, self.m, self.w, self.h))
        if h8 is None:
            self.bad += 1
            return []
        g, ok = unpack_header(h8)
        if not ok or crc_for_group(g, pl) != int.from_bytes(h8[3:7], 'big'):
            self.bad += 1
            return []
        if g == 0:
            self.idle += 1          # the valid empty group (no datagrams)
            return []
        u = self._unwrap(g)
        pend = self._pending
        out = []
        if pend and u == pend[0][0]:
            if len(pend[0][1]) < self.r:
                pend[0][1].append((pl, frame))
                if len(pend[0][1]) == self.r:
                    out = self._flush_group(pend.pop(0))
                    if not pend:
                        self._pending_since = None
            else:
                self.dups += 1      # R copies already; extra is a decoder double
            return out
        if pend and u < pend[0][0]:
            self.dups += 1          # late copy of an already-emitted seq
            return []
        while pend and u > pend[0][0]:
            out.extend(self._flush_group(pend.pop(0)))
        if self.r <= 1:
            return out + self._emit(u, pl)
        import time as _t
        pend.append([u, [(pl, frame)]])
        self._pending_since = _t.monotonic()
        return out

    def _flush_group(self, grp):
        """Emit one pending group: use the FIRST copy that passed CRC.

        feed_frame already verified each copy's full CRC (CRC32 over the
        whole payload), so a CRC-passing copy is bit-exact on its own — no
        averaging needed. Averaging two CRC-valid copies yields the same
        exact payload at 64 ms/group of event-loop cost (1080p), which
        halved the throughput; averaging a smeared copy with a valid one
        almost always fails its own CRC and falls back to the valid copy
        anyway. The R=2 copies are kept as redundancy against video
        corruption (a smeared first copy is replaced by a clean second),
        not as averaging material. Returns the first CRC-valid copy."""
        u, copies = grp
        # copies[0] is always CRC-valid (feed_frame checked it before
        # enqueuing); a later copy is only enqueued if it also passed CRC.
        return self._emit(u, copies[0][0])

    def _emit(self, seq, payload):
        if seq <= self._last:
            return []        # already sent (decoder CFR double / late copy)
        self._last = seq
        self.groups += 1
        return self._extract_frame(payload)

    def _extract_frame(self, payload):
        """Parse one B-byte frame payload as a standalone 64-byte slot
        array. B is NOT always a multiple of SLOTF (1080p: 12126 % 64 =
        30): the trailing partial slot is pure padding (close() zero-pads
        to B, and the pad always covers the tail) and must be dropped
        before the next frame's slots are read. Parsing a continuous
        buffer across frame boundaries skews the slot grid by B % SLOTF
        every frame — 720p (5376) and 4K (48576) are both multiples of 64
        and masked the bug. The sender guarantees a datagram never spans
        two frames (feed_datagram closes the partial frame first), so no
        in-flight datagram state is carried across frames."""
        out = []
        total, acc = 0, bytearray()
        for off in range(0, len(payload) - self.SLOTF + 1, self.SLOTF):
            slot = payload[off:off + self.SLOTF]
            ln = int.from_bytes(slot[:2], 'big')
            body = slot[2:self.SLOTF]
            if total and ln != 0:
                # corruption inside an in-flight datagram: drop the partial
                # one (QUIC retransmits) and treat this slot as a start
                total, acc = 0, bytearray()
            if total:
                acc += body
                if len(acc) >= total:
                    out.append(bytes(acc[:total]))
                    total, acc = 0, bytearray()
                continue
            if ln == 0:
                continue            # padding slot
            if ln > DTG:            # implausible total length: skip slot
                self.resyncs += 1
                continue
            if ln <= self.SLOTA:
                out.append(body[:ln])
            else:
                total, acc = ln, bytearray(body)
        return out

    def drain_pending(self):
        """Force-emit every pending group (stream tail / shutdown). Each
        group already passed CRC on its first copy, so it is exact."""
        if not self._pending:
            return []
        out = []
        for grp in self._pending:
            out.extend(self._flush_group(grp))
        self._pending = []
        self._pending_since = None
        return out

    def drain_if_stale(self, max_wait=0.5):
        """Emit the pending group if its copies have been late by more than
        max_wait seconds. Copies normally follow within R frame ticks
        (~33 ms each), so this only fires at the tail of a burst, when the
        sender went quiet and the last copies never come (idle frames carry
        no seq and cannot resolve a pending group on their own)."""
        import time as _t
        if (self._pending and self._pending_since is not None
                and _t.monotonic() - self._pending_since > max_wait):
            return self.drain_pending()
        return []

class VideoLink:
    """One node's video channel: two ffmpeg processes + 7030 frame codec.

    up:   QUIC datagrams -> FrameCodec -> frame staging (R=2 copies)
          -> (fps clock) -> ffmpeg-out stdin  (our video leaves here)
    down: ffmpeg-in stdout -> frames -> FrameCodec.feed_frame (one at a
          time, R-twins dropped by seq) -> QUIC datagrams
    """

    def __init__(self, args, width, height, fps):
        self.w, self.h, self.fps = width, height, fps
        # VQIC_GRID=1: the pipe carries the COMPACT block grid
        # (W/8 x H/8 x 3 = 390 KB at 4K) instead of the full WxHx3 frame
        # (24.9 MB). ffmpeg upscales grid->WxH (scale=neighbor) before encode
        # and downscales (format=rgb24, scale=neighbor) after decode; the
        # wire stays true WxH and the FEC is byte-exact (verified CRF23).
        # This removes the ~60 ms/frame of pipe I/O that caps 4K at ~9 fps.
        self.grid = os.environ.get('VQIC_GRID', '0') == '1'
        m = 8
        self.fsize = ((width // m) * (height // m) * 3) if self.grid \
            else width * height * 3
        self.args = args
        self.codec = FrameCodec(width, height, m=m,
                                r=getattr(args, 'copies', R), grid=self.grid)
        self.dec = self.codec          # stats live on the codec
        self.staging = []          # frames waiting for the video clock
        # (payload, seq) awaiting render. Bounded: when the video layer
        # cannot keep up (4K render is ~23 ms of numpy per group) the
        # queue fills and emit drops the datagram — it then stays
        # un-ACKed in QUIC, which retransmits it (see _render).
        # Size is tunable (VQIC_RENDER_Q): a QUIC retransmit BURST hands
        # many groups to the video in one loop tick, and a small queue
        # turns that burst into drops -> more retransmits -> a storm
        # (seen at 720p: render_drop 700+, up 2.8x). Payloads are small
        # (B bytes: 5376 at 720p, 48576 at 4K), so even 64 is a few MB.
        self._render_q = asyncio.Queue(
            maxsize=int(os.environ.get('VQIC_RENDER_Q', '128')))
        # staging holds FULL rendered frames (24.9 MB each at 4K). The
        # render thread is faster than the video clock, so without a cap
        # staging grows until it OOMs (4K: ~1.4 GB free RAM). Cap it by
        # bytes: when staging is full the render thread stalls, render_q
        # fills, and emit() drops the datagram -> QUIC retransmit. Memory
        # stays flat regardless of burst size.
        self._staging_cap = int(os.environ.get('VQIC_STAGING_CAP_MB', '256')) << 20
        # in-memory buffer cap for the send-ffmpeg pipe. At 4K one frame is
        # 24.9 MB; without a cap the backpressured pipe would accumulate
        # megabytes per tick and OOM the box (1.4 GB free). When the clock
        # sees the buffer near the cap it skips the write (data frames are
        # then retransmitted by QUIC). asyncio's StreamWriter flushes the
        # buffer via its normal write scheduling — no drain() await needed.
        self._pipe_cap = int(os.environ.get('VQIC_PIPE_CAP_MB', '128')) << 20
        self.up_frames = 0
        self.up_dtg = 0            # QUIC datagrams sent into the video (up)
        self.down_frames = 0
        self.on_datagrams_up = None    # set by the QUIC side
        self.on_datagrams_down = None
        self._procs = []

    async def start(self, on_down):
        self.on_datagrams_down = on_down
        a = self.args
        self._closed = False
        self._out_gen = 0
        self._in_gen = 0
        # stderr -> per-role files (NOT pipes): an undrained stderr pipe
        # fills its 64 KB buffer and blocks ffmpeg, which then looks dead
        # and gets respawned in a loop. Files let ffmpeg run free and give
        # us the real error on post-mortem.
        err_out = open(f'{os.getcwd()}/ffmpeg_out_{a.role}.err', 'w')
        err_in = open(f'{os.getcwd()}/ffmpeg_in_{a.role}.err', 'w')
        self._errf = [err_out, err_in]
        # out: we write rawvideo into ffmpeg's stdin
        self.p_out = await asyncio.create_subprocess_exec(
            'ffmpeg', '-y', '-loglevel', 'error', *shlex.split(a.ffmpeg_out),
            stdin=asyncio.subprocess.PIPE, stderr=err_out)
        # in: ffmpeg writes rawvideo to its stdout
        self.p_in = await asyncio.create_subprocess_exec(
            'ffmpeg', '-y', '-loglevel', 'error', *shlex.split(a.ffmpeg_in),
            stdout=asyncio.subprocess.PIPE, stderr=err_in)
        self._procs = [self.p_out, self.p_in]
        log(f"ffmpeg up/down started (pid {self.p_out.pid}/{self.p_in.pid})")
        self._tasks = [
            asyncio.ensure_future(self._clock()),
            asyncio.ensure_future(self._read_in()),
            asyncio.ensure_future(self._idle_flush()),
            asyncio.ensure_future(self._render()),
            asyncio.ensure_future(self._supervise_out()),
            asyncio.ensure_future(self._supervise_in()),
        ]

    async def _supervise_out(self):
        """Restart ffmpeg-out whenever it exits (SRT caller connect failure,
        dropped link, ...), with backoff, forever. The supervisor is the ONLY
        place p_out/p_in get re-created; the pumps just re-acquire."""
        delay = 1.0
        while True:
            rc = await self.p_out.wait()
            if self._closed:
                return
            log(f"ffmpeg-out exited rc={rc}; respawn in {delay:.0f} s")
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 10.0)
            await self._respawn_out()
            self._out_gen += 1
            delay = 1.0

    async def _supervise_in(self):
        delay = 1.0
        while True:
            rc = await self.p_in.wait()
            if self._closed:
                return
            log(f"ffmpeg-in exited rc={rc}; respawn in {delay:.0f} s")
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 10.0)
            await self._respawn_in()
            self._in_gen += 1
            delay = 1.0

    async def _clock(self):
        """Release one frame per 1/fps into the send ffmpeg — never blocking.

        Frames are self-framed (each carries its own seq+header), so there is
        no pair phase to keep. Idle is a VALID empty 7030 frame (seq 0, zero
        payload) — the stream is a continuous flow of information-bearing
        frames, no black filler.

        The old version did `await stdin.drain()` unconditionally. At 4K the
        rawvideo pipe is 746 MB/s; when the send ffmpeg is backpressured
        (far SRT queue full), drain() parked the clock task FOREVER and the
        whole tunnel deadlocked (the 4K 32 MB test). The CPU was ~90% idle
        the whole time — a pure backpressure deadlock, not a CPU ceiling.

        Now the clock is non-blocking and memory-bounded: it checks the
        in-memory pipe buffer size and, when over the cap, SKIPS the write
        this tick. A skipped data frame means its QUIC datagrams never reach
        the far end, are never ACKed, and QUIC retransmits them (the video
        layer stays a lossy, self-framing pipe). The pipe buffer is flushed
        by the event loop's normal write scheduling (no drain await needed),
        so the loop stays free for QUIC timers and the recv pump.
        """
        interval = 1.0 / self.fps
        gen = self._out_gen
        idle_f = self.codec.idle_frame()
        frame_sz = self.fsize
        while not self._closed:
            if self._out_gen != gen:
                gen = self._out_gen   # ffmpeg-out was respawned: re-acquire
            t0 = asyncio.get_running_loop().time()
            if self.staging:
                f, is_data = self.staging.pop(0), True
            else:
                f, is_data = idle_f, False
            # Backpressure: if the pipe buffer is over the cap, skip this
            # tick's write entirely. Keeps the in-memory buffer bounded
            # (no OOM at 4K) and never blocks the loop. A dropped data
            # frame -> its datagrams retransmit via QUIC. write() flushes
            # on its own (verified: buffer drains without a drain() call),
            # so the transport buffer size is the only thing to check.
            buf = self.p_out.stdin.transport.get_write_buffer_size()
            if buf + frame_sz > self._pipe_cap:
                if is_data:
                    self.codec.drops_clock += 1
                    if self.codec.drops_clock % 300 == 1:
                        log(f"clock: pipe backpressured ({buf/1e6:.0f} MB), "
                            f"dropped data frame (quic retransmit); total "
                            f"{self.codec.drops_clock}")
                await asyncio.sleep(max(0.0, interval -
                                       (asyncio.get_running_loop().time() - t0)))
                continue
            try:
                self.p_out.stdin.write(f)
            except (BrokenPipeError, ConnectionResetError,
                    asyncio.IncompleteReadError):
                # the supervisor will respawn; just wait a tick
                await asyncio.sleep(0.2)
                continue
            except Exception as e:
                log(f"clock: UNCAUGHT {type(e).__name__}: {e}")
                raise
            self.up_frames += 1
            if is_data:
                tr(f"clock_out frame up_frames={self.up_frames} staging={len(self.staging)}")
            if self.up_frames % 300 == 1:
                log(f"clock: wrote {self.up_frames} frames, "
                    f"buf={len(self.staging)}")
            el = asyncio.get_running_loop().time() - t0
            await asyncio.sleep(max(0.0, interval - el))

    async def _respawn_out(self):
        a = self.args
        err = open(f'{os.getcwd()}/ffmpeg_out_{a.role}.err', 'a')
        self.p_out = await asyncio.create_subprocess_exec(
            'ffmpeg', '-y', '-loglevel', 'error', *shlex.split(a.ffmpeg_out),
            stdin=asyncio.subprocess.PIPE, stderr=err)

    async def _read_in(self):
        """Read rawvideo frames from the recv ffmpeg; feed the FEC decoder.
        Re-acquires automatically when the supervisor respawns it."""
        gen = self._in_gen
        while not self._closed:
            if self._in_gen != gen:
                gen = self._in_gen
            try:
                head = await self.p_in.stdout.readexactly(self.fsize)
            except (asyncio.IncompleteReadError,
                    ConnectionResetError, ValueError):
                await asyncio.sleep(0.1)   # supervisor will respawn
                continue
            self.down_frames += 1
            dtgs = self.dec.feed_frame(head)
            if dtgs:
                tr(f"in_frame #{self.down_frames} dtgs={len(dtgs)} bad={self.dec.bad}")
            if dtgs and self.on_datagrams_down:
                self.on_datagrams_down(dtgs)

    async def _respawn_in(self):
        a = self.args
        err = open(f'{os.getcwd()}/ffmpeg_in_{a.role}.err', 'a')
        self.p_in = await asyncio.create_subprocess_exec(
            'ffmpeg', '-y', '-loglevel', 'error', *shlex.split(a.ffmpeg_in),
            stdout=asyncio.subprocess.PIPE, stderr=err)

    async def _idle_flush(self):
        """Emit the encoder's pending partial frame after a quiet period,
        so the last datagram (e.g., a stream FIN) is never stranded.

        This timer is the small-write tail latency: interactive traffic
        (SSH, browsing) arrives in short bursts, and every quiet gap costs
        this delay at EACH end of the tunnel (RTT pays it twice). 0.15 s
        keeps the round trip snappy; a larger value only coalesces more
        small bursts into fewer partially-filled frames — and continuous
        traffic never triggers this timer at all (frames close when full),
        so throughput is untouched. Tune with VQIC_IDLE_FLUSH_S."""
        idle = float(os.environ.get('VQIC_IDLE_FLUSH_S', '0.15'))
        mark = None
        since = None
        while True:
            await asyncio.sleep(0.05)
            n = len(self.codec._lin)
            if n:
                if n == mark and since is None:
                    since = time.monotonic()
                elif n != mark:
                    mark, since = n, time.monotonic()
                if since is not None and time.monotonic() - since > idle:
                    self.emit(self.codec.flush())
                    tr(f"idle_flush EMIT (quiet {time.monotonic()-since:.2f}s)")
                    mark, since = None, None
            else:
                mark, since = None, None
            # a burst's last frame may outlive its twin (idle frames carry
            # no seq); emit a lone pending group after a short quiet gap so
            # the tail of a burst is not held hostage until the next burst
            # arrives. Copies normally follow within R frame ticks (~33 ms
            # each), so 0.15 s (5x the worst-case copy gap) is plenty: it
            # only fires at the tail, when the sender went quiet. Emitting
            # with the single CRC-valid copy on hand is the designed loss
            # path (a valid copy is bit-exact on its own). Tune with
            # VQIC_DRAIN_STALE_S.
            dtgs = self.codec.drain_if_stale(
                float(os.environ.get('VQIC_DRAIN_STALE_S', '0.15')))
            if dtgs and self.on_datagrams_down:
                self.on_datagrams_down(dtgs)

    async def _render(self):
        """Render queued groups off the event loop.

        render_group is pure numpy (~8 ms at 1080p, ~23 ms at 4K) and runs
        here in a worker thread, so a data burst never freezes the loop:
        QUIC timers, the recv pump and the clock keep ticking while 4K
        frames are being painted. Output frames go to `staging` in seq
        order; the clock consumes them at 1/fps. The bounded queue makes
        overload visible (emit drops the datagram -> QUIC retransmit)
        instead of an unbounded staging backlog."""
        while not self._closed:
            try:
                payload, seq = await self._render_q.get()
            except asyncio.CancelledError:
                raise
            loop = asyncio.get_running_loop()
            # Staging holds full 4K frames (24.9 MB each). The render thread
            # can outrun the video clock, so cap staging by bytes: when it is
            # near the cap, stall here (render_q then fills and emit drops
            # -> QUIC retransmit). Keeps memory flat at any burst size. The
            # r copies are one aliased frame, so sum(len(x)) counts them r
            # times; match that in the add term (conservative).
            while (sum(len(x) for x in self.staging) + self.codec.r * self.fsize
                   > self._staging_cap) and not self._closed:
                await asyncio.sleep(0.01)
            try:
                f = await loop.run_in_executor(
                    None, render_group, payload, seq & 0xFFFF,
                    self.codec.m, self.w, self.h, self.grid)
                self.staging.extend([f] * self.codec.r)
            except Exception as e:
                log(f"render: {type(e).__name__}: {e}")
            finally:
                self._render_q.task_done()

    def emit(self, payloads):
        """Queue each B-byte payload for R=2 consecutive 7030 frames.

        Rendering (numpy, ~8 ms at 1080p, ~23 ms at 4K) happens in the
        _render task, OFF the event loop: a 4K burst must never freeze
        QUIC timers or the recv pump. The queue is bounded — when the
        video layer cannot keep up, the datagram is dropped HERE, before
        it ever reached the other end, so it stays un-ACKed and QUIC
        retransmits it (the video layer stays a lossy, self-framing pipe).
        """
        for p in payloads:
            seq = self.codec.next_seq()
            try:
                self._render_q.put_nowait((p, seq))
            except asyncio.QueueFull:
                self.codec.drops_render += 1
                if self.codec.drops_render % 300 == 1:
                    log(f"emit: render queue full, dropped datagram "
                        f"(quic retransmit); total "
                        f"{self.codec.drops_render}")

    def send_datagram(self, data):
        """QUIC -> video: one datagram in, maybe a frame's payload out."""
        self.emit(self.codec.feed_datagram(data))
        self.up_dtg += 1
        tr(f"up_dtg #{self.up_dtg} {len(data)}B lin={len(self.codec._lin)}")

    def send_datagrams(self, dtgs):
        for d in dtgs:
            self.send_datagram(d)

    async def close(self):
        self._closed = True
        for t in self._tasks:
            t.cancel()
        for p in self._procs:
            try:
                p.terminate()
            except Exception:
                pass


# --------------------------------------------------------------------------
# the QUIC side (stock aioquic plumbing over the video link)
# --------------------------------------------------------------------------

class VideoTransport:
    def __init__(self, link):
        self.link = link

    def sendto(self, data, addr=None):
        self.link.send_datagram(data)


class VQ:
    def __init__(self, quic, link, side, stats, stream_handler=None):
        self.proto = QuicConnectionProtocol(quic, stream_handler=stream_handler)
        self.proto.connection_made(VideoTransport(link))
        self.link, self.side = link, side
        self.quic = quic

    def feed(self, dtgs):
        for d in dtgs:
            self.proto.datagram_received(d, b'peer')


class VideoServer:
    def __init__(self, link, cert, key, stream_handler):
        self.cfg = server_cfg(cert, key)
        self.link = link
        self.stream_handler = stream_handler
        self.protocols, self.quics = {}, {}

    def _new(self, dcid):
        quic = QuicConnection(configuration=self.cfg,
                              original_destination_connection_id=dcid)
        vq = VQ(quic, self.link, 'b', None, stream_handler=self.stream_handler)
        self.protocols[dcid] = vq
        self.protocols[quic.host_cid] = vq
        self.quics[dcid] = quic
        self.active = vq
        vq.proto._connection_id_issued_handler = partial(
            self._cid_issued, vq=vq)
        vq.proto._connection_id_retired_handler = partial(
            self._cid_retired, vq=vq)
        # The client creates the tunnel stream itself, so the server never
        # sees stream_handler fire. The protocol DOES fire
        # _stream_reader_handler for every new stream object, including the
        # client-created one — hook that to get our writer.
        vq.proto._stream_reader_handler = partial(
            self._on_new_stream, vq=vq)
        return vq

    def _on_new_stream(self, reader, writer, vq):
        self.active = vq
        if self.stream_handler:
            self.stream_handler(reader, writer)

    def _cid_issued(self, cid, vq):
        self.protocols[cid] = vq

    def _cid_retired(self, cid, vq):
        if self.protocols.get(cid) is vq:
            self.protocols.pop(cid, None)

    def feed(self, data):
        buf = Buffer(data=data)
        try:
            header = pull_quic_header(buf, host_cid_length=CID_LEN)
        except ValueError:
            return
        vq = self.protocols.get(header.destination_cid)
        if vq is None:
            if (len(data) >= SMALLEST_MAX_DATAGRAM_SIZE
                    and header.packet_type == QuicPacketType.INITIAL):
                log("new QUIC connection")
                vq = self._new(header.destination_cid)
            else:
                return
        vq.proto.datagram_received(data, b'peer')


# --------------------------------------------------------------------------
# the local TCP side: mux conns into the QUIC stream, demux back out
# --------------------------------------------------------------------------

class LocalSide:
    """Local TCP endpoints; muxes their bytes into the QUIC stream (up) and
    demuxes the stream (down) back onto the connections."""

    def __init__(self, args, quic_sender, stop, tunnel_ready):
        self.args, self.stop = args, stop
        self.tunnel_ready = tunnel_ready
        self.send_up = quic_sender          # async fn(bytes)
        self.conns = {}                     # id -> {reader, writer, n}
        self.pending = {}                   # id -> [(data, fin)] for conns
        self.next_id = 1                    #   that aren't connected yet
        self.demux = Demuxer()
        self.dial_reader = None
        self.dial_writer = None
        self.n_up = self.n_down = 0

    def _alloc(self):
        i = self.next_id
        self.next_id += 1
        return i

    def assign_dial(self):
        """The dial connection is conn 1 by convention (both ends)."""
        self.dial_id = 1

    async def start(self):
        await self.tunnel_ready.wait()
        tasks = []
        if self.args.tcp_srv_port:
            tasks.append(asyncio.ensure_future(self._serve()))
        if self.args.tcp_cli_port:
            tasks.append(asyncio.ensure_future(self._dial_loop()))
        self._tasks = tasks

    def stop_tasks(self):
        for t in getattr(self, '_tasks', []):
            t.cancel()

    async def _serve(self):
        srv = await asyncio.start_server(self._on_conn,
                                         '127.0.0.1', self.args.tcp_srv_port)
        log(f"local TCP listening on 127.0.0.1:{self.args.tcp_srv_port}")
        async with srv:
            await srv.serve_forever()

    async def _on_conn(self, r, w):
        cid = self._alloc()
        self.conns[cid] = {'w': w, 'n': 0}
        self._flush_pending(cid)
        peer = w.get_extra_info('peername')
        log(f"local conn {cid} from {peer}")
        try:
            while not self.stop.is_set():
                data = await r.read(CHUNK)
                if not data:
                    break
                tr(f"tcp_in cid={cid} {len(data)}B")
                await self._send(cid, data)
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            try:
                await self._send(cid, b'', fin=True)
            except Exception:
                pass
            self.conns.pop(cid, None)
            log(f"local conn {cid} closed ({peer})")
            try:
                w.close()
            except Exception:
                pass

    async def _dial_loop(self):
        while not self.stop.is_set():
            try:
                r, w = await asyncio.open_connection(
                    self.args.tcp_cli_host, self.args.tcp_cli_port)
            except OSError as e:
                log(f"dial {self.args.tcp_cli_host}:{self.args.tcp_cli_port} "
                    f"failed ({e}); retry in 2 s")
                await asyncio.sleep(2)
                continue
            self.assign_dial()
            self.conns[self.dial_id] = {'w': w, 'n': 0}
            self.dial_reader, self.dial_writer = r, w
            self._flush_pending(self.dial_id)
            log(f"dialed {self.args.tcp_cli_host}:{self.args.tcp_cli_port} "
                f"as conn {self.dial_id}")
            try:
                while not self.stop.is_set():
                    data = await r.read(CHUNK)
                    if not data:
                        break
                    await self._send(self.dial_id, data)
            except (ConnectionResetError, BrokenPipeError):
                pass
            finally:
                try:
                    await self._send(self.dial_id, b'', fin=True)
                except Exception:
                    pass
                self.conns.pop(self.dial_id, None)
                self.dial_reader, self.dial_writer = None, None
                log(f"dial conn {self.dial_id} closed; re-dialing")

    async def _send(self, cid, data, fin=False):
        # coarse backpressure: the QUIC send buffer is bounded by the cwnd,
        # but a fast local app can still pile up unacked stream data
        quic = self.send_up.quic
        try:
            st = quic._streams.get(self.send_up.node.stream_id or 0)
            if st and len(st.sender._buffer) > SEND_BUF_PAUSE:
                await asyncio.sleep(0.1)
        except Exception:
            pass
        # mux packets carry a 2-byte length field: slice data to PKT_MAX.
        # A FIN travels with the LAST slice only.
        off = 0
        while off < len(data) or (fin and off == 0):
            chunk = data[off:off + PKT_MAX]
            off += len(chunk)
            last = (off >= len(data))
            pkt = pack_packet(cid, chunk, fin=(fin and last))
            await self.send_up.send(pkt)
            if last:
                break
        self.n_up += len(data)

    def feed_down(self, data):
        """QUIC stream bytes arriving from the other end."""
        self.n_down += len(data)
        tr(f"quic_down {len(data)}B")
        for cid, chunk, fin in self.demux.feed(data):
            self._deliver(cid, chunk, fin)

    def _deliver(self, cid, data, fin):
        c = self.conns.get(cid)
        if c is None:
            # The connection hasn't been established yet (race: tunnel
            # bytes arrive before the dial/srv socket is ready). Buffer
            # instead of dropping — _flush_pending replays when it's up.
            self.pending.setdefault(cid, []).append((data, fin))
            return
        if data:
            c['w'].write(data)
        if fin:
            c['fin'] = True
            asyncio.ensure_future(self._close_local(cid))

    def _flush_pending(self, cid):
        """Replay buffered tunnel data once the local connection is up."""
        for data, fin in self.pending.pop(cid, []):
            c = self.conns.get(cid)
            if c is None:
                self.pending.setdefault(cid, []).append((data, fin))
                return
            if data:
                c['w'].write(data)
            if fin:
                c['fin'] = True
                asyncio.ensure_future(self._close_local(cid))

    async def _close_local(self, cid):
        c = self.conns.pop(cid, None)
        if c:
            try:
                c['w'].close()
            except Exception:
                pass
        log(f"tunnel closed local conn {cid}")


# --------------------------------------------------------------------------
# the node
# --------------------------------------------------------------------------

class Node:
    def __init__(self, args):
        self.args = args
        self.stop = asyncio.Event()
        self.tunnel_ready = asyncio.Event()
        self.link = VideoLink(args, args.width, args.height, args.fps)
        self.stats = {'up': 0, 'down': 0}
        self.quic = None
        self.vq = None
        self.server = None
        self.stream_reader = None
        self.stream_writer = None
        self.stream_id = None
        self.local = None

    async def run(self):
        a = self.args
        # --- QUIC side ---
        if a.role == 'server':
            cert, key = make_cert()
            self.server = VideoServer(self.link, cert, key,
                                      stream_handler=self._stream_cb)
        else:
            self.quic = QuicConnection(configuration=client_cfg())
            self.vq = VQ(self.quic, self.link, 'a', self.stats)

        # video in (recv ffmpeg) -> datagrams -> QUIC
        def down(dtgs):
            self.stats['down'] += len(dtgs)
            if a.role == 'server':
                for d in dtgs:
                    self.server.feed(d)
            else:
                self.vq.feed(dtgs)

        # --- local side ---
        sender = QuicSender(self)
        self.local = LocalSide(a, sender, self.stop, self.tunnel_ready)

        await self.link.start(on_down=down)
        asyncio.ensure_future(self._status())

        if a.role == 'client':
            loop = asyncio.get_running_loop()
            self.quic.connect(addr=b'peer', now=loop.time())
            self.vq.proto.transmit()
            sid = self.quic.get_next_available_stream_id()
            self.stream_id = sid
            self.reader, self.writer = self.vq.proto._create_stream(sid)
            # _pump_stream reads self.stream_reader/writer; unify the names.
            self.stream_reader, self.stream_writer = self.reader, self.writer
            asyncio.ensure_future(self._pump_stream())
            deadline = loop.time() + 60
            while not self.quic._handshake_complete \
                    and loop.time() < deadline:
                await asyncio.sleep(0.05)
            if not self.quic._handshake_complete:
                log("FAIL: QUIC handshake did not complete over the video link")
                await self.link.close()
                return 1
            log("QUIC handshake OK over the video link")
            self.tunnel_ready.set()

        await self.local.start()
        await self.stop.wait()
        log("shutting down")
        self.local.stop_tasks()
        await self.link.close()
        return 0

    def _stream_cb(self, reader, writer):
        log("tunnel stream opened (server)")
        self.stream_reader, self.stream_writer = reader, writer
        asyncio.ensure_future(self._pump_stream())
        self.tunnel_ready.set()

    async def _pump_stream(self):
        """Read the QUIC tunnel stream and demux it to local conns."""
        while True:
            try:
                chunk = await self.stream_reader.read(CHUNK)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log(f"tunnel stream read ended: {type(e).__name__}: {e}")
                return
            if not chunk:
                log("tunnel stream EOF")
                return
            self.local.feed_down(chunk)

    async def _status(self):
        while not self.stop.is_set():
            await asyncio.sleep(5)
            d = self.link.codec
            log(f"stats: up={self.link.up_dtg} dtg down={self.stats['down']} dtg "
                f"vid_out={self.link.up_frames} vid_in={self.link.down_frames} "
                f"groups={d.groups} avg={d.averaged} dups={d.dups} "
                f"bad={d.bad} idle={d.idle} resync={d.resyncs} "
                f"render_drop={d.drops_render} clock_drop={d.drops_clock} "
                f"render_q={self.link._render_q.qsize()} "
                f"local up={self.local.n_up} down={self.local.n_down} B")


class QuicSender:
    """async send(bytes) onto the tunnel QUIC stream (both roles).

    The stream is bidirectional: the mux byte stream of this node's local
    connections goes up it; the other end's mux stream comes down the same
    object."""

    def __init__(self, node):
        self.node = node

    @property
    def quic(self):
        n = self.node
        if n.args.role == 'client':
            return n.quic
        if n.server and n.server.quics:
            return next(iter(n.server.quics.values()))
        return None

    async def send(self, data):
        n = self.node
        q = self.quic
        if q is None:
            return
        if n.args.role == 'client':
            q.send_stream_data(n.stream_id, data)
            n.vq.proto.transmit()
        else:
            if n.stream_writer is None:
                return
            n.stream_writer.write(data)
            vq = next(iter(n.server.protocols.values()), None)
            if vq is not None:
                vq.proto.transmit()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--role', choices=['client', 'server'], required=True)
    ap.add_argument('--ffmpeg-out', required=True,
                    help='ffmpeg args: reads rawvideo rgb24 WxH@fps from stdin '
                         'and encodes to the video service (SRT/RTMP/file). '
                         'e.g.: "-f rawvideo -s 1280x720 -pix_fmt rgb24 -r 30 '
                         '-i - -c:v libx264 -preset ultrafast -tune zerolatency '
                         '-crf 0 -g 30 -keyint_min 30 -bf 0 -f mpegts '
                         'srt://127.0.0.1:9011?mode=caller&passphrase=x"')
    ap.add_argument('--ffmpeg-in', required=True,
                    help='ffmpeg args: decodes the video service into rawvideo '
                         'rgb24 WxH on stdout. e.g.: "-i \'srt://127.0.0.1:9010'
                         '?mode=listener&passphrase=x\' -map 0:v:0 -c:v rawvideo '
                         '-pix_fmt rgb24 -f rawvideo -"')
    ap.add_argument('--tcp-srv-port', type=int, default=0,
                    help='listen locally for VPN connections (muxed through)')
    ap.add_argument('--tcp-cli-host', default='127.0.0.1')
    ap.add_argument('--tcp-cli-port', type=int, default=0,
                    help='dial this local destination (tunnel data arrives here)')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--copies', type=int, default=R,
                    help='temporal copies per frame (R of the 7030 preset; '
                         'default 2; copies are averaged at decode)')
    return ap.parse_args()


async def amain():
    args = parse_args()
    node = Node(args)
    try:
        return await node.run()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(amain()))
    except KeyboardInterrupt:
        pass
