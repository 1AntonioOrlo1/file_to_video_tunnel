#!/usr/bin/env python3
"""The real thing: QUIC whose datagrams travel as 7030 VIDEO FRAMES.

  QUIC stream data -> datagrams -> [len16] linear pack -> B-byte groups
  -> 7030 color frames (the video stream) -> decoded back -> datagrams
  -> QUIC stream data

No UDP, no asyncio.Queue: the carrier is a queue of rendered RGB frames,
exactly the kind a video service would transport. Client writes a payload
to a QUIC stream; server echoes it; client verifies byte-exact.

This is the architectural proof for "a VPN tunnel through video services":
QUIC (TLS, streams, retransmits, 0-RTT) runs untouched on top, and the only
thing below it is a stream of color frames. Swap the frame queue for a real
RTMP/SRT/video-service sink and the same code carries the tunnel.

Key lessons baked in (both cost a long debug session):
  * A server-side QUIC connection must be registered under BOTH the initial
    dcid AND the connection's own scid (host_cid), and must track CIDs it
    issues/retires (NEW_CONNECTION_ID) — or the client's responses to the
    server's scid are dropped and the handshake stalls.
  * The frame group layout is [n_valid 2B][real bytes][zero padding] so the
    reassembler never parses padding (no phase/parity hazard); a datagram
    may span any number of frames.
"""
import asyncio
import ipaddress
import os
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

from tunnel_core import R
from fec_frame_channel import FecFrameChannel

CID_LEN = 8
SERVER_NAME = 'bitcoder-video-tunnel'
W, H = 1280, 720   # 720p carrier (B=5376 comfortably holds 1350-B datagrams)
DTG = 1350         # QUIC datagram size
FPS = 30           # the video clock


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
    # The FEC-video channel is reliable by construction (FEC repairs video
    # drops, QUIC retransmits the rest) with a fixed capacity, so a wide
    # flow-control window keeps the pipe fed; and the stripe latency (~0.2
    # s one-way) must be declared up front, or PTO (derived from
    # initial_rtt=0.1) declares in-stripe datagrams "lost" and collapses
    # the cwnd.
    c.initial_rtt = 2.0
    c.max_data = 16 << 20
    c.max_stream_data = 64 << 20
    import video_cc  # noqa: F401  registers the 'video-tunnel' CC
    c.congestion_control_algorithm = "video-tunnel"
    return c


def server_cfg(cert, key):
    c = QuicConfiguration(is_client=False, alpn_protocols=["bitcoder/1"])
    c.certificate = cert
    c.private_key = key
    c.max_datagram_size = DTG
    c.initial_rtt = 2.0
    c.max_data = 16 << 20
    c.max_stream_data = 64 << 20
    import video_cc  # noqa: F401
    c.congestion_control_algorithm = "video-tunnel"
    return c


class VideoTransport:
    """asyncio.Transport stand-in: sendto() packs the datagram into 7030
    frames on the channel. This is the ONLY surgery on aioquic."""

    def __init__(self, chan, side, stats):
        self.chan, self.side, self.stats = chan, side, stats

    def sendto(self, data, addr=None):
        self.stats['dtg'] += 1
        self.stats[f'sent_{self.side}'] = self.stats.get(f'sent_{self.side}', 0) + 1
        self.chan.send(self.side, data)


class VQ:
    """Stock QuicConnectionProtocol bound to a video-frame channel instead of
    a UDP socket. transmit()/datagram_received()/timers are all stock."""

    def __init__(self, quic, chan, side, stats, stream_handler=None):
        self.proto = QuicConnectionProtocol(quic, stream_handler=stream_handler)
        self.proto.connection_made(VideoTransport(chan, side, stats))
        self.chan, self.side = chan, side

    async def pump(self, stop):
        n = [0]

        def on_datagrams(dtgs):
            n[0] += len(dtgs)
            if n[0] <= 5 or n[0] % 100 == 0:
                print(f"  [{self.side}-pump] got {n[0]} datagrams",
                      flush=True)
            for d in dtgs:
                self.proto.datagram_received(d, b'peer')
        await self.chan.pump(self.side, on_datagrams, stop)


class VideoServer:
    """Server side: one QuicConnection per incoming dcid, like aioquic's
    stock QuicServer, but fed by video frames.

    CRITICAL: register each connection under BOTH the initial dcid AND the
    connection's own scid, and track the CIDs it issues/retires — the client
    may address responses to any of them."""

    def __init__(self, chan, stats, stream_handler, cert, key):
        self.cfg = server_cfg(cert, key)
        self.chan, self.stats = chan, stats
        self.stream_handler = stream_handler
        self.protocols, self.quics = {}, {}

    def _new(self, dcid):
        quic = QuicConnection(configuration=self.cfg,
                              original_destination_connection_id=dcid)
        vq = VQ(quic, self.chan, 'b', self.stats,
                stream_handler=self.stream_handler)
        self.protocols[dcid] = vq
        self.protocols[quic.host_cid] = vq
        self.quics[dcid] = quic
        vq.proto._connection_id_issued_handler = partial(
            self._cid_issued, vq=vq)
        vq.proto._connection_id_retired_handler = partial(
            self._cid_retired, vq=vq)
        return vq

    def _cid_issued(self, cid, vq):
        self.protocols[cid] = vq

    def _cid_retired(self, cid, vq):
        if self.protocols.get(cid) is vq:
            self.protocols.pop(cid, None)

    async def pump(self, stop):
        def on_datagrams(dtgs):
            for d in dtgs:
                self.feed(d)
        await self.chan.pump('b', on_datagrams, stop)

    def feed(self, data):
        self.stats.setdefault('feed', 0)
        self.stats['feed'] += 1
        if self.stats['feed'] <= 6 or self.stats['feed'] % 200 == 0:
            print(f"  [srv-feed #{self.stats['feed']}] len={len(data)} "
                  f"conns={len(self.quics)}", flush=True)
        buf = Buffer(data=data)
        try:
            header = pull_quic_header(buf, host_cid_length=CID_LEN)
        except ValueError:
            return
        vq = self.protocols.get(header.destination_cid)
        if vq is None:
            if (len(data) >= SMALLEST_MAX_DATAGRAM_SIZE
                    and header.packet_type == QuicPacketType.INITIAL):
                vq = self._new(header.destination_cid)
            else:
                return
        vq.proto.datagram_received(data, b'peer')


async def main():
    payload = os.urandom(1 * 1024 * 1024)
    chan = FecFrameChannel(W, H, fps=FPS)
    stop = asyncio.Event()
    stats = {'dtg': 0}
    got = []

    def stream_handler(reader, writer):
        asyncio.ensure_future(echo(reader, writer))

    async def echo(reader, writer):
        try:
            # Streaming echo: read in chunks and write them back as they
            # arrive, so flow-control windows open while data is in flight
            # (a full-read echo would hold the 1 MB payload hostage and
            # deadlock against the initial stream window).
            while True:
                chunk = await reader.read(64 << 10)
                if not chunk:
                    break
                got.append(chunk)
                writer.write(chunk)
            writer.write_eof()
        except Exception as e:
            print(f"  [echo] {type(e).__name__}: {e}", flush=True)

    def _watch(name, t):
        def cb(_t):
            if _t.cancelled():
                return
            exc = _t.exception()
            if exc is not None:
                import traceback
                print(f"  [TASK {name} DIED] {exc!r}", flush=True)
                traceback.print_exception(type(exc), exc, exc.__traceback__)
        t.add_done_callback(cb)

    cert, key = make_cert()
    server = VideoServer(chan, stats, stream_handler, cert, key)
    sp = asyncio.ensure_future(server.pump(stop))
    _watch('server-pump', sp)

    quic = QuicConnection(configuration=client_cfg())
    vq = VQ(quic, chan, 'a', stats)
    loop = asyncio.get_running_loop()

    t_start = time.time()
    quic.connect(addr=b'peer', now=loop.time())
    vq.proto.transmit()
    cp = asyncio.ensure_future(vq.pump(stop))
    _watch('client-pump', cp)
    ia = asyncio.ensure_future(chan.idle_flush('a', stop, idle=0.4))
    _watch('idle-a', ia)
    ib = asyncio.ensure_future(chan.idle_flush('b', stop, idle=0.4))
    _watch('idle-b', ib)
    ta = asyncio.ensure_future(chan.ticker('a', stop))   # video clock ->A
    _watch('ticker-a', ta)
    tb = asyncio.ensure_future(chan.ticker('b', stop))   # video clock ->B
    _watch('ticker-b', tb)

    async def status_log():
        while not stop.is_set():
            await asyncio.sleep(2)
            ra, rb = chan._recv.get('a'), chan._recv.get('b')
            print(f"  [stat t={time.time()-t_start:.1f}] "
                  f"lin_a={len(chan._linear['a'])} pend_a={len(chan._pending['a'])} "
                  f"q_b={chan._frames['b'].qsize()} "
                  f"lin_b={len(chan._linear['b'])} pend_b={len(chan._pending['b'])} "
                  f"q_a={chan._frames['a'].qsize()} "
                  f"sent_a={stats.get('sent_a', 0)} sent_b={stats.get('sent_b', 0)} "
                  f"recv_a_groups={ra.groups if ra else 0} "
                  f"recv_b_groups={rb.groups if rb else 0} "
                  f"buf_a={len(ra.buf) if ra else 0} buf_b={len(rb.buf) if rb else 0}",
                  flush=True)
    stt = asyncio.ensure_future(status_log())

    deadline = loop.time() + 30
    while not quic._handshake_complete and loop.time() < deadline:
        await asyncio.sleep(0.02)
    if not quic._handshake_complete:
        print("FAIL: handshake did not complete")
        for t in (sp, cp, ia, ib, ta, tb, stt):
            t.cancel()
        return 1
    hs_time = time.time() - t_start
    print(f"handshake OK over the video-frame channel in {hs_time:.3f} s")

    stream_id = quic.get_next_available_stream_id()
    reader, writer = vq.proto._create_stream(stream_id)
    quic.send_stream_data(stream_id, payload)
    writer.write_eof()
    vq.proto.transmit()

    deadline = loop.time() + 150
    got_echo = b""
    last_log = 0
    while loop.time() < deadline:
        try:
            chunk = await asyncio.wait_for(reader.read(1 << 20), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        if not chunk:
            break                      # EOF fed (stream FIN received)
        got_echo += chunk
        if time.time() - t_start - last_log > 2:
            last_log = time.time() - t_start
            print(f"  [t={time.time()-t_start:.1f}s] echoed so far "
                  f"{len(got_echo)} client_sent={stats.get('sent_a')} "
                  f"server_sent={stats.get('sent_b')} "
                  f"chan_a={chan.frames_sent['a']} chan_b={chan.frames_sent['b']}",
                  flush=True)
    total_time = time.time() - t_start
    echoed = got_echo
    print(f"client got {len(echoed)} bytes (echo)")
    print(f"client at_eof={reader.at_eof()}")
    print(f"sent_a={stats.get('sent_a')} sent_b={stats.get('sent_b')} "
          f"fed_b={stats.get('feed')}")
    try:
        loss = quic._loss
        print(f"client cwnd={loss.congestion_window} "
              f"in_flight={loss.bytes_in_flight} "
              f"srtt={loss._rtt_smoothed:.4f}")
    except Exception as e:
        print(f"(no loss stats: {e})")
    for q, who in ((quic, 'client'), (list(server.quics.values())[0], 'server')):
        print(f"  {who} state={q._state} close={q._close_event}")
        for sid, st in q._streams.items():
            print(f"    {who} stream {sid}: recv_fin={st.receiver.is_finished} "
                  f"recv_highest={st.receiver.highest_offset} "
                  f"recv_buf={len(st.receiver._buffer)} "
                  f"send_fin={st.sender.is_finished} "
                  f"send_buf={len(st.sender._buffer)} "
                  f"send_pend={len(st.sender._pending)}")
    print(f"server got {len(b''.join(got))} bytes")
    print(f"payload    {len(payload)} bytes")
    print(f"carrier    {W}x{H}@{FPS} 7030 frames, FEC k=8 m=2, R={R}; "
          f"{chan.frames_sent['a']} A->B, {chan.frames_sent['b']} B->A")
    st = chan._recv.get('b')
    if st:
        print(f"fec stats  groups={st.groups} repaired={st.repaired} "
              f"dropped_stripes={st.dropped_stripes} bad={st.bad}")
    print(f"wall time  {total_time:.2f} s")

    ok = echoed == payload and b"".join(got) == payload
    print("RESULT:", "PASS" if ok else "FAIL")
    stop.set()
    for t in (sp, cp, ia, ib, ta, tb, stt):
        t.cancel()
    for t in (sp, cp, ia, ib, ta, tb, stt):
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
