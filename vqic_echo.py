#!/usr/bin/env python3
"""Proof: aioquic QuicConnection runs over an arbitrary datagram channel
(two asyncio.Queue "pipes"), not real UDP. This is exactly what the video
tunnel does — frames are the datagram carrier.

Client creates a stream, writes a payload, closes it; server echoes the
bytes back on the same stream; client verifies byte-exact match.

Key lesson (learned the hard way): a server-side QUIC connection must be
registered under BOTH the initial dcid AND the connection's own scid
(host_cid) — exactly like the stock QuicServer — or the client's
responses addressed to the server's scid are dropped and the handshake
stalls in an INITIAL-echo loop.
"""
import asyncio
import ipaddress
import os
from datetime import datetime, timedelta, timezone

from aioquic.buffer import Buffer
from aioquic.quic.configuration import QuicConfiguration, SMALLEST_MAX_DATAGRAM_SIZE
from aioquic.quic.connection import QuicConnection
from aioquic.quic.packet import QuicPacketType, pull_quic_header
from aioquic.asyncio.protocol import QuicConnectionProtocol

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

CID_LEN = 8
SERVER_NAME = 'bitcoder-video-tunnel'


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


def make_client_config():
    cfg = QuicConfiguration(
        is_client=True,
        alpn_protocols=["bitcoder/1"],
        server_name=SERVER_NAME,
    )
    cfg.verify_mode = 0
    cfg.max_datagram_size = 1350
    return cfg


def make_server_config(cert, key):
    cfg = QuicConfiguration(
        is_client=False,
        alpn_protocols=["bitcoder/1"],
    )
    cfg.certificate = cert
    cfg.private_key = key
    cfg.max_datagram_size = 1350
    return cfg


class Pipe:
    """A virtual datagram channel: two asyncio.Queue, one per direction.

    recv(side) returns the queue addressed TO `side` (datagrams the other
    side sent us)."""

    def __init__(self):
        self.a_to_b = asyncio.Queue()
        self.b_to_a = asyncio.Queue()
        self.n = {"a_to_b": 0, "b_to_a": 0, "bytes": 0}

    def send(self, from_side, data):
        key = "a_to_b" if from_side == "a" else "b_to_a"
        self.n[key] += 1
        self.n["bytes"] += len(data)
        (self.a_to_b if from_side == "a" else self.b_to_a).put_nowait(data)

    def recv(self, to_side):
        return (self.a_to_b if to_side == "b" else self.b_to_a).get()


class VirtualTransport:
    """asyncio.Transport stand-in: sendto() routes into the Pipe. This is
    the ONLY surgery needed on aioquic's stock protocol."""

    def __init__(self, pipe: Pipe, side: str, stats: dict):
        self.pipe = pipe
        self.side = side
        self.stats = stats

    def sendto(self, data, addr=None):
        self.stats["sent"] += 1
        self.pipe.send(self.side, data)


class VirtualQuic:
    """A stock QuicConnectionProtocol bound to a Pipe instead of a UDP
    socket. transmit()/datagram_received()/timers are all stock aioquic."""

    def __init__(self, quic, pipe, side, stats, stream_handler=None):
        self.proto = QuicConnectionProtocol(quic, stream_handler=stream_handler)
        self.proto.connection_made(VirtualTransport(pipe, side, stats))
        self.pipe = pipe
        self.side = side

    async def pump(self, stop: asyncio.Event):
        while not stop.is_set():
            try:
                data = await asyncio.wait_for(self.pipe.recv(self.side),
                                              timeout=0.05)
            except asyncio.TimeoutError:
                continue
            self.proto.datagram_received(data, b"peer")


class VirtualQuicServer:
    """Server side: one QuicConnection per incoming dcid, like aioquic's
    stock QuicServer, but fed from the Pipe.

    CRITICAL: register each connection under BOTH the initial dcid AND the
    connection's own scid (host_cid) — the client addresses its responses
    to the server's scid."""

    def __init__(self, pipe, stats, stream_handler, cert, key):
        self.cfg = make_server_config(cert, key)
        self.pipe = pipe
        self.stats = stats
        self.stream_handler = stream_handler
        self.protocols = {}
        self.quics = {}

    def _new_connection(self, dcid):
        quic = QuicConnection(configuration=self.cfg,
                              original_destination_connection_id=dcid)
        vq = VirtualQuic(quic, self.pipe, "b", self.stats,
                         stream_handler=self.stream_handler)
        self.protocols[dcid] = vq
        self.protocols[quic.host_cid] = vq
        self.quics[dcid] = quic
        return vq

    async def pump(self, stop: asyncio.Event):
        while not stop.is_set():
            try:
                data = await asyncio.wait_for(self.pipe.recv("b"),
                                              timeout=0.05)
            except asyncio.TimeoutError:
                continue
            self.feed(data)

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
                vq = self._new_connection(header.destination_cid)
            else:
                return
        vq.proto.datagram_received(data, b"peer")


async def main():
    payload = os.urandom(512 * 1024)  # 512 KB
    pipe = Pipe()
    stop = asyncio.Event()
    stats = {"sent": 0}
    got = []

    def stream_handler(reader, writer):
        asyncio.ensure_future(echo(reader, writer))

    async def echo(reader, writer):
        data = await reader.read()
        got.append(data)
        writer.write(data)
        writer.write_eof()

    cert, key = make_cert()
    server = VirtualQuicServer(pipe, stats, stream_handler, cert, key)
    sp = asyncio.ensure_future(server.pump(stop))

    quic = QuicConnection(configuration=make_client_config())
    vq = VirtualQuic(quic, pipe, "a", stats)
    loop = asyncio.get_running_loop()

    quic.connect(addr=b"peer", now=loop.time())
    vq.proto.transmit()
    cp = asyncio.ensure_future(vq.pump(stop))

    deadline = loop.time() + 20
    while not quic._handshake_complete and loop.time() < deadline:
        await asyncio.sleep(0.02)
    if not quic._handshake_complete:
        print("FAIL: handshake did not complete")
        stop.set()
        sp.cancel(); cp.cancel()
        return 1
    print("handshake OK over the virtual datagram channel")

    stream_id = quic.get_next_available_stream_id()
    reader, writer = vq.proto._create_stream(stream_id)
    quic.send_stream_data(stream_id, payload)
    writer.write_eof()
    vq.proto.transmit()

    deadline = loop.time() + 60
    while not reader.at_eof() and loop.time() < deadline:
        await asyncio.sleep(0.05)
    echoed = reader._buffer.copy()
    print(f"client got {len(echoed)} bytes (echo)")
    print(f"server got {len(b''.join(got))} bytes")
    print(f"payload    {len(payload)} bytes")
    print(f"datagrams on the pipe: {stats['sent']}")

    ok = echoed == payload and b"".join(got) == payload
    print("RESULT:", "PASS" if ok else "FAIL")
    stop.set()
    sp.cancel(); cp.cancel()
    for t in (sp, cp):
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
