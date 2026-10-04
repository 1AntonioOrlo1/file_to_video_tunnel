#!/usr/bin/env python3
"""QUIC transport validation on this machine: self-signed TLS, one QUIC
connection, bidirectional chunk echo with 4-byte length framing
(the exact framing the tunnel transmitters will use)."""
import asyncio
import ipaddress
import os
import random
import sys
from datetime import datetime, timedelta, timezone

from aioquic.asyncio import serve, connect
from aioquic.quic.configuration import QuicConfiguration
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quic_test')
os.makedirs(OUT, exist_ok=True)
CERT = os.path.join(OUT, 'cert.pem')
KEY = os.path.join(OUT, 'key.pem')
PORT = 9443
CHUNKS = 200
CHUNK_MAX = 4 * 1024


def make_cert():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, 'tunnel-test'),
    ])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=30))
        .add_extension(x509.SubjectAlternativeName([
            x509.DNSName('tunnel-test'),
            x509.IPAddress(ipaddress.ip_address('127.0.0.1')),
        ]), critical=False)
        .sign(key, hashes.SHA256())
    )
    with open(KEY, 'wb') as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()))
    with open(CERT, 'wb') as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    return cert, key


def chunk_stream(chunks):
    out = b''
    for c in chunks:
        out += len(c).to_bytes(4, 'big') + c
    return out


def read_chunk_stream(data, want_n):
    chunks, off = [], 0
    while len(chunks) < want_n and off + 4 <= len(data):
        n = int.from_bytes(data[off:off + 4], 'big')
        off += 4
        chunks.append(data[off:off + n])
        off += n
    return chunks


async def server(stream_reader, stream_writer):
    print('[srv] stream handler called', flush=True)
    buf = b''
    total = 0
    while True:
        d = await stream_reader.read(65536)
        if not d:
            break
        buf += d
        total += len(d)
        # echo every complete chunk back (frame-level echo)
        while len(buf) >= 4:
            n = int.from_bytes(buf[:4], 'big')
            if len(buf) < 4 + n:
                break
            c = buf[4:4 + n]
            buf = buf[4 + n:]
            stream_writer.write(len(c).to_bytes(4, 'big') + c)
            await stream_writer.drain()
    stream_writer.close()


async def main():
    cert, key = make_cert()

    s_cfg = QuicConfiguration(
        is_client=False,
        alpn_protocols=['bitcoder/1'],
        certificate=cert,
        private_key=key,
        max_data=1 << 24,
        max_stream_data=1 << 24,
    )
    c_cfg = QuicConfiguration(
        is_client=True,
        alpn_protocols=['bitcoder/1'],
        server_name='tunnel-test',
        verify_mode=0,  # ssl.CERT_NONE
    )

    srv = await serve('127.0.0.1', PORT, configuration=s_cfg,
                      stream_handler=lambda r, w: asyncio.ensure_future(server(r, w)))
    try:
        chunks = [os.urandom(random.randint(1, CHUNK_MAX)) for _ in range(CHUNKS)]
        data = chunk_stream(chunks)
        got = b''
        done = asyncio.Event()

        async def on_stream(reader, writer):
            nonlocal got
            print('[cli] on_stream called', flush=True)
            while True:
                d = await reader.read(65536)
                if not d:
                    break
                got += d
            writer.close()
            done.set()

        async with connect('127.0.0.1', PORT, configuration=c_cfg,
                           stream_handler=lambda r, w: None) as protocol:
            reader, writer = await protocol.create_stream()
            print(f'[OK] QUIC connection up (ALPN bitcoder/1, TLS 1.3)')
            writer.write(data)
            await writer.drain()
            writer.close()
            # the server echoes into THIS SAME stream; read it here
            while True:
                d = await asyncio.wait_for(reader.read(65536), timeout=30)
                if not d:
                    break
                got += d
    finally:
        srv.close()

    ok = read_chunk_stream(got, CHUNKS) == chunks and len(got) == len(data)
    print(f'[{"OK" if ok else "FAIL"}] round-trip: got {len(got)}/{len(data)} B, '
          f'{CHUNKS} chunks, byte-exact={got == data}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
