#!/usr/bin/env python3
"""Raw QUIC (aioquic) loopback throughput probe — what the transport itself
can move on this machine, before any color codec."""
import asyncio
import ipaddress
import os
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
PORT = 9444


def make_cert():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'tunnel-test')])
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
        f.write(key.private_bytes(serialization.Encoding.PEM,
                                  serialization.PrivateFormat.PKCS8,
                                  serialization.NoEncryption()))
    with open(CERT, 'wb') as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    return cert, key


async def main(mb=300):
    cert, key = make_cert()
    s_cfg = QuicConfiguration(is_client=False, alpn_protocols=['bitcoder/1'],
                              certificate=cert, private_key=key,
                              max_data=1 << 26, max_stream_data=1 << 26,
                              idle_timeout=10.0)
    c_cfg = QuicConfiguration(is_client=True, alpn_protocols=['bitcoder/1'],
                              server_name='tunnel-test', verify_mode=0,
                              max_data=1 << 26, max_stream_data=1 << 26,
                              idle_timeout=10.0)
    srv = await serve('127.0.0.1', PORT, configuration=s_cfg,
                      stream_handler=lambda r, w: asyncio.ensure_future(
                          server(r, w)))
    payload = os.urandom(mb * 1024 * 1024)
    got = 0
    t0 = [None]
    t1 = [None]
    done = asyncio.Event()

    async def server(reader, writer):
        nonlocal got
        while True:
            d = await reader.read(262144)
            if not d:
                break
            if t0[0] is None:
                t0[0] = asyncio.get_running_loop().time()
            got += len(d)
            t1[0] = asyncio.get_running_loop().time()
        writer.close()
        done.set()

    async with connect('127.0.0.1', PORT, configuration=c_cfg,
                       stream_handler=lambda r, w: None) as protocol:
        reader, writer = await protocol.create_stream()
        w0 = asyncio.get_running_loop().time()
        # write in 8 MB slices so the client isn't a single monster write
        off = 0
        while off < len(payload):
            writer.write(payload[off:off + 8 * 1024 * 1024])
            off += 8 * 1024 * 1024
            await writer.drain()
        writer.close()
    await done.wait()
    win = (t1[0] - t0[0]) or 0.001
    print(f'raw QUIC loopback: {got/1e6:.0f} MB in {win:.2f} s = '
          f'{got/win/1024:.0f} KB/s (one direction)')
    srv.close()


if __name__ == '__main__':
    MB = int(os.environ.get('PROBE_MB', 100))
    asyncio.run(main(MB))
