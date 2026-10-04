import asyncio, ipaddress, os, sys, time
from datetime import datetime, timedelta, timezone
from aioquic.asyncio import serve, connect
from aioquic.quic.configuration import QuicConfiguration
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

sys.setswitchinterval(0.001)
key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 't')])
now = datetime.now(timezone.utc)
cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
    .public_key(key.public_key()).serial_number(x509.random_serial_number())
    .not_valid_before(now - timedelta(minutes=5)).not_valid_after(now + timedelta(days=1))
    .add_extension(x509.SubjectAlternativeName([x509.DNSName('t')]), critical=False)
    .sign(key, hashes.SHA256()))

got = [0]

async def server(reader, writer):
    print('srv: handler started', flush=True)
    t0 = None
    while True:
        d = await reader.read(262144)
        if not d:
            break
        if t0 is None:
            t0 = time.time()
        got[0] += len(d)
    print(f'srv: closed, got {got[0]} B in {time.time()-t0:.1f} s = {got[0]/(time.time()-t0)/1024:.0f} KB/s', flush=True)
    writer.close()

async def main():
    s_cfg = QuicConfiguration(is_client=False, alpn_protocols=['bitcoder/1'],
                              certificate=cert, private_key=key,
                              max_data=1 << 26, max_stream_data=1 << 26, idle_timeout=10.0)
    c_cfg = QuicConfiguration(is_client=True, alpn_protocols=['bitcoder/1'],
                              server_name='t', verify_mode=0,
                              max_data=1 << 26, max_stream_data=1 << 26, idle_timeout=10.0)
    srv = await serve('127.0.0.1', 9445, configuration=s_cfg,
                      stream_handler=lambda r, w: asyncio.ensure_future(server(r, w)))
    print('srv: listening', flush=True)
    payload = os.urandom(20 * 1024 * 1024)
    async with connect('127.0.0.1', 9445, configuration=c_cfg) as proto:
        print('cli: connected', flush=True)
        reader, writer = await proto.create_stream()
        t0 = time.time()
        off = 0
        while off < len(payload):
            writer.write(payload[off:off + 8 * 1024 * 1024])
            off += 8 * 1024 * 1024
            await writer.drain()
        print(f'cli: wrote {off} B in {time.time()-t0:.1f} s', flush=True)
        writer.close()
        # keep reading until EOF so the server can finish
        total = 0
        while True:
            d = await reader.read(262144)
            if not d:
                break
            total += len(d)
        print(f'cli: stream ended, echoed {total} B', flush=True)
    srv.close()

asyncio.run(main())
