#!/usr/bin/env python3
"""bitcoder QUIC tunnel node.

One program, run as TWO instances = two live video channels, one per
direction. The PROTOCOL (QUIC, TLS 1.3, ALPN bitcoder/1) establishes the
connection itself and owns reconnection; the node only feeds it
frame-by-frame:

  local TCP clients -> mux (7030 packet format) -> color blocks
  (render_group: 8-corner palette, 8B header w/ CRC) -> Cauchy FEC stripes
  (k data + m parity groups) -> length-framed QUIC stream records
  [len u32 BE][tag u8][frame u8...]

The peer node does the reverse: QUIC record -> color-block decode ->
FEC repair (any m lost records of a stripe are reconstructed) -> demux
-> local TCP.

QUIC guarantees ordered, reliable, encrypted delivery per stream, so a
frame (one codec block) always arrives whole and in order. The 7030
FEC layer stays as the second safety net on top.

Resolutions scale with the canvas: payload per frame B =
(blocks - HEADER_BITS) * 3 / 8, so 640x360 / 720p / 1080p / 4K all fit
a whole payload block into one frame.
"""

import argparse
import asyncio
import ipaddress
import logging
import os
import signal
from collections import deque
from datetime import datetime, timedelta, timezone

import numpy as np
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from aioquic.asyncio import serve, connect
from aioquic.quic.configuration import QuicConfiguration

from tunnel_core import (K, M, M_PAR, R, R_META, EOF_PAYLOAD,
                         group_capacity, throughput_bps,
                         cauchy_matrix, render_group, render_meta,
                         decode_group, decode_group_fast, decode_meta,
                         verify_group, stripe_of, stripe_data, parity_rows,
                         SeqWrapTracker)

# QUIC delivers every record byte-exact and reliably, so the R=2 frame
# copies (a lossy-transport trick: threshold-averaging smeared x264 block
# boundaries) buy nothing here — they double both the wire payload and the
# receive-side decode_group CPU, which throttles the whole channel. Use one
# copy per group; the 7030 FEC (k=8+m=2) stays as the safety net.
RQ = 1
from mux import Muxer
from demux import Demuxer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('quic-node')

ALPN = ['bitcoder/1']
CERT_CN = 'bitcoder-tunnel'

T_META = 1
T_GROUP = 2
T_EOF = 3

FLUSH_IDLE_AFTER = 0.2        # flush a stalled partial stripe after this
FLUSH_HARD_AFTER = 1.5        # absolute cap: a client may vanish mid-stripe
REDIAL_WAIT = 2.0
MAX_BUFFER = 8 * 1024 * 1024
LOCAL_QUEUE_MAX = 4     # local clients waiting for peer data (per node)


def make_selfsigned(out_dir):
    """Generate (or reuse) a self-signed cert for the QUIC server."""
    os.makedirs(out_dir, exist_ok=True)
    cert_p = os.path.join(out_dir, 'quic_cert.pem')
    key_p = os.path.join(out_dir, 'quic_key.pem')
    if os.path.exists(cert_p) and os.path.exists(key_p):
        cert = x509.load_pem_x509_certificate(open(cert_p, 'rb').read())
        key = serialization.load_pem_private_key(open(key_p, 'rb').read(),
                                                 password=None)
        return cert, key
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, CERT_CN)])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(
            [x509.DNSName(CERT_CN),
             x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]),
            critical=False)
        .sign(key, hashes.SHA256()))
    open(key_p, 'wb').write(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()))
    open(cert_p, 'wb').write(cert.public_bytes(serialization.Encoding.PEM))
    return cert, key


import threading

_ZSTD_LOCAL = threading.local()   # ZstdCompressor is NOT thread-safe


def _zstd_c():
    c = getattr(_ZSTD_LOCAL, 'c', None)
    if c is None:
        import zstandard
        c = zstandard.ZstdCompressor(level=3)
        _ZSTD_LOCAL.c = c
    return c


_ZSTD_D_LOCAL = threading.local()   # ZstdDecompressor is NOT thread-safe


def _zstd_d():
    d = getattr(_ZSTD_D_LOCAL, 'd', None)
    if d is None:
        import zstandard
        d = zstandard.ZstdDecompressor()
        _ZSTD_D_LOCAL.d = d
    return d


def record(tag, frame, compress=True):
    """Length-framed QUIC stream record: [len u32 BE][tag u8][flag u8][body].

    flag 0 = raw, 1 = zstd-compressed body. The color frames are ~93%
    0/255, so zstd level 3 shrinks a 4K frame 67x (24.9 MB -> 0.37 MB) and
    — unlike zlib — the C binding releases the GIL while compressing, so a
    full thread pool of renderers/decoders cannot starve the event loop
    (GIL-holding zlib once collapsed the QUIC congestion window via UDP
    drops and pinned 4K at ~10% of theoretical)."""
    if compress:
        body = _zstd_c().compress(frame)
        if len(body) + 1 >= len(frame):
            flag, body = 0, frame
        else:
            flag, body = 1, body
    else:
        flag, body = 0, frame
    payload = bytes([tag, flag]) + body
    return len(payload).to_bytes(4, 'big') + payload


def decode_record_body(rec):
    """Invert record(): returns (tag, body)."""
    if rec[1] == 1:
        return rec[0], _zstd_d().decompress(rec[2:], max_output_size=1 << 26)
    return rec[0], rec[2:]


def make_parity(payloads, P, m):
    arr = np.stack([np.frombuffer(p, dtype=np.uint8) for p in payloads])
    return parity_rows(arr, P)[:m]


def stripe_records(stripe_no, payloads, parity, width, height):
    """Render one FEC stripe as QUIC records: (k+m) groups, R frames each.
    CPU-heavy at 1080p/4K — call via asyncio.to_thread.

    Groups are rendered IN PARALLEL (render + zlib both release the GIL),
    so a 4K stripe takes ~one group's time instead of ten serial ones."""
    seq0 = stripe_no * (len(payloads) + len(parity))
    allp = payloads + [p.tobytes() for p in parity]
    n = len(allp)
    recs = [None] * n

    def _one(i):
        f = render_group(allp[i], seq0 + i, M, width, height)
        r = record(T_GROUP, f)
        return [r] * RQ

    if n <= 2:
        for i in range(n):
            recs[i] = _one(i)
    else:
        # 4 workers: zlib/numpy hold the GIL most of the time, so more
        # threads only add convoy contention and starve the event loop
        # (a 16-thread render once stalled QUIC reads for ~30 s at 4K)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(n, 4)) as ex:
            for i, chunk in enumerate(ex.map(_one, range(n))):
                recs[i] = chunk
    out = []
    for chunk in recs:
        out.extend(chunk)
    return out


# ---------------------------------------------------------------------------
# SEND side: local TCP -> mux -> color blocks + FEC -> QUIC stream
# ---------------------------------------------------------------------------

class SendSide:
    def __init__(self, args):
        self.args = args
        self.B = group_capacity(args.width, args.height, M)
        self.P = cauchy_matrix(args.k, args.m)
        self.mux = Muxer(self.B)
        self.srv = None
        self.next_cid = [1]
        self.clients = set()          # live local client conn ids
        self.writer = None            # QUIC stream writer (None: no link)
        self.write_q = None           # rendered records queued for the writer
        self.render_q = None          # (stripe_no, payloads) queued for render
        self.writer_task = None       # pacing writer task (one per link)
        self.render_task = None       # in-order render task (one per link)
        self.stripe = []
        self.stripe_no = 0
        self.stripe_grew_at = 0.0      # last time a group entered the stripe
        self.idle_period = (args.k + args.m) * RQ / args.fps
        self.stop = False
        self.meta_records = None

    async def init(self):
        """Precompute the (expensive at 4K) meta records off-loop."""
        loop = asyncio.get_running_loop()
        self.stripe_open_at = loop.time()
        self.last_idle_emit = loop.time()
        self.last_log = loop.time()
        w, h = self.args.width, self.args.height

        def _meta():
            meta = {'fn': 'bitcoder-quic', 'w': w, 'h': h, 'M': M, 'R': RQ,
                    'k': self.args.k, 'm': self.args.m, 'B': self.B,
                    'fps': self.args.fps}
            return [record(T_META, render_meta(meta, w, h))
                    for _ in range(R_META)]

        self.meta_records = await asyncio.to_thread(_meta)
        # measure one idle-stripe render so the keepalive period can be
        # stretched on big canvases (idle stripes must be rendered with
        # their LIVE seq, so they cannot be pre-baked)
        t0 = loop.time()
        await asyncio.to_thread(
            stripe_records, 0, [b'\x00' * self.B] * self.args.k,
            make_parity([b'\x00' * self.B] * self.args.k, self.P,
                        self.args.m),
            w, h)
        self._idle_render_s = loop.time() - t0
        self.idle_period = max(self.idle_period,
                               1.3 * self._idle_render_s + 0.1)
        log.info('[send] idle stripe renders in ~%.0f ms; idle period %.2f s',
                 1000 * self._idle_render_s, self.idle_period)

    async def start(self):
        a = self.args
        self.srv = await asyncio.start_server(
            self._on_local_client, a.tcp_srv_host, a.tcp_srv_port)
        log.info('[send] local TCP server on %s:%d (clients connect here; '
                 'data leaves over QUIC)', a.tcp_srv_host, a.tcp_srv_port)

    async def _on_local_client(self, reader, writer):
        cid = self.next_cid[0]
        if cid > 254:
            writer.close()
            return
        self.next_cid[0] += 1
        self.clients.add(cid)
        log.info('[send] local client %d connected', cid)
        got = 0
        try:
            while not self.stop:
                data = await reader.read(65536)
                if not data:
                    break
                got += len(data)
                if got <= 200000 or got % 1000000 < 65536:
                    log.info('[send] client %d read %d B (total %d)',
                             cid, len(data), got)
                # socket-read backpressure: a client whose data is held
                # back simply stops being drained (this also parks it
                # during a link outage — its data stays in the mux and is
                # sent on the re-established stream)
                while self.mux.backpressure(cid) and not self.stop:
                    await asyncio.sleep(0.005)
                self.mux.feed(cid, data)
        except (ConnectionError, OSError):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass
            self.clients.discard(cid)
            self.mux.feed(cid, b'', fin=True)
            log.info('[send] client %d closed', cid)

    async def _emit(self, payloads):
        """Hand the stripe to the render pipeline. Non-blocking: the
        worker renders in a thread while the writer keeps writing the
        previous stripe, so render CPU no longer adds to wire time."""
        await self.render_q.put((self.stripe_no, payloads))
        self.stripe_no += 1
        self.stripe_open_at = asyncio.get_running_loop().time()

    async def _render_task(self):
        """Render stripes IN ORDER in worker threads (a single worker is
        what preserves order) and hand the ready records to the writer.
        Pipelined with the wire: stripe N+1 renders while stripe N is
        still being written."""
        try:
            while True:
                no, payloads = await self.render_q.get()
                recs = await asyncio.to_thread(
                    stripe_records, no, payloads,
                    make_parity(payloads, self.P, self.args.m),
                    self.args.width, self.args.height)
                if self.writer is not None and not self.stop:
                    await self.write_q.put(recs)
                # else: link died while rendering — the peer's fresh
                # stream starts a clean session, old records are
                # worthless; drop them
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception('[send] render task crashed')

    async def _writer_task(self):
        """Write queued records to the live QUIC stream, paced at ~fps
        (one video frame per slot). Runs as its OWN task so the event loop
        stays free for QUIC timers — a render-blocking main loop would miss
        the peer's idle timeout and get disconnected (or write into a
        connection that is already dead)."""
        slot = 1.0 / self.args.fps
        loop = asyncio.get_running_loop()
        drain_s = 0.0
        n_stripes = 0
        try:
            while True:
                recs = await self.write_q.get()
                w = self.writer
                if w is None or self.stop:
                    # link down while the stripe was queued: the peer's
                    # fresh stream starts a clean session, old records are
                    # worthless — drop and wait for new ones
                    continue
                t0 = loop.time()
                next_t = loop.time()
                for rec in recs:
                    w.write(rec)
                    next_t += slot
                    now = loop.time()
                    if now < next_t:
                        await asyncio.sleep(next_t - now)
                    else:
                        next_t = now
                t1 = loop.time()
                await w.drain()
                t2 = loop.time()
                drain_s += t2 - t1
                n_stripes += 1
                if n_stripes <= 20 or n_stripes % 50 == 0:
                    log.info('[send] stripe %d written: loop %.0f ms, '
                             'drain-avg %.0f ms',
                             self.stripe_no, 1000 * (t1 - t0),
                             1000 * drain_s / n_stripes)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception('[send] writer task crashed')

    async def _flush_partial(self):
        # Drain ALL complete groups into the stripe first, then the
        # sub-group tail — flush_partial() alone raises when the buffer
        # holds more than one group.
        while self.mux.pending() >= self.B and len(self.stripe) < self.args.k:
            self.stripe.append(self.mux.next_group())
        if self.mux.pending():
            self.stripe.append(self.mux.flush_partial())
        if self.stripe:
            while len(self.stripe) < self.args.k:
                self.stripe.append(b'\x00' * self.B)
            await self._emit(self.stripe)
            self.stripe = []
            self.stripe_open_at = asyncio.get_running_loop().time()

    async def _main_loop(self):
        a = self.args
        while not self.stop:
            now = asyncio.get_running_loop().time()
            if self.writer is None:
                # link down: clients are stalled by mux backpressure,
                # their data stays buffered; nothing to emit
                await asyncio.sleep(0.01)
            else:
                # admit held-back data as group space frees up
                for cid in list(self.mux.holdback.keys()):
                    n = self.mux.admitted(cid)
                    if n:
                        self.mux.admit(cid, n)
                # fill the stripe
                while (self.mux.pending() >= self.B
                        and len(self.stripe) < a.k):
                    self.stripe.append(self.mux.next_group())
                    self.stripe_grew_at = now
                if len(self.stripe) == a.k:
                    await self._emit(self.stripe)
                    self.stripe = []
                if self.clients:
                    # client present: flush a stripe that has stopped
                    # GROWING (no new group entered in FLUSH_IDLE_AFTER).
                    # A group at the channel's max rate arrives every
                    # B/rate ≈ 125 ms (1080p), so a 200 ms silence means
                    # the client genuinely paused — the tail leaves with
                    # zero padding and the demux concatenates payloads
                    # in order and skips idle packets. A hard 1.5 s cap
                    # also catches a vanished client mid-stripe.
                    if self.stripe:
                        if (now - self.stripe_grew_at > FLUSH_IDLE_AFTER
                                or now - self.stripe_open_at
                                > FLUSH_HARD_AFTER):
                            await self._flush_partial()
                    # ALWAYS yield: the fill loops are sync work and a
                    # hot stripe must not starve the event loop
                    await asyncio.sleep(0.001)
                else:
                    # no local clients: flush whatever is in flight, then
                    # keep the QUIC stream alive with idle zero stripes
                    # (rendered live in a worker thread so seq stays
                    # monotonic)
                    if self.stripe or self.mux.pending():
                        await self._flush_partial()
                    if now - self.last_idle_emit >= self.idle_period:
                        self.last_idle_emit = now
                        await self._emit([b'\x00' * self.B] * a.k)
                    await asyncio.sleep(0.005)
            if now - self.last_log > 10:
                self.last_log = now
                log.info('[send] stripes=%d clients=%d link=%s mux=%d '
                         'hold=%d q=%s',
                         self.stripe_no, len(self.clients),
                         'up' if self.writer is not None else 'down',
                         self.mux.pending(),
                         sum(len(v) for v in self.mux.holdback.values()),
                         self.write_q.qsize() if self.write_q is not None
                         else '-')

    # -- QUIC client: dial the peer; redial forever -------------------------
    async def run(self):
        a = self.args
        while not self.stop:
            cfg = QuicConfiguration(is_client=True, alpn_protocols=ALPN,
                                    server_name=CERT_CN, verify_mode=0,
                                    max_data=1 << 26,
                                    max_stream_data=1 << 26,
                                    idle_timeout=10.0)
            try:
                async with connect(a.peer_host, a.peer_port,
                                   configuration=cfg) as proto:
                    reader, writer = await proto.create_stream()
                    for mr in self.meta_records:
                        writer.write(mr)
                    await writer.drain()
                    # link (re)established: the main loop resumes emitting
                    # into the fresh stream. Data buffered in the mux during
                    # the outage is sent as-is — the peer's recv binds its
                    # local clients to observed conn ids BY ORDER, so the
                    # absolute conn values don't need to line up across a
                    # failover.
                    # create the pipeline queues FIRST, then expose the
                    # writer (the main loop only emits when writer is set,
                    # so it can never touch a None queue)
                    self.write_q = asyncio.Queue(maxsize=2)
                    self.render_q = asyncio.Queue(maxsize=2)
                    self.render_task = asyncio.ensure_future(
                        self._render_task())
                    self.writer_task = asyncio.ensure_future(
                        self._writer_task())
                    self.writer = writer
                    log.info('[send] QUIC up to %s:%d (ALPN %s, TLS 1.3); '
                             'header sent', a.peer_host, a.peer_port,
                             ALPN[0])
                    # wait until the peer dies (idle timeout) or we stop
                    await proto.wait_closed()
                    log.info('[send] QUIC link to peer dropped')
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if self.stop:
                    break
                log.warning('[send] QUIC dial to %s:%d failed (%s); '
                            're-dialing in %.0f s', a.peer_host, a.peer_port,
                            e.__class__.__name__, REDIAL_WAIT)
            # link lost: the main loop idles, clients are held by mux
            # backpressure; stop the pipeline, re-dial
            self.writer = None
            for t in (self.writer_task, self.render_task):
                if t is not None:
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass
            self.writer_task = None
            self.render_task = None
            await asyncio.sleep(REDIAL_WAIT)

    async def shutdown(self):
        self.stop = True
        for t in (self.writer_task, self.render_task):
            if t is not None:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        if self.srv is not None:
            self.srv.close()
            await self.srv.wait_closed()
        log.info('[send] stopped after %d stripes', self.stripe_no)


# ---------------------------------------------------------------------------
# RECEIVE side: QUIC stream -> color-block decode + FEC -> demux -> local TCP
# ---------------------------------------------------------------------------

class _Session:
    """Receive state for ONE QUIC stream. The peer's conn-id space restarts
    on every fresh stream (it redials with a clean mux after a link failure),
    so the local-client bindings, pending buffers and finished flags must be
    per-stream, not per-node — otherwise a baseline conn=1 'finished' blocks
    the post-failover stream's conn=1."""
    __slots__ = ('socks', 'pending', 'finished', 'fin_flags',
                 'waiters', 'seen')

    def __init__(self):
        self.socks = {}          # conn -> local client writer (client ACTIVE)
        self.pending = {}        # conn -> bytes buffered before an active client
        self.finished = set()    # conns that sent their FIN this stream
        self.fin_flags = set()   # conns with a pending FIN, no active client
        self.waiters = {}        # conn -> _Waiter bound but not yet ACTIVE
        self.seen = set()        # conns observed on this stream


class _Waiter:
    """A local TCP client that connected but is not yet bound to a peer conn.
    Binding is by order: the k-th waiting client is bound to the k-th distinct
    conn the stream observes — this absorbs any cid offset the peer carries.
    Once bound, the waiter parks in waiters[conn] (data is buffered) until it
    wakes, drains its buffer, and takes the conn in socks — only then is it
    'active'. A FIN that arrives while it is still parked is remembered in
    fin_flags instead of closing a writer the waiter hasn't written to yet."""
    __slots__ = ('reader', 'writer', 'evt', 'conn', 'sess')

    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer
        self.evt = asyncio.Event()
        self.conn = -1
        self.sess = None


class RecvSide:
    def __init__(self, args):
        self.args = args
        self.B = group_capacity(args.width, args.height, M)
        os.makedirs(args.out, exist_ok=True)
        self.dumper = open(os.path.join(args.out, 'quic_dump.bin'), 'ab')
        self.waiting = deque()   # local clients awaiting a stream binding
        self.sessions = set()    # live _Session objects
        self.groups = 0
        self.repaired = 0
        self.crc_bad = 0
        self.resyncs = 0
        self.srv = None
        self.dlv = {}            # conn -> delivered data bytes (diagnostics)

    async def start(self):
        a = self.args
        cert, key = make_selfsigned(a.out)
        cfg = QuicConfiguration(is_client=False, alpn_protocols=ALPN,
                                certificate=cert, private_key=key,
                                max_data=1 << 26,
                                max_stream_data=1 << 26,
                                idle_timeout=10.0)

        def stream_cb(reader, writer):
            async def _guarded():
                try:
                    await self._on_stream(reader, writer)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception('[recv] stream task crashed')
            asyncio.ensure_future(_guarded())

        self.srv = await serve(a.quic_host, a.quic_port, configuration=cfg,
                               stream_handler=stream_cb)
        log.info('[recv] QUIC server on %s:%d (ALPN %s, TLS 1.3)',
                 a.quic_host, a.quic_port, ALPN[0])

    # -- local TCP serving (peer data goes to local clients) ----------------
    async def _accept_local(self):
        a = self.args
        srv = await asyncio.start_server(self._on_local_client,
                                         a.tcp_cli_host, a.tcp_cli_port)
        log.info('[recv] local TCP client-port on %s:%d (peer data is '
                 'served here)', a.tcp_cli_host, a.tcp_cli_port)
        await srv.wait_closed()

    async def _on_local_client(self, reader, writer):
        w = _Waiter(reader, writer)
        self.waiting.append(w)
        try:
            # wait for the stream to observe a peer conn and bind us to it
            await asyncio.wait_for(w.evt.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            log.warning('[recv] local client waited 60 s for peer conn; '
                        'closing')
            try:
                writer.close()
            except Exception:
                pass
            return
        except asyncio.CancelledError:
            try:
                writer.close()
            except Exception:
                pass
            raise
        sess, conn = w.sess, w.conn
        # become ACTIVE: drain what was buffered while parked, then take the
        # conn in socks. No await between the drain and the takeover, so the
        # stream task cannot interleave a FIN between the two.
        buf = sess.pending.pop(conn, b'')
        had_fin = conn in sess.fin_flags
        if had_fin:
            sess.fin_flags.discard(conn)
        sess.socks[conn] = writer
        del sess.waiters[conn]
        log.info('[recv] local client bound to conn %d (%d B buffered)',
                 conn, len(buf))
        try:
            if buf:
                writer.write(buf)
                await writer.drain()
            if had_fin:
                return
            while True:
                # the local client only reads; detect its EOF to clean up
                data = await reader.read(65536)
                if not data:
                    break
        except (ConnectionError, OSError):
            pass
        finally:
            # unbind: later data for this conn goes back to buffering
            if sess.socks.get(conn) is writer:
                del sess.socks[conn]
            try:
                writer.close()
            except Exception:
                pass

    async def _deliver(self, sess, conn, data, fin):
        if conn > 254:
            return
        if data and self.dumper is not None:
            self.dumper.write(data)
            self.dumper.flush()
            self.dlv[conn] = self.dlv.get(conn, 0) + len(data)
        writer = sess.socks.get(conn)
        if writer is not None:
            if data:
                writer.write(data)
                await writer.drain()
            if fin:
                sess.finished.add(conn)
                sess.socks.pop(conn, None)
                try:
                    writer.close()
                except Exception:
                    pass
                log.info('[recv] connection %d finished after %d B',
                         conn, self.dlv.get(conn, 0))
            return
        if conn in sess.finished:
            return
        # no active client: buffer the bytes. A bound-but-parked waiter (or a
        # fresh one from the waiting queue) will pick the buffer up; a FIN is
        # remembered so the client sees its data and THEN the EOF.
        p = sess.pending.get(conn, b'')
        if len(p) + len(data) > MAX_BUFFER:
            log.warning('[recv] connection %d buffer overflow', conn)
            return
        sess.pending[conn] = p + data
        if fin:
            sess.fin_flags.add(conn)
            log.info('[recv] connection %d FIN (buffered, %d B total)',
                     conn, self.dlv.get(conn, 0))
            return
        if conn not in sess.waiters and self.waiting:
            # bind the next waiting local client to this conn
            w = self.waiting.popleft()
            sess.waiters[conn] = w
            w.sess = sess
            w.conn = conn
            w.evt.set()

    # -- QUIC stream reading --------------------------------------------------
    async def _on_stream(self, reader, writer):
        a = self.args
        sess = _Session()
        self.sessions.add(sess)
        k, m, B, M_ = a.k, a.m, self.B, M
        P = None
        meta = None
        meta_frames = []
        dm = Demuxer()
        seq_trk = SeqWrapTracker()
        stripe_no = -1
        stripe_recv = []
        flushed = set()
        buf = b''
        read_wait_s = 0.0
        pending = []
        BATCH = 10

        async def read_record():
            nonlocal buf, read_wait_s
            while True:
                if len(buf) >= 4:
                    n = int.from_bytes(buf[:4], 'big')
                    if 0 < n <= (1 << 24) and len(buf) >= 4 + n:
                        rec = buf[4:4 + n]
                        buf = buf[4 + n:]
                        return rec
                t0 = asyncio.get_running_loop().time()
                d = await reader.read(65536)
                read_wait_s += asyncio.get_running_loop().time() - t0
                if not d:
                    return None
                buf += d

        async def flush_stripe(no):
            nonlocal stripe_recv
            if no in flushed:
                return
            flushed.add(no)
            recv = stripe_recv[:k + m]
            n_lost = sum(1 for x in recv if x is None)
            if n_lost > m:
                log.warning('[recv] stripe %d: %d of %d groups lost '
                            '(> %d correctable); dropping stripe', no,
                            n_lost, k + m, m)
                return
            if n_lost:
                rows = stripe_data(recv, k, P)
                self.repaired += n_lost
                log.info('[recv] stripe %d: repaired %d lost group(s)',
                         no, n_lost)
                rows = [r.tobytes() for r in rows]
            else:
                rows = [x for x in recv[:k] if x is not None]
            for p in rows:
                for conn, data, fin in dm.feed(p):
                    if conn not in sess.seen:
                        sess.seen.add(conn)
                    await self._deliver(sess, conn, data, fin)

        def _decode_sync(rec):
            # sync worker: runs in a thread pool. QUIC records arrive
            # byte-exact, so the lossless single-sample decode is an exact
            # inverse of render_group (91x faster than the R-copy path).
            try:
                tag, frame = decode_record_body(rec)
            except Exception:
                # a corrupt record is an erasure: FEC repairs it
                return (T_GROUP, None, 'nocrc')
            if tag == T_META or tag == T_EOF:
                return (tag, frame, None)
            try:
                h8, payload = decode_group_fast(frame, M_, a.width, a.height)
            except Exception:
                h8, payload = None, None
            if h8 is None:
                return (tag, None, 'nocrc')
            seq, ok = verify_group(h8, payload)
            if not ok:
                return (tag, None, 'badcrc')
            return (tag, (seq, payload), None)

        async def process(recs):
            """Decode a batch of records in parallel and feed the results
            (in stream order) into the stripe/demux state machine."""
            nonlocal stripe_no, stripe_recv, read_wait_s
            t0 = asyncio.get_running_loop().time()
            decoded = await asyncio.gather(
                *(asyncio.to_thread(_decode_sync, r) for r in recs))
            t1 = asyncio.get_running_loop().time()
            for tag, val, err in decoded:
                if err == 'nocrc':
                    self.crc_bad += 1
                    continue
                if err == 'badcrc':
                    self.crc_bad += 1
                    self.resyncs += 1
                    log.warning('[recv] bad-CRC frame; skipping')
                    continue
                if tag == T_META:
                    continue  # meta handled at read time
                if tag == T_EOF:
                    frame = val
                    h8, payload = await asyncio.to_thread(
                        decode_group_fast, frame, M_, a.width, a.height)
                    seq, ok = (verify_group(h8, payload)
                               if h8 is not None else (None, False))
                    if ok and seq == 0xFFFF and payload[:4] == EOF_PAYLOAD:
                        log.info('[recv] tunnel-idle EOF after %d groups',
                                 self.groups)
                    continue
                seq, payload = val
                g = seq_trk.true_index(seq)
                s = stripe_of(g, k, m)
                if s != stripe_no:
                    if stripe_no >= 0:
                        await flush_stripe(stripe_no)
                    stripe_no = s
                    stripe_recv = [None] * (k + m)
                idx = g - s * (k + m)
                if 0 <= idx < k + m and stripe_recv[idx] is None:
                    stripe_recv[idx] = payload
                    if all(x is not None for x in stripe_recv):
                        await flush_stripe(stripe_no)
                self.groups += 1
                if self.groups <= 20 or self.groups % 200 == 0:
                    t2 = asyncio.get_running_loop().time()
                    log.info('[recv] group %d (seq %d, stripe %d) | '
                             'decode %.0f ms, feed %.0f ms, read-wait %.2f s, '
                             'batch %d',
                             self.groups, seq, s, 1000 * (t1 - t0),
                             1000 * (t2 - t1), read_wait_s, len(recs))
                    read_wait_s = 0.0

        try:
            while True:
                rec = await read_record()
                if rec is None:
                    break
                # meta arrives before any group and is small: decode it
                # inline (three 6 MB frames only at startup)
                if rec and rec[0] == T_META:
                    meta_frames.append(decode_record_body(rec)[1])
                    if meta is None and len(meta_frames) == R_META:
                        meta = await asyncio.to_thread(
                            decode_meta, meta_frames, a.width, a.height)
                        if not meta:
                            log.error('[recv] metadata decode failed')
                            return
                        k = int(meta.get('k', k))
                        m = int(meta.get('m', m))
                        B = int(meta.get('B', B))
                        M_ = int(meta.get('M', M))
                        P = cauchy_matrix(k, m)
                        log.info('[recv] peer meta: k=%d m=%d B=%d', k, m, B)
                        if (int(meta.get('w', -1)) != a.width
                                or int(meta.get('h', -1)) != a.height):
                            log.error('[recv] geometry mismatch with '
                                      'peer (%sx%s)', meta.get('w'),
                                      meta.get('h'))
                            return
                    continue
                if meta is None:
                    continue
                pending.append(rec)
                if len(pending) >= BATCH:
                    await process(pending)
                    pending = []
            if pending:
                await process(pending)
                pending = []
        finally:
            self.sessions.discard(sess)
            log.info('[recv] stream closed: %d groups, %d repaired, '
                     '%d bad-CRC, %d resyncs', self.groups, self.repaired,
                     self.crc_bad, self.resyncs)
            try:
                self.dumper.flush()
            except Exception:
                pass

    async def shutdown(self):
        if self.srv is not None:
            self.srv.close()
        if self.dumper is not None:
            self.dumper.close()
        log.info('[recv] stopped (delivered %s)', self.dlv)


# ---------------------------------------------------------------------------

async def _watch(name, coro):
    """Run a coroutine, logging any exception it raises (so a dead task
    is visible in the node log instead of vanishing into gather)."""
    try:
        return await coro
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception('[%s] task crashed', name)
        raise


async def _loop_heartbeat():
    """Measure event-loop starvation: a task that sleeps 50 ms and checks
    how much REAL time passed. Max lag > 100 ms means the loop (usually
    the GIL) is blocked and QUIC timers/ACKs/reads starve."""
    loop = asyncio.get_running_loop()
    worst = 0.0
    n = 0
    try:
        while True:
            t0 = loop.time()
            await asyncio.sleep(0.05)
            lag = loop.time() - t0 - 0.05
            if lag > worst:
                worst = lag
            n += 1
            if n % 200 == 0:  # every 10 s
                log.info('[hb] loop max-lag %.1f ms over %d ticks',
                         1000 * worst, n)
                worst = 0.0
    except asyncio.CancelledError:
        raise


async def run(args):
    B = group_capacity(args.width, args.height, M)
    log.info('node: %dx%d stream, %d B/frame, ~%.0f KB/s per direction',
             args.width, args.height, B,
             throughput_bps(args.width, args.height, args.fps,
                            args.k, args.m) / 1024)
    os.makedirs(args.out, exist_ok=True)
    send = SendSide(args)
    await send.init()          # precompute meta/EOF records (off-loop)
    recv = RecvSide(args)
    await recv.start()
    await send.start()

    loop = asyncio.get_running_loop()
    stop_evt = asyncio.Event()
    try:
        loop.add_signal_handler(signal.SIGINT, stop_evt.set)
        loop.add_signal_handler(signal.SIGTERM, stop_evt.set)
    except NotImplementedError:
        pass

    tasks = [
        asyncio.ensure_future(_watch('send-quic', send.run())),
        asyncio.ensure_future(_watch('send-main', send._main_loop())),
        asyncio.ensure_future(_watch('recv-local', recv._accept_local())),
        asyncio.ensure_future(_watch('loop-hb', _loop_heartbeat())),
    ]
    try:
        await stop_evt.wait()
        log.info('interrupted; shutting down node')
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await send.shutdown()
        await recv.shutdown()
        log.info('node stopped')


def main():
    # faster GIL handoff: numpy/zlib workers hold the GIL, and a 5 ms
    # default switch interval lets 10-20 concurrent workers convoy the
    # event loop (observed: 29 s QUIC read stalls at 4K)
    import sys
    sys.setswitchinterval(0.001)
    ap = argparse.ArgumentParser(
        description='bitcoder QUIC tunnel node: simultaneous send + receive')
    ap.add_argument('--quic-host', default='127.0.0.1')
    ap.add_argument('--quic-port', type=int, default=9500,
                    help='QUIC port to LISTEN on (peer data arrives here)')
    ap.add_argument('--peer-host', default='127.0.0.1')
    ap.add_argument('--peer-port', type=int, default=9501,
                    help='peer QUIC port to DIAL (local data leaves here)')
    ap.add_argument('--tcp-srv-host', default='127.0.0.1')
    ap.add_argument('--tcp-srv-port', type=int, default=9510,
                    help='local port: clients connect HERE, data goes to peer')
    ap.add_argument('--tcp-cli-host', default='127.0.0.1')
    ap.add_argument('--tcp-cli-port', type=int, default=9511,
                    help='local port: peer data is served HERE')
    ap.add_argument('--width', type=int, default=1280,
                    help='frame width: 640/1280/1920/3840 (any even)')
    ap.add_argument('--height', type=int, default=720,
                    help='frame height: 360/720/1080/2160 (any even)')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--k', type=int, default=K)
    ap.add_argument('--m', type=int, default=M_PAR)
    ap.add_argument('--out', default='quic_out')
    args = ap.parse_args()
    if args.width % 2 or args.height % 2:
        ap.error('width/height must be even (yuv420p)')
    if not (1 <= args.k and args.k + args.m <= 255):
        ap.error('invalid stripe size k+m')
    try:
        B = group_capacity(args.width, args.height, M)
        log.info('frame geometry: %dx%d -> %d B payload/frame, '
                 '~%.0f KB/s per direction at full stripes', args.width,
                 args.height, B,
                 throughput_bps(args.width, args.height, args.fps,
                                args.k, args.m) / 1024)
    except ValueError as e:
        ap.error(str(e))
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
