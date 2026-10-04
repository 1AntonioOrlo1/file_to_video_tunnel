#!/usr/bin/env python3
"""Sustained-stream QUIC tunnel test.

test_quic.py sends one-shot bursts; this one holds a CONTINUOUS transfer in
both directions at once (a real video link under load) and verifies:
  * the ENTIRE received stream is byte-exact (sha + length),
  * the measured throughput (first→last byte) reaches a target fraction of
    the geometry's theoretical maximum ((k*B) per (k+m)*R frames at fps).

The client is BURSTY (pushes as fast as the channel takes; mux backpressure
rate-limits it to the channel's real throughput) — this is the case that
exposed the partial-stripe throughput collapse: a slow client makes every
stripe partial, and a partial stripe costs the same 0.67 s of wire as a
full one, halving the channel.
"""
import hashlib
import os
import signal
import socket
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
W = int(os.environ.get('TUNNEL_W', 1920))
H = int(os.environ.get('TUNNEL_H', 1080))
DURATION = float(os.environ.get('TUNNEL_STREAM_S', 20.0))
MIN_RATE_FRACTION = 0.5    # measured rate must reach 50% of theoretical
OUT = os.path.join(HERE, f'quic_stream_{W}x{H}')
os.makedirs(OUT, exist_ok=True)

A_Q, B_Q = 9620, 9621
A_TCP_SRV, A_TCP_CLI = 9630, 9631
B_TCP_SRV, B_TCP_CLI = 9632, 9633


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def start_node(name, q_port, peer_port, tcp_srv, tcp_cli):
    logp = os.path.join(OUT, f'{name}.log')
    open(logp, 'w').close()
    dump = os.path.join(OUT, name, 'quic_dump.bin')
    if os.path.exists(dump):
        open(dump, 'wb').close()
    log = open(logp, 'ab')
    p = subprocess.Popen(
        [sys.executable, os.path.join(HERE, 'quic_node.py'),
         '--quic-host', '127.0.0.1', '--quic-port', str(q_port),
         '--peer-host', '127.0.0.1', '--peer-port', str(peer_port),
         '--tcp-srv-port', str(tcp_srv), '--tcp-cli-port', str(tcp_cli),
         '--width', str(W), '--height', str(H),
         '--out', os.path.join(OUT, name)],
        stdout=log, stderr=subprocess.STDOUT, cwd=HERE,
        preexec_fn=os.setsid)
    return p, logp


def wait_log(path, needles, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            txt = open(path, errors='replace').read()
        except OSError:
            txt = ''
        if all(n in txt for n in needles):
            return True
        time.sleep(0.3)
    return False


def run_thread(name, fn, errlog):
    """Run fn in a thread; any exception goes to errlog (threads swallow
    their exceptions otherwise, which made hangs undiagnosable)."""
    def wrap():
        try:
            fn()
        except Exception:
            import traceback
            with open(errlog, 'a') as f:
                f.write(f'--- thread {name} crashed ---\n')
                traceback.print_exc(file=f)
    return threading.Thread(target=wrap)


def main():
    from tunnel_core import group_capacity, K, M_PAR
    from quic_node import RQ
    B = group_capacity(W, H, 8)
    stripe = K * B
    stripe_t = (K + M_PAR) * RQ / 30
    rate = stripe / stripe_t               # theoretical max, B/s
    payload = int(rate * DURATION * 0.8)   # ~80% of theoretical volume
    print(f'=== QUIC sustained stream at {W}x{H} ===')
    print(f'  theoretical {rate/1024:.1f} KB/s, payload {payload/1024:.0f} KB, '
          f'bursty client, channel-rate measured over {DURATION}s window')
    blob_a = os.urandom(payload)   # A -> B
    blob_b = os.urandom(payload)   # B -> A
    a, la = start_node('a', A_Q, B_Q, A_TCP_SRV, A_TCP_CLI)
    b, lb = start_node('b', B_Q, A_Q, B_TCP_SRV, B_TCP_CLI)
    ok = True
    try:
        ra = wait_log(la, ['QUIC server on', 'local TCP server'])
        rb = wait_log(lb, ['QUIC server on', 'local TCP server'])
        if not (ra and rb):
            print('[FAIL] nodes did not start')
            for n, p in (('a', la), ('b', lb)):
                print(f'--- {n} ---')
                print(open(p, errors='replace').read()[-2000:])
            return 1
        print('[OK] both nodes up', flush=True)

        # receivers connect BEFORE the send (like real clients)
        far_ab = socket.create_connection(('127.0.0.1', B_TCP_CLI), timeout=15)
        far_ba = socket.create_connection(('127.0.0.1', A_TCP_CLI), timeout=15)
        far_ab.settimeout(600)
        far_ba.settimeout(600)

        def drain(sock, name, results, t0):
            """Drain to EOF; record first/last byte times so the rate is
            measured over the data window, not including node startup."""
            data = b''
            first = last = None
            try:
                while True:
                    d = sock.recv(262144)
                    if not d:
                        break
                    if first is None:
                        first = time.time()
                        print(f'  [{name}] first byte at {first:.2f} '
                              f'(+{first-t0:.2f}s after start)', flush=True)
                    last = time.time()
                    data += d
            except (socket.timeout, OSError):
                pass
            sock.close()
            if first is not None:
                print(f'  [{name}] last byte at {last:.2f} '
                      f'(+{last-t0:.2f}s), {len(data)} B', flush=True)
            win = (last - first) if (first is not None and last is not None) \
                else (time.time() - t0)
            results[name] = (data, win)

        def sender(port, blob):
            """Bursty client: push as fast as the channel will take. The
            mux backpressure (cap 8*B) + socket buffer naturally rate-limit
            this to the channel's real throughput — no artificial pacing."""
            s = socket.create_connection(('127.0.0.1', port), timeout=15)
            t0 = time.time()
            off = 0
            while off < len(blob):
                chunk = blob[off:off + 262144]
                s.sendall(chunk)
                off += len(chunk)
            s.shutdown(socket.SHUT_WR)
            s.close()
            return (time.time() - t0)

        results = {}
        t0 = time.time()
        errlog = os.path.join(OUT, 'test_threads.log')
        th_a = run_thread('drain A->B', lambda: drain(
            far_ab, 'A->B', results, t0), errlog)
        th_b = run_thread('drain B->A', lambda: drain(
            far_ba, 'B->A', results, t0), errlog)
        th_a.start(); th_b.start()

        # A's local client sends blob_a into A_TCP_SRV (goes to B);
        # B's local client sends blob_b into B_TCP_SRV (goes to A)
        send_times = {}

        def sa():
            send_times['A->B'] = sender(A_TCP_SRV, blob_a)

        def sb():
            send_times['B->A'] = sender(B_TCP_SRV, blob_b)

        th_sa = run_thread('send A->B', sa, errlog)
        th_sb = run_thread('send B->A', sb, errlog)
        th_sa.start(); th_sb.start()
        th_sa.join(); th_sb.join()
        th_a.join(); th_b.join()
        if os.path.exists(errlog):
            txt = open(errlog).read()
            if txt:
                print('THREAD ERRORS:\n' + txt)

        for tag, blob in (('A->B', blob_a), ('B->A', blob_b)):
            data, win = results[tag]
            st = send_times[tag]
            mrate = len(data) / win if win else 0
            oklen = len(data) == len(blob)
            oksha = sha(data) == sha(blob) if oklen else False
            okr = mrate >= rate * MIN_RATE_FRACTION
            ok &= oklen and oksha and okr
            print(f'[{"OK" if (oklen and oksha and okr) else "FAIL"}] '
                  f'{tag}: got {len(data)}/{len(blob)} B '
                  f'({"sha match" if oksha else "MISMATCH"}) '
                  f'measured {mrate/1024:.1f} KB/s over {win:.1f}s '
                  f'({100*mrate/rate:.0f}% of theoretical), '
                  f'client drained in {st:.1f}s', flush=True)

        # dumps: the receiver node tees everything it demuxes
        da = open(os.path.join(OUT, 'a', 'quic_dump.bin'), 'rb').read()
        db = open(os.path.join(OUT, 'b', 'quic_dump.bin'), 'rb').read()
        da_ok = len(da) == len(blob_b) and sha(da) == sha(blob_b)
        db_ok = len(db) == len(blob_a) and sha(db) == sha(blob_a)
        print(f'[{"OK" if da_ok else "FAIL"}] node A dump = full B stream '
              f'({len(da)}/{len(blob_b)} B)')
        print(f'[{"OK" if db_ok else "FAIL"}] node B dump = full A stream '
              f'({len(db)}/{len(blob_a)} B)')
        ok &= da_ok and db_ok
        print('RESULT:', 'PASS' if ok else 'FAIL', flush=True)
        return 0 if ok else 1
    finally:
        for p in (a, b):
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGINT)
                    p.wait(timeout=20)
                except Exception:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    except Exception:
                        pass
        time.sleep(1)


if __name__ == '__main__':
    sys.exit(main())
