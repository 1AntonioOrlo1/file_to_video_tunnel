#!/usr/bin/env python3
"""Two-node QUIC bidirectional tunnel test.

node A: --quic-port 9600 (listen) --peer-port 9601 (dial)  -> A sends to B
node B: --quic-port 9601 (listen) --peer-port 9600 (dial)  -> B sends to A

Channel A->B: client -> A's tcp-srv -> QUIC -> B's tcp-cli -> far client.
Channel B->A: the same in reverse. Both directions at once.

Runs at the requested resolution (default 640x360); the same script is
re-invoked for 720p, 1080p and 4K by test_quic_res.py.
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
W = int(os.environ.get('TUNNEL_W', 640))
H = int(os.environ.get('TUNNEL_H', 360))
OUT = os.path.join(HERE, f'quic_e2e_{W}x{H}')
os.makedirs(OUT, exist_ok=True)

A_Q, B_Q = 9600, 9601
A_TCP_SRV, A_TCP_CLI = 9710, 9711
B_TCP_SRV, B_TCP_CLI = 9720, 9721

PAYLOAD_A = os.urandom(30000)   # A -> B
PAYLOAD_B = os.urandom(45000)   # B -> A


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def start_node(name, q_port, peer_port, tcp_srv, tcp_cli):
    logp = os.path.join(OUT, f'{name}.log')
    open(logp, 'w').close()
    # fresh dump each run: the node opens it in append mode
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


def wait_log(path, needles, timeout=90):
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


def send_blob(port, blob, tag):
    s = socket.create_connection(('127.0.0.1', port), timeout=15)
    s.sendall(blob)
    s.shutdown(socket.SHUT_WR)
    s.close()
    print(f'  sent {tag}: {len(blob)} B')


def recv_blob(port, want, timeout=120):
    s = socket.create_connection(('127.0.0.1', port), timeout=15)
    s.settimeout(timeout)
    data = b''
    t0 = time.time()
    try:
        while len(data) < want and time.time() - t0 < timeout:
            d = s.recv(65536)
            if not d:
                break
            data += d
    except (socket.timeout, OSError):
        pass
    s.close()
    return data


def check(got, blob, tag, dt):
    ok = got == blob
    print(f'[{"OK" if ok else "FAIL"}] {tag}: got {len(got)}/{len(blob)} B '
          f'(sha {"match" if ok else "MISMATCH: " + sha(got[:16])}) '
          f'{dt:.1f}s')
    return ok


def main():
    print(f'=== QUIC bidirectional test at {W}x{H} ===')
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
        # far-side receivers connect BEFORE the send (like real clients)
        far_ab = socket.create_connection(('127.0.0.1', B_TCP_CLI), timeout=15)
        far_ba = socket.create_connection(('127.0.0.1', A_TCP_CLI), timeout=15)
        far_ab.settimeout(120)
        far_ba.settimeout(120)
        print('[OK] far-side clients connected on both nodes')

        def drain(sock, want, timeout=120):
            data = b''
            t0 = time.time()
            try:
                while len(data) < want and time.time() - t0 < timeout:
                    d = sock.recv(65536)
                    if not d:
                        break
                    data += d
            except (socket.timeout, OSError):
                pass
            return data

        def sa():
            send_blob(B_TCP_SRV, PAYLOAD_B, 'B->A payload')

        def sb():
            send_blob(A_TCP_SRV, PAYLOAD_A, 'A->B payload')

        t0 = time.time()
        ta, tb = threading.Thread(target=sa), threading.Thread(target=sb)
        ta.start(); tb.start()
        ta.join(); tb.join()

        got_ab = drain(far_ab, len(PAYLOAD_A))   # A->B, read at B_CLI
        got_ba = drain(far_ba, len(PAYLOAD_B))   # B->A, read at A_CLI
        dt = time.time() - t0
        ok &= check(got_ab, PAYLOAD_A, 'channel A->B', dt)
        ok &= check(got_ba, PAYLOAD_B, 'channel B->A', dt)
        far_ab.close(); far_ba.close()
        print(f'both directions in {dt:.1f}s total')

        # dumps should hold the payloads too (truncated at node start, so
        # each run starts a fresh dump file)
        da = open(os.path.join(OUT, 'a', 'quic_dump.bin'), 'rb').read()
        db = open(os.path.join(OUT, 'b', 'quic_dump.bin'), 'rb').read()
        da_ok = len(da) >= len(PAYLOAD_B) and da[:len(PAYLOAD_B)] == PAYLOAD_B
        db_ok = len(db) >= len(PAYLOAD_A) and db[:len(PAYLOAD_A)] == PAYLOAD_A
        print(f'[{"OK" if da_ok else "FAIL"}] node A dump holds B payload '
              f'({len(da)} B)')
        print(f'[{"OK" if db_ok else "FAIL"}] node B dump holds A payload '
              f'({len(db)} B)')
        ok &= da_ok and db_ok
        print('RESULT:', 'PASS' if ok else 'FAIL')
        return 0 if ok else 1
    finally:
        for p in (a, b):
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGINT)
                    p.wait(timeout=15)
                except Exception:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    except Exception:
                        pass
        time.sleep(1)


if __name__ == '__main__':
    sys.exit(main())
