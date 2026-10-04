#!/usr/bin/env python3
"""QUIC failover test: node B dies hard (SIGKILL) mid-session, then comes
back. Both directions must work again afterwards:
  A->B: A's QUIC client must re-dial B's (fresh) QUIC server
  B->A: B's QUIC client must re-dial A's still-running QUIC server
The QUIC protocol itself handles reconnection; the node just redials.
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
OUT = os.path.join(HERE, f'quic_failover_{W}x{H}')
os.makedirs(OUT, exist_ok=True)

A_Q, B_Q = 9800, 9801
A_TCP_SRV, A_TCP_CLI = 9810, 9811
B_TCP_SRV, B_TCP_CLI = 9820, 9821


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def start_node(name, q_port, peer_port, tcp_srv, tcp_cli):
    logp = os.path.join(OUT, f'{name}.log')
    open(logp, 'w').close()
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
    print(f'  sent {tag}: {len(blob)} B', flush=True)


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


def transfer(tag, payload, near_port, far_port, timeout=120):
    """Open a far receiver, send the payload from the near side, wait."""
    far = socket.create_connection(('127.0.0.1', far_port), timeout=15)
    far.settimeout(timeout)

    def snd():
        send_blob(near_port, payload, tag)

    t0 = time.time()
    th = threading.Thread(target=snd)
    th.start()
    th.join()
    got = drain(far, len(payload), timeout)
    far.close()
    ok = got == payload
    print(f'[{"OK" if ok else "FAIL"}] {tag}: got {len(got)}/{len(payload)} B '
          f'({"sha match" if ok else "MISMATCH"}) {time.time()-t0:.1f}s',
          flush=True)
    return ok


def preflight():
    """Fail fast if stray nodes from a previous run hold the ports —
    otherwise results are untrustworthy (data flows into a zombie)."""
    import subprocess as sp
    out = sp.run(['ps', '-eo', 'args'], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if 'quic_nod' in line and 'python' in line:
            print(f'[FAIL] stray quic_node process still running: '
                  f'{line.strip()}; kill it and retry')
            return False
    for port in (A_TCP_SRV, A_TCP_CLI, B_TCP_SRV, B_TCP_CLI):
        s = socket.socket()
        s.settimeout(0.3)
        if s.connect_ex(('127.0.0.1', port)) == 0:
            s.close()
            print(f'[FAIL] port {port} already in use; kill the holder '
                  f'and retry')
            return False
        s.close()
    return True


def kill_hard(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        pass
    p.wait()


def main():
    print(f'=== QUIC failover test at {W}x{H} ===', flush=True)
    ok = True
    a, la = start_node('a', A_Q, B_Q, A_TCP_SRV, A_TCP_CLI)
    b, lb = start_node('b', B_Q, A_Q, B_TCP_SRV, B_TCP_CLI)
    try:
        ra = wait_log(la, ['QUIC server on', 'local TCP server'])
        rb = wait_log(lb, ['QUIC server on', 'local TCP server'])
        if not (ra and rb):
            print('[FAIL] nodes did not start')
            return 1
        print('[OK] both nodes up', flush=True)

        # phase 0: baseline both directions
        ok &= transfer('baseline A->B', os.urandom(30000),
                       A_TCP_SRV, B_TCP_CLI)
        ok &= transfer('baseline B->A', os.urandom(20000),
                       B_TCP_SRV, A_TCP_CLI)

        # phase 1: kill B hard
        print('  killing B (SIGKILL)', flush=True)
        kill_hard(b)
        b = lb = None
        time.sleep(3)

        # phase 2: restart B; both directions must work again
        b, lb = start_node('b', B_Q, A_Q, B_TCP_SRV, B_TCP_CLI)
        rb = wait_log(lb, ['QUIC server on', 'local TCP server'], timeout=90)
        if not rb:
            print('[FAIL] B did not restart')
            return 1
        print('[OK] B back up', flush=True)

        # give the QUIC re-dial cycle a moment: A only notices B's death at
        # the 10 s idle timeout, then re-dials ~2 s later
        time.sleep(15)
        ok &= transfer('post-failover A->B', os.urandom(40000),
                       A_TCP_SRV, B_TCP_CLI)
        ok &= transfer('post-failover B->A', os.urandom(30000),
                       B_TCP_SRV, A_TCP_CLI)

        print('RESULT:', 'PASS' if ok else 'FAIL', flush=True)
        return 0 if ok else 1
    finally:
        for p in (a, b):
            if p is not None and p.poll() is None:
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
