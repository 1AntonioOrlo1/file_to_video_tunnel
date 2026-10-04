#!/usr/bin/env python3
"""Minimal manual QUIC tunnel driver (debug): 2 nodes, small payload each
way, generous logging to a file so we can see exactly where data stalls."""
import os
import signal
import socket
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'quic_dbg')
os.makedirs(OUT, exist_ok=True)
W, H = 640, 360
A_Q, B_Q = 9650, 9651
A_SRV, A_CLI = 9750, 9751
B_SRV, B_CLI = 9760, 9761


def node(name, q, peer, srv, cli):
    logp = os.path.join(OUT, f'{name}.log')
    open(logp, 'w').close()
    p = subprocess.Popen(
        [sys.executable, os.path.join(HERE, 'quic_node.py'),
         '--quic-host', '127.0.0.1', '--quic-port', str(q),
         '--peer-host', '127.0.0.1', '--peer-port', str(peer),
         '--tcp-srv-port', str(srv), '--tcp-cli-port', str(cli),
         '--width', str(W), '--height', str(H),
         '--out', os.path.join(OUT, name)],
        stdout=open(logp, 'ab'), stderr=subprocess.STDOUT, cwd=HERE,
        preexec_fn=os.setsid)
    return p


def main():
    a = node('a', A_Q, B_Q, A_SRV, A_CLI)
    b = node('b', B_Q, A_Q, B_SRV, B_CLI)
    try:
        time.sleep(8)   # let both QUIC links come up (redial every 2 s)
        payload = os.urandom(int(os.environ.get('TUNNEL_DBG_N', 5000)))
        # far receivers
        ra = socket.create_connection(('127.0.0.1', B_CLI), timeout=10)
        rb = socket.create_connection(('127.0.0.1', A_CLI), timeout=10)
        ra.settimeout(25); rb.settimeout(25)
        # near senders (both directions at once)
        def snd(port, blob):
            s = socket.create_connection(('127.0.0.1', port), timeout=10)
            s.sendall(blob)
            s.shutdown(socket.SHUT_WR)
        t1 = threading.Thread(target=snd, args=(B_SRV, payload))
        t2 = threading.Thread(target=snd, args=(A_SRV, payload))
        t1.start(); t2.start()
        t1.join(); t2.join()
        def drain(sock, want, timeout=25):
            d = b''
            try:
                while len(d) < want:
                    c = sock.recv(65536)
                    if not c:
                        break
                    d += c
            except socket.timeout:
                pass
            return d
        import hashlib
        got_a = drain(ra, len(payload))   # A->B at B_CLI
        got_b = drain(rb, len(payload))   # B->A at A_CLI
        for tag, got in (('A->B @B_CLI', got_a), ('B->A @A_CLI', got_b)):
            ok = got == payload
            print(f'{tag}: got {len(got)}/{len(payload)} '
                  f'sha={"MATCH" if ok else "DIFF " + hashlib.sha256(got).hexdigest()[:12] + "/" + hashlib.sha256(payload).hexdigest()[:12]}',
                  flush=True)
        for p, lg in ((a, 'a'), (b, 'b')):
            print(f'--- {lg}.log tail ---', flush=True)
            for line in open(os.path.join(OUT, f'{lg}.log'), errors='replace').readlines()[-16:]:
                print('   ', line.rstrip(), flush=True)
    finally:
        for p in (a, b):
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGINT)
                    p.wait(timeout=10)
                except Exception:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    except Exception:
                        pass
        time.sleep(1)


if __name__ == '__main__':
    main()
