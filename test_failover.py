#!/usr/bin/env python3
"""Failover test: node B dies hard (SIGKILL) mid-session, then comes back.
Both directions must work again:
  A->B: exercises A's caller re-dial (already implemented)
  B->A: exercises A's listener re-spawn (the fix under test)
"""
import hashlib
import os
import signal
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'e2e3')
os.makedirs(OUT, exist_ok=True)

W, H, PASS = 640, 360, 'fo-pass-123'   # SRT requires passphrase 10-128 chars
A_SRT_SRV, A_SRT_CLI = 9230, 9220
B_SRT_SRV, B_SRT_CLI = 9220, 9230
A_TCP_SRV, A_TCP_CLI = 9310, 9311
B_TCP_SRV, B_TCP_CLI = 9320, 9321


def start_node(name, srt_srv, srt_cli, tcp_srv, tcp_cli):
    logp = f'{OUT}/{name}.log'
    open(logp, 'wb').close()
    log = open(logp, 'ab')
    p = subprocess.Popen(
        [sys.executable, f'{HERE}/tunnel_node.py',
         f'--srt-listen', f'srt://127.0.0.1:{srt_srv}',
         f'--srt-dial', f'srt://127.0.0.1:{srt_cli}',
         f'--srt-passphrase', PASS,
         f'--tcp-srv-port', str(tcp_srv), f'--tcp-cli-port', str(tcp_cli),
         '--width', str(W), '--height', str(H)],
        stdout=log, stderr=subprocess.STDOUT, cwd=HERE,
        preexec_fn=os.setsid)
    rf = os.open(logp, os.O_RDONLY)
    return p, rf, log


def wait_for(rfs, *patterns, timeout=120.0):
    deadline = time.time() + timeout
    logs = {fd: b'' for fd in rfs}
    while time.time() < deadline:
        for fd in list(rfs):
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                continue
            if chunk:
                logs[fd] += chunk
        # EVERY node's log must contain EVERY pattern (readiness of both
        # directions), not just one of the logs:
        if all(all(p.encode() in logs[fd] for p in patterns)
               for fd in rfs):
            return True
        time.sleep(0.2)
    for fd in rfs:
        try:
            os.close(fd)
        except OSError:
            pass
    print(f'--- wait_for timeout, last 30 lines per node:')
    for fd in logs:
        tail = logs[fd].decode(errors='replace').splitlines()[-30:]
        print('\n'.join(tail))
    return False


def send_payload(port, blob, name):
    s = socket.create_connection(('127.0.0.1', port), timeout=15)
    s.sendall(blob)
    s.shutdown(socket.SHUT_WR)
    s.close()
    print(f'  sent {name}: {len(blob)} B')


def recv_payload(port, timeout, path):
    s = socket.create_connection(('127.0.0.1', port), timeout=15)
    s.settimeout(timeout)
    data = b''
    t0 = time.time()
    try:
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
            if len(data) > 2 * timeout:
                break
    except socket.timeout:
        pass
    s.close()
    with open(path, 'wb') as f:
        f.write(data)
    return data, time.time() - t0


def check(got, blob, name, dt):
    ok = len(got) == len(blob) and hashlib.sha256(got).digest() == \
        hashlib.sha256(blob).digest()
    print(f'[{"OK" if ok else "FAIL"}] {name}: got {len(got)}/{len(blob)} B '
          f'({"sha matches" if ok else "MISMATCH"}), {dt:.1f}s')
    return ok


def kill_hard(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        pass
    p.wait()


def main():
    for port in (A_SRT_SRV, A_SRT_CLI, A_TCP_SRV, A_TCP_CLI,
                 B_SRT_SRV, B_TCP_SRV, B_TCP_CLI):
        try:
            s = socket.create_connection(('127.0.0.1', port), timeout=0.3)
            print(f'!! port {port} busy, aborting')
            s.close()
            return 1
        except OSError:
            pass

    a, la_r, la_log = start_node('A', A_SRT_SRV, A_SRT_CLI, A_TCP_SRV, A_TCP_CLI)
    b, lb_r, lb_log = start_node('B', B_SRT_SRV, B_SRT_CLI, B_TCP_SRV, B_TCP_CLI)
    fb = fb_r = fb_log = None
    ok_all = True
    try:
        ok = wait_for([la_r, lb_r],
                      'ffmpeg up, SRT dial target', 'peer meta')
        if not ok:
            print('FAIL: links did not come up')
            return 1

        # --- phase 0: baseline ------------------------------------------------
        blob1 = os.urandom(30000)
        send_payload(B_TCP_SRV, blob1, 'A->B pre-failover')
        got, dt = recv_payload(A_TCP_CLI, 90, f'{OUT}/got1')
        ok_all &= check(got, blob1, 'A->B pre-failover', dt)

        # --- phase 1: kill B hard --------------------------------------------
        print('  killing B (SIGKILL)')
        kill_hard(b)
        os.close(lb_r)
        lb_log.close()
        time.sleep(3)

        # --- phase 2: restart B, both directions must work again -------------
        b, fb_r, fb_log = start_node('B', B_SRT_SRV, B_SRT_CLI,
                                     B_TCP_SRV, B_TCP_CLI)
        time.sleep(1)

        blob3 = os.urandom(20000)
        send_payload(A_TCP_SRV, blob3, 'B->A post-failover')
        got3, dt3 = recv_payload(B_TCP_CLI, 120, f'{OUT}/got3')
        ok_all &= check(got3, blob3, 'B->A post-failover', dt3)

        blob2 = os.urandom(40000)
        send_payload(B_TCP_SRV, blob2, 'A->B post-failover')
        got2, dt2 = recv_payload(A_TCP_CLI, 120, f'{OUT}/got2')
        ok_all &= check(got2, blob2, 'A->B post-failover', dt2)

        print('RESULT:', 'PASS' if ok_all else 'FAIL')
        return 0 if ok_all else 2
    finally:
        for p, lg, rf in ((a, la_log, la_r), (b, fb_log, fb_r)):
            if p is not None and p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGINT)
                    p.wait(timeout=15)
                except Exception:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    except Exception:
                        pass
            if lg is not None:
                lg.close()
            if rf is not None:
                try:
                    os.close(rf)
                except OSError:
                    pass
        time.sleep(1)


if __name__ == '__main__':
    sys.exit(main())
