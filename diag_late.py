#!/usr/bin/env python3
"""Isolated diagnostic: does a LATE receiver (connects after the sender
already sent) get the full payload? Runs two nodes with --dump so the
receiver's dumper shows every byte the walker actually delivered,
independent of socket timing."""
import hashlib
import os
import signal
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'diag_late')
os.makedirs(OUT, exist_ok=True)
for sub in ('a', 'b'):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

W, H, PASS = 640, 360, 'diag-late-key-1'
# A: listen 9250 (recv), dial 9251 (send->B)
# B: listen 9251 (recv), dial 9250 (send->A)
A_TCP_SRV, A_TCP_CLI = 9350, 9351
B_TCP_SRV, B_TCP_CLI = 9360, 9361


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def start_node(name, srt_listen, srt_dial, tcp_srv, tcp_cli):
    logp = f'{OUT}/{name}.log'
    open(logp, 'w').close()
    log = open(logp, 'ab')
    p = subprocess.Popen(
        [sys.executable, f'{HERE}/tunnel_node.py',
         f'--srt-listen', f'srt://127.0.0.1:{srt_listen}',
         f'--srt-dial', f'srt://127.0.0.1:{srt_dial}',
         '--srt-passphrase', PASS,
         f'--tcp-srv-port', str(tcp_srv), f'--tcp-cli-port', str(tcp_cli),
         '--width', str(W), '--height', str(H),
         '--out', os.path.join(OUT, name), '--dump'],
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


def main():
    PAYLOAD = os.urandom(30000)
    print(f'payload: {len(PAYLOAD)} B sha {sha(PAYLOAD)}')
    a, la = start_node('a', 9250, 9251, A_TCP_SRV, A_TCP_CLI)
    b, lb = start_node('b', 9251, 9250, B_TCP_SRV, B_TCP_CLI)
    try:
        ready = wait_log(la, ['peer meta', 'ffmpeg up, SRT dial target']) and \
            wait_log(lb, ['peer meta', 'ffmpeg up, SRT dial target'])
        if not ready:
            print('FAIL: nodes did not come up')
            return 1
        print('[OK] both nodes up')

        # SENDER connects to B's send server, sends, shuts down.
        s = socket.create_connection(('127.0.0.1', B_TCP_SRV), timeout=15)
        s.sendall(PAYLOAD)
        s.shutdown(socket.SHUT_WR)
        s.close()
        print(f'[OK] sender handed off {len(PAYLOAD)} B (B->A direction)')

        # LATE receiver: wait 1s, THEN connect to A's client port.
        time.sleep(1.0)
        r = socket.create_connection(('127.0.0.1', A_TCP_CLI), timeout=15)
        r.settimeout(30)
        got = b''
        try:
            while True:
                d = r.recv(65536)
                if not d:
                    break
                got += d
                if len(got) > 2 * len(PAYLOAD):
                    break
        except socket.timeout:
            pass
        r.close()
        # save both for a precise byte-level diff
        with open(f'{OUT}/got.bin', 'wb') as f:
            f.write(got)
        with open(f'{OUT}/want.bin', 'wb') as f:
            f.write(PAYLOAD)
        ok = got == PAYLOAD
        # byte-level diff
        n = min(len(got), len(PAYLOAD))
        first_diff = next((i for i in range(n)
                           if got[i] != PAYLOAD[i]), None)
        print(f'[{"OK" if ok else "FAIL"}] late receiver got {len(got)}/'
              f'{len(PAYLOAD)} B')
        print(f'   got sha  {sha(got)}')
        print(f'   want sha {sha(PAYLOAD)}')
        if first_diff is not None:
            print(f'   first byte diff at offset {first_diff}: '
                  f'got {got[first_diff]:02x} want {PAYLOAD[first_diff]:02x}')
            # is `got` a rotated / shifted view of the payload?
            for shift in (1, 2, 4, 8, 1326):
                if len(got) >= shift and got[shift:shift + n] == PAYLOAD[:n]:
                    print(f'   NOTE: got is PAYLOAD shifted by +{shift} '
                          f'(prefix dropped)')
                    break
        else:
            print('   (no byte diff in common prefix)')

        # let dumper flush
        time.sleep(2)
        da = open(f'{OUT}/a/tunnel_dump.bin', 'rb').read()
        db = open(f'{OUT}/b/tunnel_dump.bin', 'rb').read()
        print(f'[info] A dump (receiver of payload) = {len(da)} B '
              f'(want {len(PAYLOAD)})')
        print(f'[info] B dump (send side)           = {len(db)} B')
        if len(da) >= len(PAYLOAD) and da[:len(PAYLOAD)] == PAYLOAD:
            print('[OK] A dumper holds full payload -> walker OK, '
                  'socket delivery dropped the tail')
        elif 0 < len(da) < len(PAYLOAD):
            print(f'[info] A dumper has {len(da)} B -> walker/recv lost the '
                  f'tail ({len(PAYLOAD)-len(da)} B missing)')
        else:
            print('[info] A dumper empty')
        return 0 if ok else 2
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
