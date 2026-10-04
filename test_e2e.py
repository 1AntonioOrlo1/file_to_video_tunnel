#!/usr/bin/env python3
"""End-to-end loopback test: real SRT connection, real ffmpeg both ends,
random data through the tunnel, byte-compare at the far side.

Also prints an ss snapshot proving the SRT link is live UDP (not a file
copy), and the sender/receiver logs.
"""
import hashlib
import os
import socket
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, 'e2e')
os.makedirs(WORK, exist_ok=True)

SRT_PORT = 9899
TCP_SEND_PORT = 9900
TCP_RECV_PORT = 9901
PY = sys.executable
PYTHONDONTWRITEBYTECODE = 1


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def wait_log(path, needle, timeout, what):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(path):
            txt = open(path, errors='replace').read()
            if needle in txt:
                return True
        time.sleep(0.3)
    print(f"  [FAIL] {what}: '{needle}' not found in {path}")
    print("  --- log tail ---")
    if os.path.exists(path):
        print(open(path, errors='replace').read()[-2000:])
    return False


def main():
    print(f"== e2e tunnel test (SRT port {SRT_PORT}) ==")
    recv = subprocess.Popen(
        [PY, 'tunnel_receive.py', '--srt', f'srt://127.0.0.1:{SRT_PORT}',
         '--tcp-port', str(TCP_RECV_PORT), '--dump', '--out', WORK],
        cwd=HERE, stdout=open(os.path.join(WORK, 'recv.log'), 'w'),
        stderr=subprocess.STDOUT)
    time.sleep(1.0)
    snd = subprocess.Popen(
        [PY, 'tunnel_send.py', '--srt', f'srt://127.0.0.1:{SRT_PORT}',
         '--tcp-port', str(TCP_SEND_PORT)],
        cwd=HERE, stdout=open(os.path.join(WORK, 'send.log'), 'w'),
        stderr=subprocess.STDOUT)
    time.sleep(1.0)

    ok = True
    ok &= wait_log(os.path.join(WORK, 'recv.log'), 'tunnel TCP serving', 90,
                   'receiver up')
    ok &= wait_log(os.path.join(WORK, 'send.log'), 'metadata header sent', 90,
                   'sender up')
    if not ok:
        cleanup(snd, recv)
        return 1

    # prove the SRT link is live UDP on the wire
    ss = subprocess.run(['ss', '-unp'], capture_output=True, text=True)
    srt_lines = [l for l in ss.stdout.splitlines() if str(SRT_PORT) in l]
    print(f"  ss -u snapshot (SRT link): {len(srt_lines)} matching line(s)")
    for l in srt_lines[:4]:
        print('   ', l[:120])

    payload = os.urandom(75000)
    print(f"  payload: {len(payload)} B random, sha256 {sha(payload)}")

    # local client on the receiver side (connects BEFORE data flows)
    lsock = socket.create_connection(('127.0.0.1', TCP_RECV_PORT), timeout=10)
    received = b''

    def drain():
        nonlocal received
        try:
            while True:
                d = lsock.recv(65536)
                if not d:
                    break
                received += d
        except (OSError, socket.timeout):
            pass

    dr = threading.Thread(target=drain, daemon=True)
    dr.start()
    time.sleep(0.5)

    # client on the sender side
    csock = socket.create_connection(('127.0.0.1', TCP_SEND_PORT), timeout=10)
    t0 = time.time()
    csock.sendall(payload)
    csock.shutdown(socket.SHUT_WR)
    print(f"  sender-side client sent {len(payload)} B at t=0")

    # wait for FIN (EOF on the local socket) or a hard timeout
    t_end = t0 + 60
    while len(received) < len(payload) and time.time() < t_end:
        time.sleep(0.2)
    if len(received) >= len(payload):
        time.sleep(1.0)
        dr.join(timeout=15)
    elapsed = time.time() - t0

    got = received[:len(payload)]
    match = sha(got) == sha(payload)
    print(f"  received: {len(received)} B in {elapsed:.1f}s "
          f"(~{len(payload)/max(elapsed,0.001)/1024:.1f} KB/s)")
    print(f"  [ {'OK' if match else 'FAIL'} ] far-side data identical "
          f"(sha {sha(got)})")
    ok &= match

    # sender-side client should see its connection close (FIN)
    try:
        tail = csock.recv(4096)
        print(f"  sender-side client got EOF after sending: {not tail}")
    except OSError:
        print("  sender-side client socket closed: True")

    # dump file should also hold the bytes
    dump = open(os.path.join(WORK, 'tunnel_dump.bin'), 'rb').read()
    print(f"  [ {'OK' if sha(dump) == sha(payload) else 'FAIL'} ] "
          f"dump file matches ({len(dump)} B)")
    ok &= sha(dump) == sha(payload)

    # now shut the sender down (SIGINT) -> shutdown EOF -> stream ends ->
    # the receiver should exit on its own
    print("  sending SIGINT to sender (clean shutdown test)...")
    snd.send_signal(2)
    try:
        rc_snd = snd.wait(timeout=30)
    except subprocess.TimeoutExpired:
        rc_snd = None
        snd.kill()
    print(f"  sender exit code: {rc_snd}")
    t0 = time.time()
    while time.time() - t0 < 40 and recv.poll() is None:
        time.sleep(0.5)
    if recv.poll() is None:
        print("  [FAIL] receiver did not exit after sender shutdown; killing")
        recv.kill()
        ok = False
    else:
        print(f"  [OK] receiver exited on its own "
              f"(code {recv.returncode}, {time.time()-t0:.1f}s)")

    print("\n-- sender log --")
    print(open(os.path.join(WORK, 'send.log'), errors='replace').read()[-1500:])
    print("\n-- receiver log --")
    print(open(os.path.join(WORK, 'recv.log'), errors='replace').read()[-1500:])

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cleanup(snd, recv):
    for p in (snd, recv):
        if p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                p.kill()


if __name__ == '__main__':
    try:
        sys.exit(main())
    finally:
        pass
