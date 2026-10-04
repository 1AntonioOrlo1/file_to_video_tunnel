#!/usr/bin/env python3
"""Two-node bidirectional tunnel test.

node A:  --srt-listen :9030  --srt-dial 127.0.0.1:9020   (A sends -> B)
node B:  --srt-listen :9020  --srt-dial 127.0.0.1:9030   (B sends -> A)

Channel 1 (A->B): a client connects to A's tcp-srv port, sends PAYLOAD_A,
and the far client on B's tcp-cli port must receive exactly PAYLOAD_A.
Channel 2 (B->A): the same in reverse with PAYLOAD_B.

Both directions are exercised SIMULTANEOUSLY (two live SRT video channels
at once), which is the whole point of the node design.
"""
import hashlib
import os
import socket
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, 'e2e2')
os.makedirs(WORK, exist_ok=True)

# channel 1: A sends (dial 9020), B receives (listen 9020)
# channel 2: B sends (dial 9030), A receives (listen 9030)
A_SRV, A_CLI = 9110, 9111   # A: clients->A-SRV go out to B; B's data->A-CLI
B_SRV, B_CLI = 9120, 9121

PAYLOAD_A = os.urandom(32000)   # A -> B  (~2 s at 16 KB/s)
PAYLOAD_B = os.urandom(48000)   # B -> A  (~3 s at 16 KB/s)


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def recv_until(sock, want, timeout=120):
    """Read until `want` bytes arrive OR the peer closes (EOF)."""
    data = b''
    t0 = time.time()
    while len(data) < want and time.time() - t0 < timeout:
        try:
            d = sock.recv(65536)
        except (OSError, socket.timeout):
            break
        if not d:
            break
        data += d
    return data


def wait_log(path, needles, timeout=90):
    """Wait until all needle strings appear in the log file."""
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
    ok = True
    print(f"payload A ({len(PAYLOAD_A)} B, A->B): {sha(PAYLOAD_A)}")
    print(f"payload B ({len(PAYLOAD_B)} B, B->A): {sha(PAYLOAD_B)}")

    log_a = os.path.join(WORK, 'a.log')
    log_b = os.path.join(WORK, 'b.log')
    for p in (log_a, log_b):
        open(p, 'w').close()

    # shared SRT AES-CTR passphrase: the whole UDP video stream is encrypted
    PASS = 'tunnel-e2e-key-42'
    node_a = subprocess.Popen(
        [sys.executable, 'tunnel_node.py',
         '--srt-listen', 'srt://127.0.0.1:9030',
         '--srt-dial', 'srt://127.0.0.1:9020',
         '--srt-passphrase', PASS,
         '--tcp-srv-port', str(A_SRV), '--tcp-cli-port', str(A_CLI),
         '--width', '640', '--height', '360',
         '--out', os.path.join(WORK, 'a'), '--dump'],
        cwd=HERE, stdout=open(log_a, 'w'), stderr=subprocess.STDOUT)
    node_b = subprocess.Popen(
        [sys.executable, 'tunnel_node.py',
         '--srt-listen', 'srt://127.0.0.1:9020',
         '--srt-dial', 'srt://127.0.0.1:9030',
         '--srt-passphrase', PASS,
         '--tcp-srv-port', str(B_SRV), '--tcp-cli-port', str(B_CLI),
         '--width', '640', '--height', '360',
         '--out', os.path.join(WORK, 'b'), '--dump'],
        cwd=HERE, stdout=open(log_b, 'w'), stderr=subprocess.STDOUT)

    try:
        # wait for both nodes: SRT dial ffmpeg up + local client port bound
        ready_a = wait_log(log_a, ['ffmpeg up, SRT dial target',
                                   'local TCP client-port'])
        ready_b = wait_log(log_b, ['ffmpeg up, SRT dial target',
                                   'local TCP client-port'])
        # the SRT caller connects to the peer listener: give the link a
        # moment to settle before starting traffic
        time.sleep(4.0)
        if not (ready_a and ready_b):
            print(f"[FAIL] nodes did not come up (a={ready_a} b={ready_b})")
            return 1
        print("[OK] both nodes up, SRT links established")

        # far-side clients (on the receiving nodes' client ports)
        far_a = socket.create_connection(('127.0.0.1', A_CLI), timeout=15)
        far_b = socket.create_connection(('127.0.0.1', B_CLI), timeout=15)
        print("[OK] far-side clients connected on both nodes")

        # near-side clients send SIMULTANEOUSLY (two directions at once)
        def send_a():
            s = socket.create_connection(('127.0.0.1', A_SRV), timeout=15)
            s.sendall(PAYLOAD_A)
            s.shutdown(socket.SHUT_WR)

        def send_b():
            s = socket.create_connection(('127.0.0.1', B_SRV), timeout=15)
            s.sendall(PAYLOAD_B)
            s.shutdown(socket.SHUT_WR)

        t0 = time.time()
        th_a = threading.Thread(target=send_a)
        th_b = threading.Thread(target=send_b)
        th_a.start(); th_b.start()
        th_a.join(); th_b.join()
        print(f"[OK] both payloads handed off ({time.time()-t0:.2f}s)")

        # wait for the far side to receive each full payload (or EOF).
        # far_a (on A_CLI) receives B's payload; far_b (on B_CLI) A's.
        got_b_far = recv_until(far_b, len(PAYLOAD_A))  # A->B, at B_CLI
        got_a_far = recv_until(far_a, len(PAYLOAD_B))  # B->A, at A_CLI
        elapsed = time.time() - t0

        a_ok = got_b_far[:len(PAYLOAD_A)] == PAYLOAD_A
        b_ok = got_a_far[:len(PAYLOAD_B)] == PAYLOAD_B
        print(f"[{'OK' if a_ok else 'FAIL'}] channel A->B: "
              f"got {len(got_b_far)}/{len(PAYLOAD_A)} B, sha {sha(got_b_far[:len(PAYLOAD_A)])} "
              f"(want {sha(PAYLOAD_A)})")
        print(f"[{'OK' if b_ok else 'FAIL'}] channel B->A: "
              f"got {len(got_a_far)}/{len(PAYLOAD_B)} B, sha {sha(got_a_far[:len(PAYLOAD_B)])} "
              f"(want {sha(PAYLOAD_B)})")
        print(f"both directions in {elapsed:.1f}s total")
        ok = a_ok and b_ok
    finally:
        # Graceful shutdown: SIGINT runs the node's finally-block (send EOF,
        # drain the dumper, close ffmpeg). SIGTERM would skip it and lose the
        # buffered dump. Fall back to terminate/kill if SIGINT stalls.
        import signal
        for n in (node_a, node_b):
            try:
                n.send_signal(signal.SIGINT)
                n.wait(timeout=20)
            except Exception:
                try:
                    n.terminate()
                    n.wait(timeout=5)
                except Exception:
                    n.kill()

    # dump files should also hold the payloads
    da = open(os.path.join(WORK, 'a', 'tunnel_dump.bin'), 'rb').read()
    db = open(os.path.join(WORK, 'b', 'tunnel_dump.bin'), 'rb').read()
    d_b_ok = sha(db[:len(PAYLOAD_A)]) == sha(PAYLOAD_A)
    d_a_ok = sha(da[:len(PAYLOAD_B)]) == sha(PAYLOAD_B)
    print(f"[{'OK' if d_b_ok else 'FAIL'}] node B dump holds A's payload "
          f"({len(db)} B)")
    print(f"[{'OK' if d_a_ok else 'FAIL'}] node A dump holds B's payload "
          f"({len(da)} B)")

    print("\n-- node A log (tail) --")
    for l in open(log_a, errors='replace').readlines()[-12:]:
        print('   ', l.rstrip())
    print("\n-- node B log (tail) --")
    for l in open(log_b, errors='replace').readlines()[-12:]:
        print('   ', l.rstrip())

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
