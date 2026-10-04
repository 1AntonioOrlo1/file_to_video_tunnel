#!/usr/bin/env python3
"""Diagnostic driver: run 2 cross-linked nodes WITH --raw-dump, push random
data both ways, then we decode the raw-dumps offline to separate a walker
logic bug from a real transport corruption under load."""
import os
import socket
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, 'e2e2')
os.makedirs(WORK, exist_ok=True)

PASS = 'diag-pass-123'
A_SRV, A_CLI = 9110, 9111
B_SRV, B_CLI = 9120, 9121


def wait_log(path, needle, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if needle in open(path, errors='replace').read():
                return True
        except OSError:
            pass
        time.sleep(0.3)
    return False


def send_random(port, n):
    try:
        c = socket.create_connection(('127.0.0.1', port), timeout=10)
        c.sendall(os.urandom(n))
        c.shutdown(socket.SHUT_WR)
        c.close()
        return True
    except OSError as e:
        print(f"  send to {port} failed: {e}")
        return False


def main():
    for f in ('ra.raw', 'rb.raw', 'na.log', 'nb.log'):
        try:
            os.remove(os.path.join(WORK, f))
        except OSError:
            pass
    for d in ('a', 'b'):
        for f in ('tunnel_dump.bin',):
            try:
                os.remove(os.path.join(WORK, d, f))
            except OSError:
                pass

    node_a = subprocess.Popen(
        [sys.executable, 'tunnel_node.py',
         '--srt-listen', f'srt://127.0.0.1:9030',
         '--srt-dial', f'srt://127.0.0.1:9020',
         '--srt-passphrase', PASS,
         '--tcp-srv-port', str(A_SRV), '--tcp-cli-port', str(A_CLI),
         '--width', '640', '--height', '360',
         '--out', os.path.join(WORK, 'a'), '--dump',
         '--raw-dump', os.path.join(WORK, 'ra.raw')],
        cwd=HERE, stdout=open(os.path.join(WORK, 'na.log'), 'w'),
        stderr=subprocess.STDOUT)
    node_b = subprocess.Popen(
        [sys.executable, 'tunnel_node.py',
         '--srt-listen', f'srt://127.0.0.1:9020',
         '--srt-dial', f'srt://127.0.0.1:9030',
         '--srt-passphrase', PASS,
         '--tcp-srv-port', str(B_SRV), '--tcp-cli-port', str(B_CLI),
         '--width', '640', '--height', '360',
         '--out', os.path.join(WORK, 'b'), '--dump',
         '--raw-dump', os.path.join(WORK, 'rb.raw')],
        cwd=HERE, stdout=open(os.path.join(WORK, 'nb.log'), 'w'),
        stderr=subprocess.STDOUT)

    try:
        ra = wait_log(os.path.join(WORK, 'na.log'), 'ffmpeg up, SRT dial target')
        rb = wait_log(os.path.join(WORK, 'nb.log'), 'ffmpeg up, SRT dial target')
        print(f"nodes up: a={ra} b={rb}")
        # let the links settle, then push data both ways (bidirectional at once)
        time.sleep(4)
        ta = threading.Thread(target=send_random, args=(A_SRV, 40000))
        tb = threading.Thread(target=send_random, args=(B_SRV, 40000))
        ta.start(); tb.start(); ta.join(); tb.join()
        print("data pushed both ways (40KB each); letting it stream...")
        time.sleep(12)
    finally:
        for n in (node_a, node_b):
            try:
                n.terminate()
                n.wait(timeout=12)
            except Exception:
                n.kill()

    for f in ('ra.raw', 'rb.raw'):
        p = os.path.join(WORK, f)
        sz = os.path.getsize(p) if os.path.exists(p) else 0
        print(f"raw dump {f}: {sz} bytes = {sz // (640*360*3)} frames")


if __name__ == '__main__':
    main()
