# file_to_video_tunnel

Carry a QUIC/VPN byte tunnel **inside a live video stream**. Arbitrary TCP
connections (the VPN side) are multiplexed onto a QUIC stream, and that
stream is then packed into "7030" color frames — small 8×8 blocks, 3 bits
per block, each frame self-validating (magic + seq + CRC). The frames are
rendered to rawvideo, encoded (x264) and pushed over any video service:
**SRT, RTMP, OBS, or a file**. A receiver decodes the stream back into
frames, verifies each one, and hands the recovered QUIC datagrams back to
the tunnel.

The headline property: **a lost video frame is not a lost packet**. Because
each frame self-validates and the QUIC layer sits on top, a dropped or
corrupted frame simply never ACKs its datagrams, and **QUIC retransmits
them** (carried by later frames). The video layer is a plain *lossy,
self-framing* pipe; reliability is QUIC's job.

```
  video service (SRT / RTMP / OBS / file)     <- your --ffmpeg-in/out
  7030 color frames, R copies, self-framing   <- render_group / decode_group
  QUIC datagrams (64-byte slots in each frame)
  QUIC (aioquic: TLS 1.3, streams, 0-RTT)     <- rides the video, no ports
  mux byte stream (conn | len16 | flags | data)
  local TCP connections (the VPN side)
```

## Stacks in this repo

Three generations live side by side (oldest → newest); each is self-contained.

| Stack | Node | Test | Transport |
|-------|------|------|-----------|
| SRT/ffmpeg | `tunnel_node.py` (+`tunnel_send.py`, `tunnel_receive.py`) | `test_e2e.py`, `test_bidi.py`, `test_failover.py` | ffmpeg byte→video over SRT |
| QUIC underlay | `quic_node.py` | `test_quic.py`, `test_quic_failover.py` | aioquic over a direct UDP link |
| **QUIC inside video** | `vqic_tunnel.py` | `test_vqic_e2e.py` | QUIC datagrams packed into 7030 frames, over SRT/RTMP/file |

Shared core: `tunnel_core.py` (frame render/decode, FEC, group capacity),
`bitcoder_fec.py` (GF(256) Cauchy MDS FEC + header/CRC), `mux.py` /
`demux.py` (connection multiplexing), `video_cc.py` (a constant-window
congestion controller; the node sizes it to the carrier's bandwidth-delay
product so it scales with resolution and fps).

The **current** stack is `vqic_tunnel.py` — it is the one that rides the
video and is the one the benchmarks below refer to.

## The 7030 frame codec

Each frame carries a self-validating 8-byte header (magic + 16-bit seq +
CRC32) over a payload of complete QUIC datagrams packed as fixed 64-byte
slots `[len16][data][pad]`. A datagram never spans two frames (the current
frame is zero-padded and closed first), so there is no stream phase to keep
— a frame is either fully valid or discarded, and garbage costs exactly
itself.

- **R copies**: the sender renders each logical frame in `--copies`
  consecutive copies (default 2, the proven 7030 preset) so the blocks
  survive x264 smearing. The receiver uses the first CRC-valid copy
  (a valid copy is bit-exact on its own — no averaging needed).
- **grid transport** (opt-in `VQIC_GRID=1`): the node↔ffmpeg pipe carries a
  compact `W/8 × H/8 × 3` block grid instead of the full `W×H×3` frame
  (64× smaller at 4K). ffmpeg upscales before encode and downscales after
  decode; the wire stays true resolution and the FEC is byte-exact. This
  removed the pipe-I/O wall that capped 4K throughput.

## Run

Two instances, one per end. Each owns two ffmpeg points and a local TCP side:

```
# node A (client)
python3 vqic_tunnel.py --role client \
  --ffmpeg-out "-f rawvideo -s 1280x720 -pix_fmt rgb24 -r 30 -i - -vf scale=1280:720:flags=neighbor -c:v libx264 -pix_fmt yuv420p -preset ultrafast -tune zerolatency -crf 23 -g 1 -bf 0 -f mpegts srt://<B>:19201?mode=caller" \
  --ffmpeg-in  "-i 'srt://127.0.0.1:19200?mode=listener' -map 0:v:0 -vf format=rgb24,scale=160:90:flags=neighbor -c:v rawvideo -pix_fmt rgb24 -f rawvideo -" \
  --tcp-srv-port 19101 --width 1280 --height 720 --fps 30 --copies 2

# node B (server) — symmetric; --tcp-cli-port points at the local echo/dest
python3 vqic_tunnel.py --role server \
  --ffmpeg-out "... -f mpegts srt://127.0.0.1:19200?mode=caller" \
  --ffmpeg-in  "-i 'srt://127.0.0.1:19201?mode=listener' ..." \
  --tcp-cli-port <dest> --width 1280 --height 720 --fps 30 --copies 2
```

The e2e harness wires all of this up for you — see `test_vqic_e2e.py`.

## Test

```
# unit (frame codec, FEC repair, mux/demux, seq wrap) — fast
python3 test_core.py

# full end-to-end, two nodes, byte-exact (default 720p @30fps)
python3 test_vqic_e2e.py PAYLOAD_KB TIMEOUT_S [720p|1080p|4k] [copies] [crf] [fps]
# e.g. 1 MB, 90 s budget, 720p, R=2, real-streaming CRF23
python3 test_vqic_e2e.py 1024 90 720p 2 23
# 60 fps doubles the group ceiling (30 groups/s) — every resolution
python3 test_vqic_e2e.py 4096 140 720p 2 23 60

# grid transport (opt-in)
VQIC_GRID=1 python3 test_vqic_e2e.py 1024 90 720p 2 23

# latency probe (clean small-write RTT through both nodes; LP_FPS for 60fps)
python3 lat_probe.py

# forced frame-loss: physically removes video frames (VQIC_DROP_PCT) and
# proves QUIC retransmits their data — payload stays byte-exact
python3 test_vqic_loss.py PAYLOAD_KB TIMEOUT_S [drop_pct]   # e.g. 512 150 40
```

Older-stack e2e: `test_e2e.py`, `test_bidi.py`, `test_failover.py`,
`test_loss_fec.py`, `test_quic.py`, `test_quic_failover.py`.

## Measured (grid transport, CRF23, R=2, byte-exact)

| Resolution | 30 fps | 60 fps |
|-----------:|-------:|-------:|
| 720p | ~43 KB/s | ~95 KB/s |
| 1080p | ~137 KB/s | ~278 KB/s |
| 4K | ~562 KB/s | ~810 KB/s |

60 fps roughly doubles 720p/1080p (the group ceiling is `fps / R` groups/s:
15 → 30). 4K gains less (×1.44) and sits well under its 1422 KB/s ceiling —
at that geometry the bottleneck moves to encode+scale, not the tunnel.

**Why 60 fps also lowers latency.** Small-write round-trip drops from ~1.7 s
at 30 fps to ~1.07 s at 60 fps on the same 720p carrier (window floored at
1 MB in both, so the drop is fps, not window). Faster frames flush the
per-end quiet-detection tails more often, so an interactive write waits less
before its partial frame ships. Tunables: `VQIC_IDLE_FLUSH_S`,
`VQIC_DRAIN_STALE_S`.

**The window is auto-sized.** `video_cc` keeps a constant window, and the
node sizes it to the carrier's bandwidth-delay product (`2 × B × fps/R ×
RTT`, floor 1 MB) before the QUIC connection opens. This is what makes 60
fps pay off on 4K: at 30 fps a fixed 1 MB window is fine (BDP < 1 MB), but
doubling fps doubles the BDP and a stale 1 MB window starves the pipe
(4K: 1 MB → 697 KB/s, auto ≈4.9 MB → 810 KB/s). Override with
`VQIC_CWND_BYTES`; tune the RTT estimate with `VQIC_RTT_EST_S`.

Raising throughput beyond the fps ceiling means changing the preset
(`--copies 1` for ×2 at the cost of the redundancy copy), not further
optimization.

**Frame loss** (`test_vqic_loss.py`, frames physically removed from the
video): 512 KB stays byte-exact at any tested loss; the frame count in the
video really shrinks and QUIC retransmits the missing frames' datagrams
(factor ≈ 1/(1−p)). 720p grid CRF23 R=2: 0% → 13 s, 20% → 24 s, 40% →
40 s, 60% → 98 s.

## Install

```
pip install --user --break-system-packages aioquic cryptography numpy pillow
# and a system ffmpeg with libx264 + SRT
```

`requirements.txt` lists the Python dependencies.

## Kill hygiene

These tests spawn long-lived node + ffmpeg processes. If a run is
interrupted, kill by explicit PID (a `/proc` scan for the exact script
name), never a `pkill -f` pattern that can match the running shell.
