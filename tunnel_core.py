"""Shared core for the bitcoder video tunnel (SRT transport).

Reuses the color-mode machinery of file_to_video_bitcoder_color:
  * 8-corner RGB palette, 3 bits per MxM block (chroma-neutral 0/255 levels)
  * 8-byte group header (magic 0x5A, 16-bit seq, crc32 over seq+payload, spare)
  * systematic Cauchy MDS FEC over GF(256) — any k of k+m groups per stripe
    recover the whole stripe
  * R identical copies of every group frame (threshold averaging)

Stream format (live):
  [R_META meta frames (same JSON-in-blocks canvas as the file encoder)]
  [group frames: 1 stripe = k data groups + m parity groups, R copies each]
  [EOF sentinel group: header seq=0xFFFF, payload b'EOF!']

Payload (multiplexed byte stream), one packet per frame:
  conn (1B, 0 = idle) | len (2B big) | flags (1B, bit0=FIN) | data (len bytes)
Every group payload is exactly B bytes (B = group_capacity(width, height)),
so idle fillers keep the stream group-aligned. The receiver demuxes packets
back onto per-connection sockets.
"""

import json
import numpy as np
from PIL import Image

from bitcoder_fec import (HEADER_BITS, HEADER_BYTES, MAGIC, SPARE,
                          cauchy_matrix, crc_for_group, fec_decode, fec_encode,
                          pack_header, unpack_header, render_payload,
                          SeqWrapTracker)

# --- protocol constants -----------------------------------------------------
M = 8                 # block size (even: whole yuv420p chroma samples)
R = 2                 # copies per group frame
R_META = 3            # leading metadata copies
K = 8                 # data groups per stripe
M_PAR = 2             # parity groups per stripe (repairs up to 2 lost groups)
FPS = 30
PKT_MAX = 65535       # max data bytes per mux packet
IDLE_CONN = 0         # conn id reserved for idle filler
EOF_PAYLOAD = b'EOF!'

SHIFT = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=np.uint8)
WEIGHTS = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)


def group_capacity(width, height, m=M):
    """Payload bytes one group frame can carry at this geometry."""
    blocks = (width // m) * (height // m)
    if blocks <= HEADER_BITS:
        raise ValueError(f"{width}x{height} too small for M={m}")
    return (blocks - HEADER_BITS) * 3 // 8


def throughput_bps(width, height, fps=FPS, k=K, m=M_PAR, m_block=M):
    """Approximate tunnel throughput in bytes/s (payload, before FEC).

    Each group occupies R frames, so a stripe of (k+m) groups takes
    (k+m)*R frames = (k+m)*R/fps seconds, and carries k*B payload bytes."""
    b = group_capacity(width, height, m_block)
    return b * k * fps // ((k + m) * R)


def render_group(payload, seq, m=M, width=1280, height=720, grid=False):
    """Render one FEC group frame (header + payload) as raw rgb24 bytes.

    Header: 8 bytes drawn in the first 64 blocks, black/white on ALL THREE
    channels (chroma-neutral, so x264 chroma averaging can never flip it).
    Payload: 3 bits per block, bit 3b,3b+1,3b+2 -> R,G,B of block b, MSB-first.

    grid=True returns the COMPACT block matrix (blocks_y, blocks_x, 3) — one
    0/255 pixel per block — instead of the full MxM-expanded frame. The pipe
    then ships (W/m)*(H/m)*3 bytes (390 KB at 4K) instead of W*H*3 (24.9 MB);
    ffmpeg upscales with scale=neighbor before encode and the FEC decodes the
    grid directly (each pixel IS a block). The wire stays true WxH. Verified
    byte-exact through x264 CRF23 (see /tmp/grid_gate.py).
    """
    B = group_capacity(width, height, m)
    if len(payload) != B:
        raise ValueError(f"payload must be exactly {B} bytes (got {len(payload)})")
    blocks_x = width // m
    blocks_y = height // m
    total = blocks_x * blocks_y
    if total < HEADER_BITS + 1:
        raise ValueError("geometry too small for a color FEC group")
    header = pack_header(seq, payload)
    hbits = ((np.frombuffer(header, dtype=np.uint8)[:, None] >> SHIFT) & 1)
    hflat = np.tile(hbits.reshape(HEADER_BITS, 1), (1, 3)).reshape(-1)
    raw = np.frombuffer(payload, dtype=np.uint8)
    pbits = ((raw[:, None] >> SHIFT) & 1).reshape(-1)
    # The header occupies 64 BLOCKS (192 bits); the payload fills the rest.
    n_payload_blocks = total - HEADER_BITS
    pad = (3 - pbits.size % 3) % 3
    if pad:
        pbits = np.concatenate([pbits, np.zeros(pad, dtype=bool)])
    flat = np.concatenate([hflat, pbits[:3 * n_payload_blocks]])
    # Fast path: build the 0/255 block matrix directly and repeat MxM
    # (byte-identical to the old boolean-zeros+expand; 21 ms vs 57 ms at 4K)
    matrix = (flat.astype(np.uint8)).reshape(blocks_y, blocks_x, 3) * 255
    if grid:
        return matrix.tobytes()
    img = np.repeat(np.repeat(matrix, m, axis=0), m, axis=1)
    return img.tobytes()


def render_meta(meta_dict, width, height):
    """Render the metadata frame (raw rgb24 bytes) with an adaptive block size.

    Same canvas convention as the file encoder: JSON -> bits -> 0/255 blocks.
    The block size is stored in the JSON ('mb'); the receiver probes the
    same candidate list (16, 14, 12, 10, 8, 6, 4) in the same order."""
    meta = dict(meta_dict)
    for block in (16, 14, 12, 10, 8, 6, 4):
        meta['mb'] = block
        b = json.dumps(meta, separators=(',', ':')).encode('utf-8')
        need = int(len(b) * 8 * 1.1)
        if (width // block) * (height // block) >= need:
            break
    else:
        raise ValueError("metadata does not fit the canvas")
    b = json.dumps(meta, separators=(',', ':')).encode('utf-8')
    total_bits = len(b) * 8
    blocks_x = width // block
    blocks_y = height // block
    total_blocks = blocks_x * blocks_y
    bit_array = np.zeros(total_blocks, dtype=bool)
    for i in range(total_bits):
        byte_idx = i // 8
        bit_in_byte = i % 8
        if byte_idx < len(b):
            bit_array[i] = (b[byte_idx] >> (7 - bit_in_byte)) & 1
    bit_matrix = bit_array[:blocks_x * blocks_y].reshape(blocks_y, blocks_x)
    expanded = bit_matrix.repeat(block, axis=0).repeat(block, axis=1)
    full = np.zeros((height, width, 3), dtype=np.uint8)
    full[:blocks_y * block, :blocks_x * block] = expanded[..., None] * 255
    return full.tobytes()


def decode_meta(frames, width, height):
    """Parse the embedded JSON from R_META meta frames. `frames` are raw
    rgb24 byte buffers. Returns the meta dict or None."""
    for block in (16, 14, 12, 10, 8, 6, 4):
        meta = _decode_meta_at(frames, width, height, block)
        if not meta:
            continue
        declared = int(meta.get('mb', 16))
        if declared != block:
            meta2 = _decode_meta_at(frames, width, height, declared)
            if meta2:
                return meta2
        return meta
    return None


def _decode_meta_at(frames, width, height, block):
    try:
        cw = (width // block) * block
        ch = (height // block) * block
        bx = cw // block
        by = ch // block
        total = bx * by
        stacked = np.stack([np.frombuffer(f, dtype=np.uint8).reshape(height, width, 3)
                            [:ch, :cw] for f in frames])
        reshaped = stacked.reshape(len(frames), by, block, bx, block, 3)
        avgs = reshaped.mean(axis=(2, 4, 5))          # per block, per frame
        bits = (avgs.mean(axis=0) >= 128).ravel()[:total].astype(np.uint8)
        bytes_ = (bits.reshape(-1, 8) * WEIGHTS).sum(axis=1, dtype=np.uint8)
        s = bytes_.tobytes().decode('utf-8', errors='ignore')
        end = s.rfind('}')
        if end != -1:
            s = s[:end + 1]
        meta = json.loads(s)
        if 'fn' not in meta:
            return None
        return meta
    except Exception:
        return None


def color_bits_from_frames(frames, m, width, height):
    """Threshold each MxM block across R copies -> (blocks_y, blocks_x, 3)
    bool (R, G, B bits). Frames are raw rgb24 buffers (H*W*3)."""
    cw = (width // m) * m
    ch = (height // m) * m
    bx = cw // m
    by = ch // m
    acc = np.zeros((by, bx, 3), dtype=np.int64)
    for frame in frames:
        arr = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3)
        b = arr[:ch, :cw].reshape(by, m, bx, m, 3)
        acc += b.sum(axis=(1, 3), dtype=np.int64)
    bits = (acc >= 128 * len(frames) * m * m)
    return bits


def decode_group(frames, m, width, height):
    """Decode R raw rgb24 frames back to (header8, payload) or (None, None).

    Header: first 64 blocks, 3 channels drawn identically -> OR them so a
    single-channel smear still recovers the bit. Payload length is the full
    geometry capacity B."""
    try:
        bits = color_bits_from_frames(frames, m, width, height)
        flat = bits.reshape(-1).astype(np.uint8)  # (3*n_blocks,) R,G,B interleaved
        n_blocks = bits.size // 3
        n_payload_bytes = (3 * (n_blocks - HEADER_BITS)) // 8
        hdr = flat[:3 * HEADER_BITS].reshape(HEADER_BITS, 3)
        hdr_flat = np.zeros(64, dtype=np.uint8)
        hdr_flat[:HEADER_BITS] = hdr.any(axis=1).astype(np.uint8)
        header8 = (hdr_flat.reshape(-1, 8) * WEIGHTS).sum(axis=1,
                                                          dtype=np.uint8).tobytes()
        pb = flat[3 * HEADER_BITS: 3 * HEADER_BITS + 8 * n_payload_bytes]
        payload = (pb.reshape(-1, 8) * WEIGHTS).sum(axis=1,
                                                    dtype=np.uint8).tobytes()
        return header8, payload
    except Exception:
        return None, None


def verify_group(header8, payload):
    """Check magic/seq/crc. Returns (seq, ok)."""
    g, h_ok = unpack_header(header8)
    if not h_ok or g is None:
        return None, False
    ok = crc_for_group(g, payload) == int.from_bytes(header8[3:7], 'big')
    return g, ok


def decode_group_fast(frame, m, width, height):
    """Single-frame decode for a LOSSLESS transport (QUIC: records arrive
    byte-exact). render_group draws every MxM block uniformly 0/255, so one
    centre sample per block is an exact inverse — no R-copy averaging, no
    int64 block sums (~60x faster than decode_group at 4K: ~2 ms vs 130 ms).

    Returns (header8, payload) or (None, None) like decode_group."""
    try:
        arr = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3)
        s = arr[m // 2::m, m // 2::m]            # (by, bx, 3) block centres
        flat = (s >= 128).reshape(-1).astype(np.uint8)
        n_blocks = s.size // 3
        n_payload_bytes = (3 * (n_blocks - HEADER_BITS)) // 8
        hdr = flat[:3 * HEADER_BITS].reshape(HEADER_BITS, 3)
        hdr_flat = np.zeros(64, dtype=np.uint8)
        hdr_flat[:HEADER_BITS] = hdr.any(axis=1).astype(np.uint8)
        header8 = (hdr_flat.reshape(-1, 8) * WEIGHTS).sum(axis=1,
                                                          dtype=np.uint8).tobytes()
        pb = flat[3 * HEADER_BITS: 3 * HEADER_BITS + 8 * n_payload_bytes]
        payload = (pb.reshape(-1, 8) * WEIGHTS).sum(axis=1,
                                                    dtype=np.uint8).tobytes()
        return header8, payload
    except Exception:
        return None, None


def decode_group_grid(frame, blocks_x, blocks_y):
    """Decode a COMPACT grid frame (blocks_y, blocks_x, 3 of 0/255) to
    (header8, payload) or (None, None). Each pixel is one block, so no
    center-sampling is needed — threshold the pixels directly. This is the
    inverse of render_group(..., grid=True): the pipe ships the 390 KB grid
    (4K) instead of the 24.9 MB frame, ffmpeg upscales before encode and
    downscales (format=rgb24 then scale=neighbor) before we get here.
    Byte-exact through x264 CRF23 (see /tmp/grid_gate.py)."""
    try:
        n = blocks_x * blocks_y * 3
        if len(frame) < n:
            return None, None
        flat = (np.frombuffer(frame[:n], dtype=np.uint8) >= 128).reshape(
            blocks_y, blocks_x, 3).reshape(-1).astype(np.uint8)
        n_blocks = blocks_x * blocks_y
        n_payload_bytes = (3 * (n_blocks - HEADER_BITS)) // 8
        hdr = flat[:3 * HEADER_BITS].reshape(HEADER_BITS, 3)
        hdr_flat = np.zeros(64, dtype=np.uint8)
        hdr_flat[:HEADER_BITS] = hdr.any(axis=1).astype(np.uint8)
        header8 = (hdr_flat.reshape(-1, 8) * WEIGHTS).sum(axis=1,
                                                          dtype=np.uint8).tobytes()
        pb = flat[3 * HEADER_BITS: 3 * HEADER_BITS + 8 * n_payload_bytes]
        payload = (pb.reshape(-1, 8) * WEIGHTS).sum(axis=1,
                                                    dtype=np.uint8).tobytes()
        return header8, payload
    except Exception:
        return None, None


def parity_rows(data_rows, P):
    """data_rows: (k, p) uint8 arrays -> list of (m, p) parity arrays."""
    return [row for row in fec_encode(data_rows, P)]


def stripe_data(received, k, P):
    """received: length k+m list of (p,) arrays or None. Returns (k, p)."""
    return fec_decode(list(received), k, P)


def make_eof_group(width, height, m=M):
    """EOF sentinel: header seq=0xFFFF, payload b'EOF!' + zeros (CRC over
    seq+payload exactly as pack_header does, so the receiver verifies it
    like any other group)."""
    B = group_capacity(width, height, m)
    payload = EOF_PAYLOAD + b'\x00' * (B - len(EOF_PAYLOAD))
    return render_group(payload, 0xFFFF, m, width, height)


def stripe_of(seq, k=K, m=M_PAR):
    return seq // (k + m)


class SeqWrapTracker:
    """Reconstruct true group indices from the 16-bit header seq.

    The header stores seq = g & 0xFFFF, so the field wraps every 65536
    groups. The stream is strictly increasing in TRUE index, so in stream
    order an observed seq can only jump FORWARD (a gap) or FALL (a genuine
    wrap past 2**16). Each fall increments the wrap count;
    true index = wraps * 2**16 + seq. A long-lived tunnel needs this:
    at ~50 groups/s the raw seq would wrap every ~22 minutes otherwise.
    """

    SEQ_MOD = 1 << 16

    def __init__(self):
        self._last = None
        self._wraps = 0

    def true_index(self, seq):
        if self._last is not None and seq < self._last:
            self._wraps += 1
        self._last = seq
        return self._wraps * self.SEQ_MOD + seq
