"""Byte-stream muxer: packs per-connection data into fixed-size group payloads.

Packet layout (one per frame, packed back-to-back):
  conn (1B) | len (2B big-endian) | flags (1B) | data (len bytes)
flags bit0 = FIN (connection finished). conn 0 = idle filler (len 0).
Every group payload is exactly B bytes (B = group_capacity of the geometry),
so the muxer is stateless across groups — no packet ever spans two groups.
"""

MAX_CONN = 254
FLAG_FIN = 0x01
PKT_MAX = 65535       # max data bytes per single packet (2-byte len field)


def pack_packet(conn, data, fin=False):
    """One packet: header (4B) + data. conn must be 1..MAX_CONN."""
    if not 1 <= conn <= MAX_CONN:
        raise ValueError(f"conn id out of range: {conn}")
    if len(data) > PKT_MAX:
        raise ValueError("packet data exceeds 65535 bytes")
    flags = FLAG_FIN if fin else 0
    return bytes([conn]) + len(data).to_bytes(2, 'big') + bytes([flags]) + data


class Muxer:
    """Collects (conn, data, fin) chunks and yields full B-byte payloads.

    `feed()` appends chunks (splitting long ones into <=65535-byte packets);
    `next_group()` pops exactly B bytes (call when pending() >= B);
    `flush_partial()` pads the current partial group with idle packets and
    yields it (used to keep the video moving on stalls and on shutdown)."""

    def __init__(self, group_bytes):
        self.group_bytes = group_bytes
        self.buf = bytearray()
        # hard cap on the packet buffer: a fast client can never pile up
        # unbounded data (8 full groups ~= a couple seconds of stream).
        self.cap = 8 * group_bytes
        self.holdback = {}          # conn -> bytes not yet admitted
        self.hold_fin = set()       # conns with a pending FIN

    def backpressure(self, conn=0):
        """Per-conn: True while this connection has held-back bytes (stop
        reading from its socket until they drain). conn=0: global space."""
        if conn:
            return len(self.holdback.get(conn, b'')) > 0
        return len(self.buf) >= self.cap

    def admitted(self, conn):
        """How many held-back bytes may now move into the buffer."""
        free = self.cap - len(self.buf)
        return max(0, min(len(self.holdback.get(conn, b'')), free))

    def admit(self, conn, nbytes):
        """Move up to nbytes from holdback into the packet buffer. When the
        holdback fully drains and a FIN was pending, the FIN packet is
        emitted here (after the last data packet) and the connection is
        cleaned up so it can't stall the flush logic."""
        if nbytes <= 0:
            return
        held = self.holdback[conn]
        take = held[:nbytes]
        left = held[nbytes:]
        self._pack_into(conn, take, fin=(not left and conn in self.hold_fin))
        if not left:
            # holdback fully drained for this conn
            self.holdback.pop(conn, None)
            self.hold_fin.discard(conn)
        else:
            self.holdback[conn] = left

    def feed(self, conn, data, fin=False):
        if fin:
            self.hold_fin.add(conn)
        # Empty FIN while data is still held back: remember it but do NOT
        # emit the FIN packet yet — it must arrive AFTER the held bytes,
        # otherwise the receiver closes the connection early and drops the
        # tail. `admit()` emits the FIN once holdback is fully drained.
        if not data and self.holdback.get(conn):
            return
        if len(self.buf) + len(data) <= self.cap:
            self._pack_into(conn, data, fin=fin)
            if fin:
                self.hold_fin.discard(conn)
            return
        # over the cap: pack what fits into the buffer, hold the rest
        fit = self.cap - len(self.buf)
        if fit > 0:
            self._pack_into(conn, data[:fit], fin=False)
            rest = data[fit:]
        else:
            rest = data
        if rest:
            self.holdback[conn] = self.holdback.get(conn, b'') + rest
        # fin with data still held: remembered via hold_fin

    def _pack_into(self, conn, data, fin=False):
        off = 0
        while off < len(data):
            chunk = data[off:off + PKT_MAX]
            off += len(chunk)
            self.buf += pack_packet(conn, chunk,
                                    fin=(fin and off >= len(data)))
        if fin and not data:
            self.buf += pack_packet(conn, b'', fin=True)

    def pending(self):
        return len(self.buf)

    def next_group(self):
        """Pop exactly group_bytes (the caller guarantees pending() >= it)."""
        out = bytes(self.buf[:self.group_bytes])
        del self.buf[:self.group_bytes]
        return out

    def flush_partial(self):
        """Emit the current partial group: ALL bytes already in the buffer
        (complete packets and/or the tail of a cut packet), then idle zero
        padding. The zero padding may only ever follow the real bytes — the
        demux's byte stream is the concatenation of the group payloads, and
        zeros must land at the end of the real data (they read as idle
        packets). Holding the partial tail for a later group would let the
        zeros take its place mid-packet and corrupt the demux."""
        n = len(self.buf)
        if n > self.group_bytes:
            raise ValueError("buffer overflows one group")
        out = bytes(self.buf) + b'\x00' * (self.group_bytes - n)
        self.buf.clear()
        return out

    def clear(self):
        self.buf.clear()

    def reset(self):
        """Drop ALL pending data (buffer, holdback, pending FINs). Used when
        the transport link dies and the sender must start the next session
        from a clean slate (in-flight data is lost with the link)."""
        self.buf.clear()
        self.holdback.clear()
        self.hold_fin.clear()
