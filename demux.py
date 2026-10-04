"""Demuxer: reassembles the continuous mux byte stream and parses packets
back into (conn, data, fin) tuples. The stream is a concatenation of group
payloads in order; packets may span group boundaries, so the demuxer keeps
its own buffer and is fed payload-by-payload in stream order."""

from mux import FLAG_FIN


class Demuxer:
    def __init__(self):
        self.buf = bytearray()

    def feed(self, payload):
        """Append one group payload; yields (conn, data, fin) packets."""
        self.buf += payload
        out = []
        while True:
            if len(self.buf) < 4:
                break
            conn = self.buf[0]
            ln = int.from_bytes(self.buf[1:3], 'big')
            flags = self.buf[3]
            if conn == 0:
                # idle filler: len is normally 0; skip the whole idle packet
                if ln:
                    ln = 0
                if len(self.buf) < 4 + ln:
                    break
                del self.buf[:4 + ln]
                continue
            if len(self.buf) < 4 + ln:
                break
            data = bytes(self.buf[4:4 + ln])
            del self.buf[:4 + ln]
            out.append((conn, data, bool(flags & FLAG_FIN)))
        return out
