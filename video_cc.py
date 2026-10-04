"""Congestion control for the deterministic 7030+FEC video carrier.

The video channel is paced at a fixed frame rate (one frame every 1/fps),
its capacity is fixed, and "loss" is handled by FEC (video-dropped frames)
and QUIC retransmission. In that world the classic Reno/CUBIC story —
"the path is congested, shrink the window" — is wrong: the path is never
congested, it is just slow and deterministic. Measured symptom: Reno pins
cwnd at its 2-datagram minimum against the stripe RTT (~2.4 s), which is
~1 KB/s of a 78 KB/s pipe.

The right behaviour for a trusted, fixed-capacity, loss-tolerant pipe is a
constant window sized to the bandwidth-delay product (BDP = capacity x
RTT). That keeps the pipe exactly full: no more (which would only create
duplicates/drops that FEC and retransmission then chew through) and no
less (which starves it).

The window is NOT a fixed constant: it must scale with the carrier, because
capacity = B*fps/R bytes/s per direction and BDP = capacity x RTT. Doubling
fps (30 -> 60) doubles the BDP, and a stale 1 MB window then starves the
high-capacity carriers (measured at 4K: 1 MB -> 697 KB/s, 4 MB -> 828 KB/s).
The node computes the BDP from its own geometry/fps and calls set_window()
before the QUIC connection is created; VQIC_CWND_BYTES overrides it
explicitly. 1 MB is the floor (small carriers are already saturated by it).
"""
import os

from aioquic.quic.congestion.base import (QuicCongestionControl,
                                          register_congestion_control)

# 0 = not set yet (the node calls set_window with its BDP before connect).
# VQIC_CWND_BYTES, if set, wins outright (explicit override).
_WINDOW = int(os.environ.get("VQIC_CWND_BYTES", "0"))
_FLOOR = 1 << 20


def set_window(n):
    """Size the constant window (bytes). Called by the node with its BDP."""
    global _WINDOW
    if int(os.environ.get("VQIC_CWND_BYTES", "0")) == 0:
        _WINDOW = int(n)


def current_window():
    return _WINDOW or _FLOOR


class VideoTunnelCC(QuicCongestionControl):
    """Constant window sized to the video pipe's BDP; never shrinks, never
    grows. The size comes from set_window() (the node's BDP) or the
    VQIC_CWND_BYTES override, floored at 1 MB."""

    def __init__(self, *, max_datagram_size: int):
        super().__init__(max_datagram_size=max_datagram_size)
        self.congestion_window = current_window()
        self.ssthresh = None

    def on_packet_acked(self, *, now, packet):
        self.bytes_in_flight -= packet.sent_bytes
        self.congestion_window = current_window()

    def on_packet_sent(self, *, packet):
        self.bytes_in_flight += packet.sent_bytes

    def on_packets_expired(self, *, packets):
        for p in packets:
            self.bytes_in_flight -= p.sent_bytes
        self.congestion_window = current_window()

    def on_packets_lost(self, *, now, packets):
        for p in packets:
            self.bytes_in_flight -= p.sent_bytes
        # Do not shrink: on this carrier a "lost" packet is usually one
        # waiting out its FEC stripe, not a victim of congestion.
        self.congestion_window = current_window()

    def on_rtt_measurement(self, *, now, rtt):
        pass


register_congestion_control("video-tunnel", VideoTunnelCC)
