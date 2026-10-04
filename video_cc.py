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

Tune the window with VQIC_CWND_BYTES if the carrier's resolution/fps
changes the BDP. The window must be >= capacity x RTT, where capacity =
B*fps/R per direction. 1 MB fills 1080p (BDP ~354 KB) with margin and
caps at the video's own ~177 KB/s; smaller resolutions are already
saturated by a smaller window.
"""
import os

from aioquic.quic.congestion.base import (QuicCongestionControl,
                                          register_congestion_control)

WINDOW = int(os.environ.get("VQIC_CWND_BYTES", str(1 << 20)))


class VideoTunnelCC(QuicCongestionControl):
    """Constant window sized to the video pipe's BDP; never shrinks, never
    grows."""

    def __init__(self, *, max_datagram_size: int):
        super().__init__(max_datagram_size=max_datagram_size)
        self.congestion_window = WINDOW
        self.ssthresh = None

    def on_packet_acked(self, *, now, packet):
        self.bytes_in_flight -= packet.sent_bytes
        self.congestion_window = WINDOW

    def on_packet_sent(self, *, packet):
        self.bytes_in_flight += packet.sent_bytes

    def on_packets_expired(self, *, packets):
        for p in packets:
            self.bytes_in_flight -= p.sent_bytes
        self.congestion_window = WINDOW

    def on_packets_lost(self, *, now, packets):
        for p in packets:
            self.bytes_in_flight -= p.sent_bytes
        # Do not shrink: on this carrier a "lost" packet is usually one
        # waiting out its FEC stripe, not a victim of congestion.
        self.congestion_window = WINDOW

    def on_rtt_measurement(self, *, now, rtt):
        pass


register_congestion_control("video-tunnel", VideoTunnelCC)
