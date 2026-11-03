"""Mobility traces: a user walking in and around a building, measured once a second.

The WiFi access point is inside the building; the 5G cell is a few hundred metres away.
Signals change slowly as the user moves (path loss + spatially correlated shadowing) and
the load of each network drifts over time, so consecutive samples depend on each other,
as they do in a real handover problem.

Every trace is one walk. Train and test use different walks, so an agent cannot pass the
test by remembering samples.
"""

import numpy as np
import pandas as pd

COLUMNS = ['RSSI_WIFI', 'SINR_WIFI', 'throughput_WIFI', 'latency_WIFI',
           'RSRP_5G', 'SINR_5G', 'throughput_5G', 'latency_5G']

# fixed physical ranges, so every trace is normalised the same way
RANGES = {
    'RSSI_WIFI': (-95, -30),      # dBm
    'SINR_WIFI': (-5, 40),        # dB
    'throughput_WIFI': (0, 300),  # Mbit/s
    'latency_WIFI': (0, 150),     # ms
    'RSRP_5G': (-130, -70),       # dBm
    'SINR_5G': (-5, 30),          # dB
    'throughput_5G': (0, 300),    # Mbit/s
    'latency_5G': (0, 150),       # ms
}

AREA = (100.0, 60.0)                   # metres, where the user walks
BUILDING = (0.0, 0.0, 60.0, 40.0)      # x0, y0, x1, y1: WiFi coverage is inside
AP = (30.0, 20.0)                      # WiFi access point
GNB = (420.0, 260.0)                   # 5G cell, outside the walking area
NOISE_DBM = -95.0


def _inside(x, y):
    x0, y0, x1, y1 = BUILDING
    return (x0 <= x) & (x <= x1) & (y0 <= y) & (y <= y1)


def _walk(rng, steps):
    """Random waypoint: walk to a random point at 0.8-1.6 m/s, sometimes stop for a while."""
    pos = np.empty((steps, 2))
    p = rng.uniform((0, 0), AREA)
    target, speed, pause = rng.uniform((0, 0), AREA), rng.uniform(0.8, 1.6), 0
    for t in range(steps):
        if pause > 0:
            pause -= 1
        else:
            d = target - p
            dist = np.hypot(*d)
            if dist < speed:
                p = target
                target, speed = rng.uniform((0, 0), AREA), rng.uniform(0.8, 1.6)
                pause = rng.integers(0, 30) if rng.random() < 0.4 else 0
            else:
                p = p + d / dist * speed
        pos[t] = p
    return pos


def _shadowing(rng, pos, sigma, decorrelation_m):
    """Log-normal shadowing that is correlated over distance (Gudmundson model)."""
    moved = np.r_[0.0, np.hypot(*np.diff(pos, axis=0).T)]
    s = np.empty(len(pos))
    s[0] = rng.normal(0, sigma)
    for t in range(1, len(pos)):
        rho = np.exp(-moved[t] / decorrelation_m)
        s[t] = rho * s[t - 1] + np.sqrt(1 - rho ** 2) * rng.normal(0, sigma)
    return s


def _load(rng, steps, mean, sigma, keep=0.995):
    """Share of the network others are using: a slow AR(1) drift between 0 and 0.95."""
    x = np.empty(steps)
    x[0] = mean
    for t in range(1, steps):
        x[t] = mean + keep * (x[t - 1] - mean) + rng.normal(0, sigma)
    return np.clip(x, 0.0, 0.95)


def generate_trace(seed, steps=2000):
    """One walk as raw (physical) values, one row per second."""
    rng = np.random.default_rng(seed)
    pos = _walk(rng, steps)
    x, y = pos.T
    inside = _inside(x, y)

    # WiFi: log-distance path loss, an extra wall loss outside the building
    d_ap = np.maximum(np.hypot(x - AP[0], y - AP[1]), 1.0)
    rssi = (-30 - 10 * 2.8 * np.log10(d_ap) - np.where(inside, 0.0, 12.0)
            + _shadowing(rng, pos, 4.0, 5.0) + rng.normal(0, 1.0, steps))
    wifi_load = _load(rng, steps, rng.uniform(0.2, 0.6), 0.02)
    wifi_interference = 3 + 10 * wifi_load                       # dB above noise
    sinr_wifi = rssi - NOISE_DBM - wifi_interference

    # 5G: far cell, slower shadowing, a building penetration loss inside
    d_gnb = np.hypot(x - GNB[0], y - GNB[1])
    rsrp = (-10 - 10 * 3.2 * np.log10(d_gnb) - np.where(inside, 10.0, 0.0)
            + _shadowing(rng, pos, 6.0, 20.0) + rng.normal(0, 1.0, steps))
    nr_load = _load(rng, steps, rng.uniform(0.2, 0.7), 0.015)
    sinr_5g = rsrp - (NOISE_DBM - 22) - 6 - 8 * nr_load          # per-resource-element noise

    def throughput(sinr_db, bandwidth_mhz, efficiency, load, cap):
        shannon = bandwidth_mhz * efficiency * np.log2(1 + 10 ** (np.clip(sinr_db, -10, 40) / 10))
        return np.clip(shannon * (1 - load), 0, cap)

    thr_wifi = throughput(sinr_wifi, 40, 0.6, wifi_load, 300)
    thr_5g = throughput(sinr_5g, 40, 0.5, nr_load, 300)

    def latency(base, sinr_db, load):
        # retransmissions at low SINR, queueing as the network fills up, some jitter
        retx = 40 / (1 + np.exp((sinr_db - 5) / 2))
        queue = 8 * load / (1 - load)
        return np.clip(base + retx + queue + rng.gamma(2.0, 1.0, steps), 0, None)

    lat_wifi = latency(4, sinr_wifi, wifi_load)
    lat_5g = latency(12, sinr_5g, nr_load)

    return pd.DataFrame({
        'RSSI_WIFI': rssi, 'SINR_WIFI': sinr_wifi,
        'throughput_WIFI': thr_wifi, 'latency_WIFI': lat_wifi,
        'RSRP_5G': rsrp, 'SINR_5G': sinr_5g,
        'throughput_5G': thr_5g, 'latency_5G': lat_5g,
    })


def normalise(raw):
    """Physical values to 0..1 with the fixed ranges above."""
    out = pd.DataFrame(index=raw.index)
    for col in COLUMNS:
        lo, hi = RANGES[col]
        out[col] = ((raw[col] - lo) / (hi - lo)).clip(0, 1)
    return out


def make_traces(seeds, steps=2000):
    """Normalised traces, one per seed."""
    return [normalise(generate_trace(s, steps)) for s in seeds]
