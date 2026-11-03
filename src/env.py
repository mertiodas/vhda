import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import os

from traces import COLUMNS

WIFI, NR = 0, 1  # actions and serving RAT: 0 = WiFi, 1 = 5G
THROUGHPUT_WEIGHT, LATENCY_WEIGHT = 0.7, 0.3


def qos_scores(m, tw=THROUGHPUT_WEIGHT, lw=LATENCY_WEIGHT):
    """QoS score of WiFi and of 5G for each row of measurements (columns in traces.COLUMNS order)."""
    m = np.asarray(m)
    return np.stack([tw * m[..., 2] - lw * m[..., 3], tw * m[..., 6] - lw * m[..., 7]], axis=-1)


class VHDAEnv(gym.Env):
    """Vertical handover between WiFi and 5G.

    Observation: the 8 normalised measurements of the current second plus the RAT the user
    is on now (9 values). Action: the RAT to use for this second.

    Reward per step: QoS score of the chosen RAT, minus a fixed cost when it differs from the
    RAT the user was on. The cost stands for signalling and the short break in service; it is
    what makes switching on every small fluctuation a bad idea.

    Data: a list of traces (DataFrames with the 8 columns), one per episode; or a single CSV
    (the old dataset), optionally cut into a train or test part.
    """

    def __init__(self,
                 traces=None,
                 csv_path=None,
                 handover_cost=0.1,
                 throughput_weight=THROUGHPUT_WEIGHT,
                 latency_weight=LATENCY_WEIGHT,
                 improvement_threshold=0.05,
                 initial_rat=WIFI,
                 split="all",
                 train_ratio=0.8):

        super().__init__()

        if traces is None:
            if csv_path is None:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                csv_path = os.path.join(base_dir, "Datasets", "vhda_synthetic_normalized.csv")
            data = pd.read_csv(csv_path)
            # one long series: split by position, never shuffle
            cut = int(len(data) * train_ratio)
            if split == "train":
                data = data.iloc[:cut]
            elif split == "test":
                data = data.iloc[cut:]
            elif split != "all":
                raise ValueError(f"split must be 'train', 'test' or 'all', got {split!r}")
            traces = [data]

        # per trace: (T, 8) measurements and (T, 2) QoS score of each RAT
        self.traces = [t[COLUMNS].to_numpy(dtype=np.float32) for t in traces]
        self.scores = [qos_scores(t, throughput_weight, latency_weight) for t in self.traces]

        self.handover_cost = handover_cost
        self.improvement_threshold = improvement_threshold
        self.initial_rat = initial_rat

        self.observation_space = spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.trace_index = -1
        self.reset()
        self.trace_index = -1  # the first reset() of a run starts on trace 0

    def _obs(self):
        return np.append(self.measurements[self.t], np.float32(self.rat))

    def reset(self, seed=None, options=None):
        """Starts the next trace, or the one given as ``options={"trace": i}``."""
        super().reset(seed=seed)
        if options and "trace" in options:
            self.trace_index = options["trace"]
        else:
            self.trace_index = (self.trace_index + 1) % len(self.traces)
        self.measurements = self.traces[self.trace_index]
        self.score = self.scores[self.trace_index]
        self.t = 0
        self.rat = self.initial_rat
        self.handovers = 0
        self.unnecessary_handovers = 0
        self.qos_history = []
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        qos = self.score[self.t, action]
        reward = qos

        if action != self.rat:
            reward -= self.handover_cost
            self.handovers += 1
            # unnecessary: right now the new RAT is not clearly better than the one left
            if qos - self.score[self.t, self.rat] <= self.improvement_threshold:
                self.unnecessary_handovers += 1

        self.rat = action
        self.qos_history.append(qos)
        self.t += 1

        # the walk ends because the trace does, not because the task is over: that is a
        # truncation, and a learner should still count on a future after it. The last
        # measurement stands in for the next one, which was not recorded.
        truncated = self.t >= len(self.measurements)
        obs = np.append(self.measurements[-1], np.float32(self.rat)) if truncated else self._obs()
        return obs, float(reward), False, truncated, {}

    def render(self):
        pass

    def get_qos_stats(self):
        return {
            "Handovers": self.handovers,
            "Unnecessary Handovers": self.unnecessary_handovers,
            "Average QoS": float(np.mean(self.qos_history)) if self.qos_history else 0.0,
            "QoS History": self.qos_history,
        }
