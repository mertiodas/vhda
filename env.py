import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class VHDAEnv(gym.Env):
    def __init__(self, csv_path="vhda_synthetic_normalized.csv", handover_penalty=0.35, throughput_weight=0.7, latency_weight=0.3):
        super(VHDAEnv, self).__init__()
        self.data = pd.read_csv(csv_path)
        # Radio + QoS features in observation
        self.radio_features = self.data[['RSSI_WIFI', 'SINR_WIFI', 'RSRP_5G', 'SINR_5G']].values
        self.qos_features = self.data[['throughput_WIFI', 'latency_WIFI', 'throughput_5G', 'latency_5G']].values
        self.current_step = 0
        self.last_action = None
        self.handover_penalty = handover_penalty
        self.last_qos_score = 0
        self.throughput_weight = throughput_weight
        self.latency_weight = latency_weight

        # Obs: radio + QoS values (normalized 0-1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0: WiFi, 1: 5G

        self.qos_improvement_count = 0
        self.unnecessary_handover_count = 0
        self.qos_history = []

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.last_action = None
        self.last_qos_score = 0
        self.qos_improvement_count = 0
        self.unnecessary_handover_count = 0
        self.qos_history = []
        # concat radio + QoS for initial state
        obs = np.concatenate((self.radio_features[self.current_step], self.qos_features[self.current_step]))
        return obs, {}

    def step(self, action):
        qos = self.qos_features[self.current_step]
        radio = self.radio_features[self.current_step]

        # select RAT
        if action == 0:  # WiFi
            throughput, latency = qos[0], qos[1]
        else:  # 5G
            throughput, latency = qos[2], qos[3]

        throughput_ok = throughput >= 0.5
        latency_ok = latency <= 0.5

        # weighted QoS score
        qos_score = self.throughput_weight * throughput - self.latency_weight * latency
        reward = 0

        switched = self.last_action is not None and self.last_action != action
        IMPROVEMENT_THRESHOLD = 0.05
        if switched:
            delta = qos_score - self.last_qos_score
            if delta > IMPROVEMENT_THRESHOLD:
                reward += delta
                self.qos_improvement_count += 1
            else:
                reward -= self.handover_penalty
                self.unnecessary_handover_count += 1

        if throughput_ok and latency_ok:
            reward += qos_score

        self.last_qos_score = qos_score
        self.last_action = action
        self.current_step += 1
        self.qos_history.append(qos_score)

        done = self.current_step >= len(self.radio_features)
        if not done:
            obs = np.concatenate((self.radio_features[self.current_step], self.qos_features[self.current_step]))
        else:
            obs = np.zeros_like(np.concatenate((self.radio_features[0], self.qos_features[0])))

        return obs, reward, done, False, {}

    def render(self):
        pass

    def get_qos_stats(self):
        return {
            "QoS Improvements": self.qos_improvement_count,
            "Unnecessary Handovers": self.unnecessary_handover_count,
            "Average QoS": np.mean(self.qos_history),
            "QoS History": self.qos_history
        }