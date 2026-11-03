        # alpha controls learning speed, (0.2 or higher = agent learns faster from new experiences)
        # 0.1 or lower, relies more on past knowledge making it more stable.


        # gamma controls focus on future rewards
        # closer to 1, agent values long-term rewards more, future pay off is more important
        # closer to 0, agent focuses immediate rewards only, future is uncertain, rewards are short term.


        # epsilon, epsilon decay, epsilon min control the exploration balance over time
        # epsilon, closer to 1, agent explores more at the start, random actions
        # epsilon, closer to 0, agent exploits known best actions more from the beginning
        # can speed up convergence if initial knowledge is good, but risks missing better solutions

        # epsilon decay, increase decay rate, exploration lasts longer, better for more complex tasks
        # epsilon decay, decrease decay rate, exploration ends quicker, may speed up learning risks converging

        # epsilon min, 0.1 = agent always explores at least 10% of the time
        # helps avoid getting stuck in local optima but slows exploitation

        # Higher alpha, faster but noisier learning.
        # Higher gamma, more foresight, longer-term rewards.
        # Higher epsilon or slower decay, more exploration, less early exploitation.
        # Higher epsilon_min, always keep some exploration alive.

        # for 100 episodes, alpha=0.6, gamma=0.95, epsilon_decay=0.98, epsilon_min=0.05, handover penalty = 0.3
        # for 500 episodes, alpha=0.3, gamma=0.99, epsilon_decay=0.995, epsilon_min=0.02, handover penalty = 0.35
        # for 1000 episdes, alpha=0.15, gamma=0.99, epsilon_decay=0.999, epsilon_min=0.02, handover penalty = 0.2

# notes above are from the first version (its reward had a handover_penalty; now handover_cost in env.py)

import numpy as np

from env import LATENCY_WEIGHT, THROUGHPUT_WEIGHT, qos_scores


def compact_state(obs, bins=(-0.3, -0.15, -0.08, -0.03, 0.0, 0.03, 0.08, 0.15, 0.3),
                  weights=(THROUGHPUT_WEIGHT, LATENCY_WEIGHT)):
    """The few things the decision depends on, as a small discrete state.

    The raw observation has 8 continuous values; rounded to 2 decimals almost every sample is a
    state of its own, and a Q-table can only memorise them. Here the state is how much better
    5G is than WiFi right now (in 10 bins) and the RAT the user is on: 20 states, each seen many
    times. ``weights`` must be those of the environment's reward.
    """
    wifi, nr = qos_scores(obs[:8], *weights)
    rat = int(obs[8])
    return (int(np.digitize(nr - wifi, bins)), rat)


def rounded_state(obs, decimals=2):
    """The first version's state: the 8 measurements rounded (it had no serving RAT).
    Kept to show why it does not generalise."""
    return tuple(np.round(obs[:8], decimals))


class QLearningVHDA:
    def __init__(self, state_size, action_size, alpha=0.3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.02, alpha_decay=1.0, alpha_min=0.005, state_fn=compact_state, seed=None):

        self.state_size = state_size  # kept for the old call signature; the table grows as states are seen
        self.action_size = action_size
        self.alpha = alpha          # learning rate
        self.gamma = gamma          # discount factor
        self.epsilon = epsilon      # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.alpha_decay = alpha_decay  # 1.0 = constant learning rate
        self.alpha_min = alpha_min
        self.state_fn = state_fn    # observation -> hashable state
        self.rng = np.random.default_rng(seed)

        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, obs, greedy=False):
        state = self.state_fn(obs)
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_size))
        q_vals = [self.get_q_value(state, a) for a in range(self.action_size)]
        return int(np.argmax(q_vals))

    def update_q_value(self, obs, action, reward, next_obs, terminated):
        state = self.state_fn(obs)
        old_q = self.get_q_value(state, action)
        if terminated:
            # the task is over: nothing follows, the target is the reward alone
            # (the end of a trace is a truncation, not this)
            target = reward
        else:
            next_state = self.state_fn(next_obs)
            target = reward + self.gamma * max(self.get_q_value(next_state, a) for a in range(self.action_size))

        # update rule, from wikipedia
        # check for "qlearning.PNG"
        self.q_table[(state, action)] = old_q + self.alpha * (target - old_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def end_episode(self):
        # a constant alpha keeps rarely visited states jumping long after the policy is learned;
        # lowering it with the episodes lets the values settle
        self.decay_epsilon()
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
