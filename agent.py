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

import numpy as np
class QLearningVHDA:
    def __init__(self, state_size, action_size, alpha=0.3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.02):


        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha          # learning rate
        self.gamma = gamma          # discount factor
        self.epsilon = epsilon      # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        state_tup = tuple(np.round(state, 2))  # discretize state

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_vals = [self.get_q_value(state_tup, a) for a in range(self.action_size)] # exploration path, the agent itself.
            return int(np.argmax(q_vals))

    def update_q_value(self, state, action, reward, next_state):
        state_tup = tuple(np.round(state, 2))
        next_state_tup = tuple(np.round(next_state, 2))
        old_q = self.get_q_value(state_tup, action)
        next_max = max([self.get_q_value(next_state_tup, a) for a in range(self.action_size)])

        # update rule, from wikipedia
        # check for "qlearning.PNG"
        new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
        self.q_table[(state_tup, action)] = new_q

    def decay_epsilon(self):
        # better epsilon decay function
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
