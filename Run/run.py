from env import VHDAEnv
import matplotlib.pyplot as plt
import numpy as np
from agent import QLearningVHDA

def main(episodes=500):

    env = VHDAEnv()
    agent = QLearningVHDA(state_size=8, action_size=2)
    episode_rewards = []
    avg = []
    unho = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:

            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            total_reward += reward


        stats = env.get_qos_stats()
        print(f"\nEpisode {ep + 1}")
        print("QoS improvements:", stats["QoS Improvements"])
        print("unnecessary handovers:", stats["Unnecessary Handovers"])
        print("avg QoS Score:", stats["Average QoS"])
        print(f"Total Reward: {total_reward:.2f}")
        avg.append(stats["Average QoS"])
        episode_rewards.append(total_reward)
        unho.append(stats["Unnecessary Handovers"])
        agent.decay_epsilon()

    # visualization

    plt.plot(episode_rewards, marker='o')
    plt.title(f"Total Rewards Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.plot(avg, marker='o')
    plt.title(f"Average QoS over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Average QoS")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.plot(unho, marker='o')
    plt.title(f"Unnecessary handovers over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Unnecessary handovers")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

main(episodes=500)
