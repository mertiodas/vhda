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

    window_rewards = []
    window_qos = []
    window_unho = []

    first_episode_stats = None
    last_episode_stats = None

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

        episode_rewards.append(total_reward)
        avg.append(stats["Average QoS"])
        unho.append(stats["Unnecessary Handovers"])

        window_rewards.append(total_reward)
        window_qos.append(stats["Average QoS"])
        window_unho.append(stats["Unnecessary Handovers"])

        if ep == 0:
            first_episode_stats = (total_reward,
                                   stats["Average QoS"],
                                   stats["Unnecessary Handovers"])

        if (ep + 1) % 10 == 0:
            print(f"\nEpisodes {ep - 8}–{ep + 1}")
            print(f"Avg Reward: {np.mean(window_rewards):.3f}")
            print(f"Avg QoS: {np.mean(window_qos):.3f}")
            print(f"Avg Unnecessary Handovers per Episode: {np.mean(window_unho):.1f}")


            window_rewards.clear()
            window_qos.clear()
            window_unho.clear()

        agent.decay_epsilon()

    last_episode_stats = (episode_rewards[-1], avg[-1], unho[-1])

    print("\n====== FIRST vs LAST EPISODE COMPARISON ======")
    print(f"Reward: {first_episode_stats[0]:.3f}  →  {last_episode_stats[0]:.3f}")
    print(f"Avg QoS: {first_episode_stats[1]:.3f}  →  {last_episode_stats[1]:.3f}")
    print(f"Unnecessary Handovers: {first_episode_stats[2]}  →  {last_episode_stats[2]}")

    # visualization

    plt.plot(episode_rewards, marker='o')
    plt.title("Total Rewards Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.plot(avg, marker='o')
    plt.title("Average QoS over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Average QoS")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.plot(unho, marker='o')
    plt.title("Unnecessary handovers over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Unnecessary handovers")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

main()
