import numpy as np


def evaluate(policy, env):
    """Runs ``policy`` once over every trace of ``env``; returns the totals and per-trace means.

    ``policy`` is a function (env, obs) -> action. Nothing is learned here: an agent is
    passed as ``lambda env, obs: agent.choose_action(obs, greedy=True)``.
    """
    rewards, qos, handovers, unnecessary, steps = [], [], [], [], 0
    for i in range(len(env.traces)):
        obs, _ = env.reset(options={"trace": i})
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy(env, obs))
            done = terminated or truncated
            total += reward
        stats = env.get_qos_stats()
        rewards.append(total)
        qos.append(stats["Average QoS"])
        handovers.append(stats["Handovers"])
        unnecessary.append(stats["Unnecessary Handovers"])
        steps += len(stats["QoS History"])
    per_1000 = 1000 / steps
    return {
        "reward": float(np.sum(rewards)),
        "avg_qos": float(np.mean(qos)),
        "handovers_per_1000": float(np.sum(handovers) * per_1000),
        "unnecessary_per_1000": float(np.sum(unnecessary) * per_1000),
    }
