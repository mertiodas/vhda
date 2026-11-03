"""Trains the Q-learning agent on some walks and compares every policy on walks it never saw.

    python src/run.py              # full run (about 20 s)
    python src/run.py --quick      # smaller, for a first look
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # figures are written to files, no window needed
import matplotlib.pyplot as plt
import numpy as np

from agent import QLearningVHDA, rounded_state
from baselines import always_5g, always_wifi, greedy, hysteresis, oracle
from env import VHDAEnv
from evaluate import evaluate
from traces import make_traces

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Results")


def train(agent, env, episodes, test_env=None, every=20):
    """One episode = one walk; returns the greedy test reward every ``every`` episodes."""
    curve = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.update_q_value(obs, action, reward, next_obs, terminated)
            done = terminated or truncated
            obs = next_obs
        agent.end_episode()
        if test_env is not None and (ep + 1) % every == 0:
            r = evaluate(lambda e, o: agent.choose_action(o, greedy=True), test_env)["reward"]
            curve.append((ep + 1, r))
    return curve


def best_margin(env, margins=(0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2)):
    """The hysteresis margin that does best on the training walks (a fair, tuned baseline)."""
    return max(margins, key=lambda m: evaluate(hysteresis(m), env)["reward"])


def main(quick=False, seed=0, handover_cost=0.1):
    n_train, n_test, steps, episodes = (10, 5, 1000, 60) if quick else (40, 10, 2000, 300)
    train_env = VHDAEnv(traces=make_traces(range(n_train), steps), handover_cost=handover_cost)
    test_env = VHDAEnv(traces=make_traces(range(1000, 1000 + n_test), steps), handover_cost=handover_cost)

    agent = QLearningVHDA(9, 2, alpha=0.1, gamma=0.95, epsilon_decay=0.98, alpha_decay=0.98, seed=seed)
    curve = train(agent, train_env, episodes, test_env)

    # the old design for comparison: every value rounded to 2 decimals
    old = QLearningVHDA(9, 2, alpha=0.1, gamma=0.95, epsilon_decay=0.98, alpha_decay=0.98,
                        state_fn=rounded_state, seed=seed)
    train(old, train_env, episodes)

    margin = best_margin(train_env)
    policies = [
        ("Always WiFi", always_wifi),
        ("Always 5G", always_5g),
        ("Greedy (best QoS now)", greedy),
        (f"Hysteresis (margin {margin}, tuned on train)", hysteresis(margin)),
        ("Q-learning, first version's state (8 values rounded)", lambda e, o: old.choose_action(o, greedy=True)),
        ("Q-learning, compact state", lambda e, o: agent.choose_action(o, greedy=True)),
        ("Oracle (DP, knows the future)", oracle),
    ]

    rows = []
    for name, policy in policies:
        tr, te = evaluate(policy, train_env), evaluate(policy, test_env)
        rows.append((name, tr, te))

    floor = rows[0][2]["reward"]
    top = rows[-1][2]["reward"]
    lines = [
        f"Train: {n_train} walks, test: {n_test} other walks, {steps} s each; handover cost {handover_cost}; seed {seed}",
        "",
        "| Policy | Test reward | % of the way from Always WiFi to Oracle | Avg QoS | Handovers / 1000 s "
        "| Unnecessary / 1000 s | Train reward |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, tr, te in rows:
        gap = 100 * (te["reward"] - floor) / (top - floor) if top != floor else 0.0
        lines.append(f"| {name} | {te['reward']:.1f} | {gap:.0f}% | {te['avg_qos']:.3f} | "
                     f"{te['handovers_per_1000']:.1f} | {te['unnecessary_per_1000']:.1f} | {tr['reward']:.1f} |")
    lines += ["", f"Q-table entries: compact {len(agent.q_table)}, rounded {len(old.q_table)}"]
    table = "\n".join(lines)
    print(table)

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "results.md"), "w") as f:
        f.write(table + "\n")

    # learning curve on the test walks
    eps, rewards = zip(*curve) if curve else ((), ())
    plt.figure(figsize=(7, 4))
    plt.plot(eps, rewards, marker="o", label="Q-learning, compact state")
    colors = iter(["tab:gray", "tab:orange", "tab:green"])
    for name, _, te in rows:
        if name.startswith(("Always WiFi", "Hysteresis", "Oracle")):
            plt.axhline(te["reward"], linestyle="--", linewidth=1, color=next(colors), label=name)
    plt.xlabel("Training episode")
    plt.ylabel("Total reward on test walks")
    plt.title("Learning curve (greedy policy, unseen walks)")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "learning_curve.png"), dpi=150)
    plt.close()

    # one stretch of a test walk: what each RAT offers and what each policy picks
    window = slice(0, min(400, steps))
    test_env.reset(options={"trace": 0})
    score = test_env.score[window]
    picks = {}
    for name, policy in (("Greedy", greedy),
                         ("Hysteresis", hysteresis(margin)),
                         ("Q-learning", lambda e, o: agent.choose_action(o, greedy=True))):
        obs, _ = test_env.reset(options={"trace": 0})
        chosen, done = [], False
        while not done:
            chosen.append(policy(test_env, obs))
            obs, _, terminated, truncated, _ = test_env.step(chosen[-1])
            done = terminated or truncated
        picks[name] = np.array(chosen)[window]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(score[:, 0], label="WiFi QoS score")
    axes[0].plot(score[:, 1], label="5G QoS score")
    axes[0].set_ylabel("QoS score")
    axes[0].legend(fontsize=8)
    for i, (name, chosen) in enumerate(picks.items()):
        axes[1].step(range(len(chosen)), chosen * 0.8 + i * 1.1, where="post")
    axes[1].set_yticks([i * 1.1 + 0.4 for i in range(len(picks))], list(picks))
    axes[1].set_xlabel("Time (s); in each row low = WiFi, high = 5G")
    fig.suptitle("A test walk: offered QoS and chosen RAT")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "test_walk.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--handover-cost", type=float, default=0.1)
    args = parser.parse_args()
    main(quick=args.quick, seed=args.seed, handover_cost=args.handover_cost)
