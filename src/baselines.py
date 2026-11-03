"""Policies to compare the agent with, and the best any policy could do.

A policy here is a function (env, obs) -> action. Baselines read the QoS score of each RAT
for the current second from the environment, as a rule-based handover algorithm would
read its measurements.
"""

import numpy as np

from env import WIFI, NR


def always_wifi(env, obs):
    return WIFI


def always_5g(env, obs):
    return NR


def greedy(env, obs):
    """The RAT with the better QoS score right now."""
    wifi, nr = env.score[env.t]
    return int(nr > wifi)


def hysteresis(margin):
    """Stay on the current RAT unless the other one is better by more than ``margin``."""
    def policy(env, obs):
        current = env.rat
        other = 1 - current
        return other if env.score[env.t, other] - env.score[env.t, current] > margin else current
    policy.__name__ = f"hysteresis({margin})"
    return policy


def oracle_actions(score, handover_cost, initial_rat=WIFI):
    """Best possible actions for one trace, knowing the whole future (dynamic programming).

    With the reward of the environment (QoS of the chosen RAT, minus a fixed cost on every
    switch) the only thing carried from one step to the next is the serving RAT, so the
    optimum over all action sequences is a two-state backward recursion. No real policy can
    beat it; it is the upper bound of the table.
    """
    T = len(score)
    # value[t, r]: best total reward from step t on, when the user is on RAT r before step t
    value = np.zeros((T + 1, 2))
    best = np.zeros((T, 2), dtype=int)
    for t in range(T - 1, -1, -1):
        for r in (WIFI, NR):
            options = [score[t, a] - (handover_cost if a != r else 0.0) + value[t + 1, a] for a in (WIFI, NR)]
            best[t, r] = int(np.argmax(options))
            value[t, r] = max(options)
    actions, r = [], initial_rat
    for t in range(T):
        r = best[t, r]
        actions.append(r)
    return actions


def oracle(env, obs):
    """Follows the dynamic-programming optimum of the trace the environment is on."""
    # keyed on everything the optimum depends on, so a changed env never replays an old answer
    key = (env.trace_index, env.handover_cost, env.initial_rat, id(env.score))
    cache = getattr(env, "_oracle_cache", {})
    if key not in cache:
        cache[key] = oracle_actions(env.score, env.handover_cost, env.initial_rat)
        env._oracle_cache = cache
    return cache[key][env.t]
