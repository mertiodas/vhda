Train: 40 walks, test: 10 other walks, 2000 s each; handover cost 0.1; seed 0

| Policy | Test reward | % of the way from Always WiFi to Oracle | Avg QoS | Handovers / 1000 s | Unnecessary / 1000 s | Train reward |
|---|---|---|---|---|---|---|
| Always WiFi | 1979.4 | 0% | 0.099 | 0.0 | 0.0 | 7149.3 |
| Always 5G | 186.3 | -144% | 0.009 | 0.5 | 0.3 | 1526.2 |
| Greedy (best QoS now) | 3167.0 | 96% | 0.163 | 49.6 | 33.9 | 12418.4 |
| Hysteresis (margin 0.05, tuned on train) | 3210.0 | 99% | 0.163 | 20.1 | 0.0 | 12616.6 |
| Q-learning, first version's state (8 values rounded) | 1976.7 | -0% | 0.099 | 5.9 | 3.2 | 11538.8 |
| Q-learning, compact state | 3199.4 | 98% | 0.162 | 18.9 | 3.2 | 12561.4 |
| Oracle (DP, knows the future) | 3222.2 | 100% | 0.163 | 16.0 | 7.0 | 12670.8 |

Q-table entries: compact 40, rounded 133788
