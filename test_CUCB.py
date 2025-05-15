import numpy as np
import matplotlib.pyplot as plt
from Corr_CUCB import CorruptedCUCB

class TestEnv:
    def __init__(self, means):
        self.means = np.array(means)
        self.n_arms = len(means)
        self.original_means = means

    def step(self, action):
        # action: list of arms to pull
        rewards = (np.random.rand(len(action)) < self.means[action]).astype(float)
        return None, rewards, True, {}


means = [0.1, 0.1, 0.1, 0.8, 0.9]
env = TestEnv(means)
d = 4
C = 100000  # no corruption
T = 200000



algo = CorruptedCUCB(env, d, C, T)
cum_rewards = algo.run()
regrets = algo.compute_regret()
agg_regret = np.cumsum(regrets)

# Plot aggregated regret per round
plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, T+1), agg_regret, label='Cumulative Regret')
plt.xlabel('Round')
plt.ylabel('Cumulative Regret')
plt.title('CUCB w/ C=0: Aggregated Regret per Round')
plt.legend()
plt.grid(True)
plt.show()
