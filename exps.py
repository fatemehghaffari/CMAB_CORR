import numpy as np
from BASIC import GCOBE
from CBARBAR import CBARBAR, simple_greedy_oracle
import matplotlib.pyplot as plt

class TestEnv:
    def __init__(self, means):
        self.means = np.array(means)
        self.n_arms = len(means)
        self.original_means = means

    def step(self, action, corr_percent = 0.3):
        # action: list of arms to pull
        rewards = (np.random.rand(len(action)) < self.means[action]).astype(float)
        if corr_percent > 0:
            # Generate a mask indicating which bits to flip
            flip_mask = np.random.rand(len(rewards)) < corr_percent
            # Flip the bits: 1 becomes 0, 0 becomes 1
            rewards[flip_mask] = 1 - rewards[flip_mask]
        return None, rewards, True, {}


# --- Test parameters ---
means = [0, 0, 0, 0.8, 0.6]
env = TestEnv(means)
b_1 = 1
d = 2
T = 100000
delta = 0.1
beta1 = len(means) * np.log(T) * d * b_1
beta2 = (len(means) / d) * (np.log(T))
beta3 = len(means)

# Instantiate and run GCOBE
algo = GCOBE(env, T, delta, d, beta1, beta2, beta3)
regrets = algo.gcobe()
# Sanity checks
assert len(regrets) == T, f"Expected {T} regrets, got {len(regrets)}"
assert np.all(regrets >= 0), "Regrets should be non-negative"

# Plot cumulative regret
cum_regret = np.cumsum(regrets)
# plt.figure(figsize=(8, 4))
# plt.plot(np.arange(1, T+1), cum_regret, label='G-COBE Cumulative Regret')
# plt.xlabel('Round t')
# plt.ylabel('Cumulative Regret')
# plt.title('G-COBE: Aggregated Regret per Round')
# plt.legend()
# plt.grid(True)
# plt.show()

# oracle: pick top-d arms by means
def oracle(weights, include=None):
    return simple_greedy_oracle(np.array(weights), d, include)

# run CBARBAR
algo = CBARBAR(env, oracle, T, 1.0, 1.0, delta, d, reward_function = "cascadian")
# algo = CBARBAR(env, oracle, T, alpha, beta, delta, d, reward_function = "linear")
cum_rewards = algo.run()
regs = algo.compute_regret()

# basic checks
assert len(cum_rewards) == T, "Cumulative rewards should have length T"
assert len(regs) == T, "Regrets array should have length T"
# regret non-negative
assert np.all(regs >= -1e-8), "Regrets must be non-negative"

# final regret should be finite
print(f"Final cumulative regret: {regs.sum():.3f}")

# plot aggregated (cumulative) regret per round

cum_regret_CB = np.cumsum(regs)
plt.figure()
plt.plot(cum_regret_CB, label='CBARBAR')
plt.plot(cum_regret, label='MS')
plt.xlabel('Round t')
plt.ylabel('Cumulative Regret')
plt.title('CBARBAR vs MS: W/ Corr (0.3)')
plt.legend()
plt.tight_layout()
plt.show()

