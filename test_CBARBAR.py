import numpy as np
from CBARBAR import CBARBAR, simple_greedy_oracle
import matplotlib.pyplot as plt
def test_cbarbar_no_corruption():

    # A simple test environment
    class TestEnv:
        def __init__(self, means):
            self.means = np.array(means)
            self.n_arms = len(means)
            self.original_means = means

        def step(self, action, corr_percent = 0.4):
            # action: list of arms to pull
            rewards = (np.random.rand(len(action)) < self.means[action]).astype(float)
            if corr_percent > 0:
                # Generate a mask indicating which bits to flip
                flip_mask = np.random.rand(len(rewards)) < corr_percent
                # Flip the bits: 1 becomes 0, 0 becomes 1
                rewards[flip_mask] = 1 - rewards[flip_mask]
            return None, rewards, True, {}
    # setup
    means = [0.1, 0.1, 0.1, 0.8, 0.9]
    env = TestEnv(means)
    K = len(means)
    d = 2 
    T = 100000
    alpha = 1.0  # exact oracle
    beta = 1.0   # always succeeds
    delta = 0.1

    # oracle: pick top-d arms by means
    def oracle(weights, include=None):
        return simple_greedy_oracle(np.array(weights), d, include)

    # run CBARBAR
    algo = CBARBAR(env, oracle, T, alpha, beta, delta, d, reward_function = "cascadian")
    # algo = CBARBAR(env, oracle, T, alpha, beta, delta, d, reward_function = "linear")
    cum_rewards = algo.run()
    regrets = algo.compute_regret()

    # basic checks
    assert len(cum_rewards) == T, "Cumulative rewards should have length T"
    assert len(regrets) == T, "Regrets array should have length T"
    # regret non-negative
    assert np.all(regrets >= -1e-8), "Regrets must be non-negative"

    # final regret should be finite
    print(f"Final cumulative regret: {regrets.sum():.3f}")

    # plot aggregated (cumulative) regret per round
    
    cum_regret = np.cumsum(regrets)
    plt.figure()
    plt.plot(cum_regret, label='Cumulative Regret')
    plt.xlabel('Round t')
    plt.ylabel('Cumulative Regret')
    plt.title('CBARBAR: Cumulative Regret over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test_cbarbar_no_corruption()
    print("test_cbarbar_no_corruption passed.")