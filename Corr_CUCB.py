import numpy as np

class CorruptedCUCB:
    """
    CUCB algorithm for combinatorial semi-bandits without probabilistic triggering,
    with known total corruption budget C.

    Parameters
    ----------
    env: object
        Must implement step(action: List[int]) -> (obs, rewards, done, info),
        where `rewards` is a list/array of rewards for each arm in the chosen action.
    d: int
        Size of the super-arm (d < K).
    C: float
        Known total corruption budget.
    T: int
        Time horizon (number of rounds).
    """
    def __init__(self, env, d, C, T):
        self.env = env
        self.K = env.n_arms
        assert 1 <= d < self.K, "d must be in [1, K-1]"
        self.d = d
        self.C = C
        self.T = T
        # track selections for regret calc
        self.selections = []

    def _greedy_top_d(self, values):
        # return indices of top-d entries in `values`
        return list(np.argpartition(values, -self.d)[-self.d:])

    def run(self):
        T_i = np.zeros(self.K, dtype=int)
        mu_hat = np.zeros(self.K)
        cumulative_reward = 0.0
        rewards_history = []
        self.selections = []

        for t in range(1, self.T + 1):
            UCB = np.zeros(self.K)
            for i in range(self.K):
                if T_i[i] == 0:
                    UCB[i] = np.inf
                else:
                    bonus = np.sqrt(3 * np.log(t) / (2 * T_i[i]))
                    corr_bonus = self.C / (self.d * T_i[i])
                    UCB[i] = mu_hat[i] + bonus + corr_bonus

            S = self._greedy_top_d(UCB)
            _, rec, _, _ = self.env.step(S)
            for idx, arm in enumerate(S):
                r = rec[idx]
                T_i[arm] += 1
                mu_hat[arm] += (r - mu_hat[arm]) / T_i[arm]
                cumulative_reward += r
            rewards_history.append(cumulative_reward)
            self.selections.append(S)

        return np.array(rewards_history)

    def compute_regret(self):
        """
        Compute per-round regret based on true arm means stored in env.original_means.
        Regret per round = mu_star - aggregated mean of the selected super-arm.
        Returns: numpy array of length T with regrets.
        """
        mu = np.array(self.env.original_means)
        mu_star = np.max(mu)
        regrets = []
        for S in self.selections:
            # linear aggregation only
            mean_reward = np.sum(mu[S])
            regrets.append(mu_star - mean_reward)
        return np.array(regrets)
