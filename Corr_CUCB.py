import numpy as np 

class CorruptedCUCB:
    def __init__(self, env, d, C, T):
        self.env = env
        self.K = env.n_arms
        self.d = d
        self.C = C
        self.T = T
        self.selections = []
        # per-arm statistics
        self.T_i = np.zeros(self.K, dtype=int)
        self.mu_hat = np.zeros(self.K)

        # track time
        self.t = 0

    def _greedy_top_d(self, values):
        return list(np.argpartition(values, -self.d)[-self.d:])

    def select(self):
        """
        Choose a super-arm (list of d arms) based on current UCB estimates.
        """
        self.t += 1
        UCB = np.zeros(self.K)
        for i in range(self.K):
            if self.T_i[i] == 0:
                UCB[i] = np.inf
            else:
                bonus = np.sqrt(3 * np.log(self.t) / (2 * self.T_i[i]))
                corr_bonus = self.C / (self.d * self.T_i[i])
                UCB[i] = self.mu_hat[i] + bonus + corr_bonus
        self.current_S = self._greedy_top_d(UCB)
        return self.current_S
    
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

    def update(self, rewards):
        """
        Update internal statistics given rewards from last selected super-arm.
        rewards: array of length d corresponding to current_S
        """
        for idx, arm in enumerate(self.current_S):
            r = rewards[idx]
            self.T_i[arm] += 1
            self.mu_hat[arm] += (r - self.mu_hat[arm]) / self.T_i[arm]
            
    def compute_regret(self):
        mu = np.array(self.env.original_means)
        # best super-arm of size d: sum of top-d means
        best_super = list(np.argsort(mu)[-self.d:])
        mu_star = np.sum(mu[best_super])
        regrets = []
        for S in self.selections:
            mean_reward = np.sum(mu[S])
            regrets.append(mu_star - mean_reward)
        return np.array(regrets)
