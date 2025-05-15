import numpy as np 

class CorruptedCUCB:
    def __init__(self, env, d, C, T, excluded_policy=None, reward_function = "linear"):
        self.env = env
        self.K = env.n_arms
        self.d = d
        self.C = C
        self.T = T
        self.selections = []
        # per-arm statistics
        self.T_i = np.zeros(self.K, dtype=int)
        self.mu_hat = np.zeros(self.K)
        self.excluded_policy = excluded_policy
        # track time
        self.t = 0
        self.reward_function = reward_function
        
    def reward_func(self, rewards):
        """
        Aggregates a list of arm rewards into a super-arm reward.
        If linear, returns sum; otherwise placeholder for non-linear.
        """
        if self.reward_function == 'linear':
            return np.sum(rewards)
        elif self.reward_function == 'cascadian':
            return 1 - np.prod(1 - rewards)
        # non-linear aggregator goes here
        raise NotImplementedError("Non-linear reward aggregation not yet implemented")
    
    def _greedy_top_d(self, values):
        top_d = list(np.argpartition(values, -self.d)[-self.d:])
        if self.excluded_policy and (set(self.excluded_policy) == set(top_d)):
            return list(np.argpartition(values, -self.d)[-self.d-1:-1])
        return top_d

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
                # cumulative_reward += r
            cumulative_reward += self.reward_func(rec)
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
        mu_star = self.reward_func(mu[best_super])
        regrets = []
        for S in self.selections:
            mean_reward = self.reward_func(mu[S])
            regrets.append(mu_star - mean_reward)
        return np.array(regrets)