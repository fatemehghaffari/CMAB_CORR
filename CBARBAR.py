import numpy as np

class CBARBAR:
    """
    CBARBAR algorithm for combinatorial semi-bandits with adversarial corruption (CMAB-AC).

    Implements Algorithm 1 from "Simple Combinatorial Algorithms for Combinatorial Bandits:"
    Corruptions and Approximations, with semi-bandit feedback.

    Parameters
    ----------
    env: object
        Must support step(super_arm: List[int]) -> (obs, rewards, done, info), where
        `rewards` is a sequence matching super_arm.
    oracle: function
        oracle(weights: np.ndarray, include: Optional[int]=None) -> List[int]
        Should return a size<=d subset maximizing total weight approximately.
    T: int
        Time horizon.
    alpha: float
        Oracle approximation ratio.
    beta: float
        Oracle success probability.
    delta: float
        Confidence parameter.
    d: int
        Maximum super-arm size.
    """
    def __init__(self, env, oracle, T, alpha, beta, delta, d, reward_function = "linear"):
        self.env = env
        self.oracle = oracle
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.d = d
        self.K = env.n_arms
        self.selections = []
        self.reward_function = reward_function
        true_mu = np.array(env.original_means)
        self.best_super = self.oracle(true_mu, include=None)
        self.mu_star = self.reward_func(true_mu[self.best_super])


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


    def run(self):
        Delta = np.ones(self.K)
        Z_i = [[i] for i in range(self.K)]
        Z_star = self.oracle(np.zeros(self.K))

        # λ ← 1024 · [log₂(8K / (δ · (log₂ T)²))]²
        lam = 64 * np.log2(((8 * self.K) / (self.delta)) * (np.log2(self.T)))
        t = 0
        cumulative_rewards = []
        m = 1

        while t < self.T:
            nm_star = lam * (self.d ** 2) * self.K * (2 ** ((m - 1) / 2))
            nm_i = lam * (self.d / Delta) ** 2  # vector of length K
            N = nm_star + nm_i.sum()
            q_star = nm_star / N
            q_i = nm_i / N

            sum_rewards = np.zeros(self.K)
            # no longer need count_pulls—use nm_i directly
            for _ in range(int(N)):
                if t >= self.T:
                    break
                # if np.random.rand() < q_star:
                #     S = Z_star
                else:
                    arm = np.random.choice(self.K + 1, p=np.append(q_i, q_star))
                    if arm == self.K:
                        S = Z_star
                    else:
                        S = Z_i[arm]

                _, rec, _, _ = self.env.step(S)

                for idx, arm in enumerate(S):
                    sum_rewards[arm] += rec[idx]
                cumulative_rewards.append(self.reward_func(rec))
                self.selections.append(S)  # track selection
                t += 1

            # estimate means using fixed nm_i pulls per arm
            mu_hat = np.zeros(self.K)
            for i in range(self.K):
                if nm_i[i] > 0:
                    mu_hat[i] = sum_rewards[i] / nm_i[i]
            # print("sum_rewards", sum_rewards)
            # compute lower confidence bound
            LCB = mu_hat - Delta / (16 * self.d)
            UCB = mu_hat + Delta / (16 * self.d)
            # update representative super-arms per arm
            for i in range(self.K):
                Z_i[i] = self.oracle(LCB, include=i)
            Z_star = self.oracle(LCB)
            # compute empirical rewards under current LCB
            r_star = self.reward_func(UCB[self.oracle(UCB)])
            r_i = np.array([self.reward_func(LCB[z]) for z in Z_i])
            # print("r_star", r_star)
            # print("r_i", r_i)

            # update gap estimates
            for i in range(self.K):
                Delta[i] = max(
                    2 ** (-m / 4),
                    r_star - r_i[i],
                    Delta[i] / 2
                )
            m += 1
            
        # store selections for regret
        return np.cumsum(cumulative_rewards)

    def compute_regret(self):
        """
        Compute per-round regret based on true arm means stored in env.original_means.
        Regret per round = mu_star - aggregated mean of the selected super-arm.
        Returns: numpy array of length T with regrets.
        """
        mu = np.array(self.env.original_means)
        # mu_star = np.max(mu)
        regrets = []
        for S in self.selections:
            mean_reward = self.reward_func(mu[S])

            regrets.append(self.mu_star - mean_reward)
        return np.array(regrets)


def simple_greedy_oracle(weights: np.ndarray, d: int, include: int = None) -> list:
    """
    Fast 1-approximation oracle: pick the top-d arms by weight.
    If `include` is specified, ensures that arm is in the returned subset.
    """
    K = len(weights)
    if include is None:
        return list(np.argsort(weights)[-d:][::-1])
    else:
        picks = {include}
        candidates = [i for i in np.argsort(weights)[::-1] if i != include]
        for i in candidates:
            if len(picks) >= d:
                break
            picks.add(i)
        return list(picks)
