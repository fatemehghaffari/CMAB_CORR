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
    def __init__(self, env, oracle, T, alpha, beta, delta, d):
        self.env = env
        self.oracle = oracle
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.d = d
        self.K = env.n_arms

    def run(self):
        Delta = np.ones(self.K)
        Z_i = [[i] for i in range(self.K)]
        Z_star = self.oracle(np.zeros(self.K))

        # λ ← 1024 · [log₂(8K / (δ · (log₂ T)²))]²
        lam = 1024 * (np.log2((8 * self.K) / (self.delta * (np.log2(self.T) ** 2))) ** 2)
        t = 0
        cumulative_rewards = []
        m = 1

        while t < self.T:
            nm_star = lam * (self.d ** 2) * self.K * (2 ** ((m - 1) / 2))
            nm_i = lam * (self.d / Delta) ** 2  # vector of length K
            N = int(nm_star + nm_i.sum())
            q_star = nm_star / N
            q_i = nm_i / N

            sum_rewards = np.zeros(self.K)
            # no longer need count_pulls—use nm_i directly

            for _ in range(N):
                if t >= self.T:
                    break
                if np.random.rand() < q_star:
                    S = Z_star
                else:
                    arm = np.random.choice(self.K, p=q_i)
                    S = Z_i[arm]

                _, rec, _, _ = self.env.step(S)
                for idx, arm in enumerate(S):
                    sum_rewards[arm] += rec[idx]
                cumulative_rewards.append(sum(rec))
                t += 1

            # estimate means using fixed nm_i pulls per arm
            mu_hat = np.zeros(self.K)
            for i in range(self.K):
                if nm_i[i] > 0:
                    mu_hat[i] = sum_rewards[i] / nm_i[i]

            # compute lower confidence bound
            LCB = mu_hat - Delta / (16 * self.d)
            UCB = mu_hat + Delta / (16 * self.d)
          
            # update representative super-arms per arm
            for i in range(self.K):
                Z_i[i] = self.oracle(LCB, include=i)
            Z_star = self.oracle(LCB)

            # compute empirical rewards under current LCB
            r_star = np.sum(UCB[self.oracle(UCB)])
            r_i = np.array([np.sum(LCB[z]) for z in Z_i])

            # update gap estimates
            for i in range(self.K):
                Delta[i] = max(
                    2 ** (-m / 4),
                    r_star - r_i[i],
                    Delta[i] / 2
                )
            m += 1

        # store selections for regret
        self.selections = selections  # ensure selections tracked
        return np.cumsum(cumulative_rewards)

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
            mean_reward = np.sum(mu[S])  # linear aggregation
            regrets.append(mu_star - mean_reward)
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
