import numpy as np
import gym
from gym import spaces

class CorruptedSemiBanditEnv(gym.Env):
    """
    Combinatorial semi-bandit environment with optional adversarial corruption.

    Each arm i has a fixed Bernoulli distribution with mean mu[i]. When pulled,
    it returns a sample X_i ~ Bernoulli(mu[i]). If corruption is enabled, an
    additional perturbation c_i = corruption(X_i, i) is added before returning.

    Parameters
    ----------
    n_arms : int
        Number of arms in the environment.
    mu : array-like of shape (n_arms,)
        Mean parameters for each Bernoulli arm (each in [0,1]).
    corrupted : bool, default=False
        Whether to apply corruption to each drawn reward.
    """
    def __init__(self, n_arms, mu, corrupted=False):
        super().__init__()
        if len(mu) != n_arms:
            raise ValueError("Length of mu must equal n_arms")
        self.n_arms = n_arms
        self.mu = np.array(mu, dtype=float)
        if np.any(self.mu < 0) or np.any(self.mu > 1):
            raise ValueError("Each mean must lie in [0,1]")

        self.corrupted = corrupted

        # Action: which arm to pull
        self.action_space = spaces.Discrete(self.n_arms)
        # Observation: placeholder (no sequential state)
        self.observation_space = spaces.Discrete(1)

    def reset(self):
        """Resets the environment (stateless)."""
        return 0

    def corruption(self, reward, arm):
        """
        Placeholder for corruption logic: given the raw reward and arm index, return an
        additive corruption term. To be implemented later.
        """
        # TODO: implement corruption mechanism
        raise NotImplementedError("Corruption function not yet implemented")

    def step(self, action):
        """
        Pulls arm `action`. Returns a tuple of (obs, reward, done, info).

        - obs: always 0 (no state tracking).
        - reward: X_i + c_i if corrupted else X_i, where X_i ~ Bernoulli(mu[i]).
        - done: always True (each pull is a separate episode).
        - info: empty dict.
        """
        assert self.action_space.contains(action), "Invalid arm index"

        # draw stochastic reward from Bernoulli
        base = np.random.binomial(1, self.mu[action])

        # apply corruption if enabled
        if self.corrupted:
            corr = self.corruption(base, action)
        else:
            corr = 0

        reward = base + corr
        return 0, reward, True, {}


if __name__ == "__main__":
    # Small test without corruption
    n_arms = 3
    means = [0.2, 0.5, 0.8]
    env = CorruptedSemiBanditEnv(n_arms, means, corrupted=False)

    pulls = 10000
    counts = np.zeros(n_arms)
    sums = np.zeros(n_arms)

    for i in range(n_arms):
        for _ in range(pulls):
            _, r, done, _ = env.step(i)
            sums[i] += r
            counts[i] += 1

    empirical = sums / counts
    print("True means:     ", means)
    print("Empirical means:", np.round(empirical, 3))
    # Simple assertion that empirical means are within ~0.02
    for i in range(n_arms):
        assert abs(empirical[i] - means[i]) < 0.02, f"Arm {i} estimate off: {empirical[i]} vs {means[i]}"
    print("Test passed: empirical means match true means within tolerance.")
