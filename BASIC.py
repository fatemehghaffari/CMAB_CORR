import numpy as np
from Corr_CUCB import CorruptedCUCB
from TMS import TwoModelSelect 
import matplotlib.pyplot as plt

class TestEnv:
    def __init__(self, means):
        self.means = np.array(means)
        self.n_arms = len(means)
        self.original_means = means

    def step(self, action):
        # action: list of arms to pull
        rewards = (np.random.rand(len(action)) < self.means[action]).astype(float)
        return None, rewards, True, {}


def basic_cucb(env, T, delta, k, k_max, alpha, d, return_policy=False):
    """
    BASIC meta-algorithm testing corruption budgets 2^i for i in [k..k_max],
    optionally returning per-round regrets and the candidate policy.
    """
    mu = np.array(env.original_means)
    best_super = list(np.argsort(mu)[-d:])
    mu_star = np.sum(mu[best_super])

    def R_B(n, theta):
        return np.sqrt(3 * n * np.log(T / delta)) + theta

    learners = {}
    theta_i = {}
    N = {i: 0 for i in range(k, k_max + 1)}
    R_sum = {i: 0.0 for i in range(k, k_max + 1)}
    for i in range(k, k_max + 1):
        theta_i[i] = 1.25 * alpha[i] * (2 ** i) + 21 * d * np.log(T / delta)
        learners[i] = CorruptedCUCB(env, d, theta_i[i], T)

    regrets = [] if return_policy else None

    for t in range(1, T + 1):
        idxs = list(range(k, k_max + 1))
        probs = np.array([alpha[i] for i in idxs])
        probs /= probs.sum()
        i_t = np.random.choice(idxs, p=probs)

        S = learners[i_t].select()
        _, rec, _, _ = env.step(S)
        learners[i_t].update(rec)

        if return_policy:
            reward = np.sum(rec)
            regrets.append(mu_star - reward)

        N[i_t] += 1
        R_sum[i_t] += np.sum(rec)

        for i in idxs:
            for j in idxs:
                if j <= i: continue
                lhs = R_sum[i]/alpha[i] + R_B(N[i], theta_i[i])/alpha[i]
                rhs = R_sum[j]/alpha[j] - 8*(np.sqrt(t*np.log(T/delta))/alpha[j] + (np.log(T/delta)+theta_i[j])/alpha[j])
                if lhs < rhs:
                    if return_policy:
                        return False, None, np.array(regrets)
                    else:
                        return False
    if return_policy:
        pi_hat = learners[k].select()
        return True, pi_hat, np.array(regrets)
    else:
        return True

def cobe(env, T, delta, d, beta1, beta2, beta3):
    """
    COmponent BEllman (CoBE) meta-algorithm, Algorithm 2 from
    "A Model Selection Approach for Corruption Robust RL" citeturn3file5.

    Iterates over hypothesized corruption levels 2^k and invokes BASIC (basic_cucb).

    Parameters
    ----------
    env: environment supporting .step(super_arm) and .original_means
    T: int
        Total time horizon (rounds).
    delta: float
        Confidence parameter δ.
    d: int
        Maximum super-arm size (also c_max).
    beta1: float
        Parameter β₁ controlling initial term (e.g., 3·log T).
    beta2: float
        Parameter β₂ for scaling c_max term (e.g., 1).
    beta3: float
        Parameter β₃ additive offset (often 0).

    Returns
    -------
    Smallest k for which BASIC returns True, or None if none succeed.
    """
    c_max = d

    # Determine exponent range
    term = (np.sqrt(beta1 * T) + beta2 * c_max + beta3) / beta2
    k_init = int(np.ceil(np.log2(term)))
    k_init = max(k_init, 0)
    k_max = int(np.ceil(np.log2(c_max * T)))

    # Loop over candidate corruption exponents
    for k in range(k_init, k_max + 1):
        # Build sampling weights α_i over {k,...,k_max}
        alpha = {}
        rem = 1.0
        for i in range(k + 1, k_max + 1):
            alpha[i] = 2 ** (k - i - 1)
            rem -= alpha[i]
        alpha[k] = rem

        # Invoke BASIC meta-algorithm with budgets 2^i
        ok = basic_cucb(env, T, delta, k, k_max, alpha, d)
        if ok:
            return k
    return None

class GCOBE:
    """
    Gap-Adaptive Corruption-Robust CMAB (G-COBE, Algorithm 3). Records per-round regret.
    """
    def __init__(self, env, T, delta, d, beta1, beta2, beta3, linear=True):
        self.env = env
        self.T = T
        self.delta = delta
        self.d = d
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.linear = linear
        mu = np.array(env.original_means)
        self.mu_star = np.sum(np.sort(mu)[-d:]) if linear else None
        self.regrets = []

    def run(self):
        Z = self.d
        k_init = int(np.ceil(np.log2((np.sqrt(self.beta1) + self.beta2*Z + self.beta3)/self.beta2)))
        k_init = max(k_init, 0)
        k_max = int(np.ceil(np.log2(Z*self.T)))
        beta4 = 1e4*(2*self.beta1 + 42*self.beta2*Z*np.log(self.T/self.delta) + 2*self.beta3)

        t = 0
        for k in range(k_init, k_max + 1):
            L = int(np.ceil((self.beta2*(2**k))**2 / beta4))
            if L > self.T: break
            # Phase 1: BASIC
            alpha = {i:1/(k_max-k+1) for i in range(k,k_max+1)} #IMPLEMENT THE CRRECT ALPHA
            ok, pi_hat, rp1 = basic_cucb(self.env, L, self.delta, 
                                         k, k_max, alpha,
                                           self.d, return_policy=True)
            self.regrets.extend(rp1.tolist())
            t += L
            if ok:
                # Phase 2: TwoModelSelect
                challenger = None  # TODO: construct challenger on pi_hat
                tms = TwoModelSelect(self.env, L, pi_hat, challenger, self.beta1, self.beta2, self.beta3, beta4, self.T, linear=self.linear)
                rp2 = tms.run_and_record(self.mu_star)
                self.regrets.extend(rp2.tolist()); t += len(rp2)
                # Phase 3: COBE fallback
                rem = self.T - t
                if rem>0:
                    rp3 = cobe(self.env, rem, self.delta, self.d, self.beta1, self.beta2, self.beta3, return_regrets=True)
                    self.regrets.extend(rp3.tolist())
                break
        if len(self.regrets)<self.T:
            self.regrets += [0.0]*(self.T-len(self.regrets))
        return np.array(self.regrets)

# means = [0.2, 0.5, 0.3, 0.8, 0.6]
means = [0, 0, 0, 0.8, 0.6]
env = TestEnv(means)
d = 2
T = 10000
delta = 0.1

# Uniform alpha over single index for k=k_max=0
alpha = {0: 1.0}

# Run basic_cucb test
success, pi_hat, regrets_basic = basic_cucb(env, T, delta, 0, 0, alpha, d, return_policy=True)
cum_regret_basic = np.cumsum(regrets_basic)

# Plot aggregated regret per round for basic_cucb
plt.figure()
plt.plot(cum_regret_basic, label='BASIC-CUCB')
plt.xlabel('Round t')
plt.ylabel('Cumulative Regret')
plt.title('BASIC-CUCB: Aggregated Regret per Round')
plt.legend()
plt.show()