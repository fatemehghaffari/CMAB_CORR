import numpy as np
from Corr_CUCB import CorruptedCUCB
from TMS import TwoModelSelect 
import matplotlib.pyplot as plt
from CBARBAR import simple_greedy_oracle
class TestEnv:
    def __init__(self, means, C=0):
        self.means = np.array(means)
        self.n_arms = len(means)
        self.original_means = means
        self.C = C
        self.bestarmrew = np.max(means)

    def step(self, action, corr_percent = 0):
        # action: list of arms to pull
        rewards = (np.random.rand(len(action)) < self.means[action]).astype(float)
        if self.C > 0:
            for ind in action:
                if (means[ind] != self.bestarmrew) and (rewards[action.index(ind)] == 0):
                    rewards[action.index(ind)] = 1
                    self.C -= 1
                if (means[ind] == self.bestarmrew) and (rewards[action.index(ind)] == 1):
                    rewards[action.index(ind)] = 0
                    self.C -= 1
        if corr_percent > 0:
            # Generate a mask indicating which bits to flip
            flip_mask = np.random.rand(len(rewards)) < corr_percent
            # Flip the bits: 1 becomes 0, 0 becomes 1
            rewards[flip_mask] = 1 - rewards[flip_mask]

        return None, rewards, True, {}
    



class GCOBE:
    """
    Gap-Adaptive Corruption-Robust CMAB (G-COBE, Algorithm 3). Records per-round regret.
    """
    def __init__(self, oracle, env, T, delta, d, beta1, beta2, beta3, reward_function = "linear"):
        self.env = env
        self.reward_function = reward_function
        self.T = T
        self.delta = delta
        self.d = d
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.oracle = oracle
        mu = np.array(env.original_means)
        self.best_super = self.oracle(mu, include=None)
        self.mu_star = self.reward_func(mu[self.best_super])
        self.regrets = []

        self.N = {}
        self.R_sum = {}
        self.history = {}


        self.pi_hat = None

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

    def reset(self):
        self.regrets = []
        self.N = {}
        self.R_sum = {}
        self.history = {}
    
    def set_basic_history(self, k, k_max):
        self.N = {i: 0 for i in range(k, k_max + 1)}
        self.R_sum = {i: 0.0 for i in range(k, k_max + 1)}
        self.history = {i: [] for i in range(k, k_max + 1)}

    def basic_cucb(self, T, k, k_max, alpha, return_policy=False, excluded_policy=None):
        """
        BASIC meta-algorithm testing corruption budgets 2^i for i in [k..k_max],
        optionally returning per-round regrets and the candidate policy.
        """
        self.set_basic_history(k, k_max)

        def R_B(n, theta):
            return np.sqrt(3 * n * np.log(T / self.delta)) + theta

        learners = {}
        theta_i = {}
        
        for i in range(k, k_max + 1):
            theta_i[i] = 1.25 * alpha[i] * (2 ** i) + 21 * self.d * np.log(T / self.delta)
            learners[i] = CorruptedCUCB(self.env, self.d, theta_i[i], T, excluded_policy = excluded_policy, reward_function = "cascadian")

        regrets = [] if return_policy else None

        for t in range(1, T + 1):
            idxs = list(range(k, k_max + 1))
            probs = np.array([alpha[i] for i in idxs])
            # probs /= probs.sum()
            i_t = np.random.choice(idxs, p=probs)

            S = learners[i_t].select()
            self.history[i_t].append(tuple(S))
            _, rec, _, _ = self.env.step(S)
            learners[i_t].update(rec)

            if return_policy:
                reward = self.reward_func(self.env.means[S])
                regrets.append(self.mu_star - reward)

            self.N[i_t] += 1
            self.R_sum[i_t] += self.reward_func(rec)

            for i in idxs:
                for j in idxs:
                    if j <= i: continue
                    lhs = self.R_sum[i]/alpha[i] + R_B(self.N[i], theta_i[i])/alpha[i]
                    rhs = self.R_sum[j]/alpha[j] - 8*(np.sqrt(t*np.log(T/self.delta)/alpha[j])
                                            + (np.log(T/self.delta)+theta_i[j])/alpha[j])
                    if lhs < rhs:
                        if return_policy:
                            return False, np.array(regrets)
                        else:
                            return False
                        
        if return_policy:
            if self.history[k]:
                # count occurrences of each tuple
                counts = {}
                for arm_tuple in self.history[k]:
                    counts[arm_tuple] = counts.get(arm_tuple, 0) + 1
                # pick the tuple with max count
                pi_hat_tuple = max(counts, key=counts.get)
                self.pi_hat = list(pi_hat_tuple)
            else:
                # fallback if learner k never got sampled: just its current select()
                self.pi_hat = learners[k].select()
            return True, np.array(regrets)
        else:
            return True



    def cobe(self, T, excluded_policy=None, return_regrets=False):
        """
        COBE meta-algorithm (Alg.2), optionally excluding a given super-arm policy.

        Parameters
        ----------
        env: environment with .step(super_arm) and .original_means
        T: int
        delta: float
        d: int
        beta1, beta2, beta3: floats
        excluded_policy: list[int] or None
            If provided, cobe will skip any candidate pi_hat equal to this.
        return_regrets: bool
            If True, returns (k, pi_hat, regrets). Otherwise just k.
        """
        c_max = self.d
        # exponent range
        term = (np.sqrt(self.beta1 * T) + self.beta2 * c_max + self.beta3) / self.beta2
        k_init = max(int(np.ceil(np.log2(term))), 0)
        k_max  = int(np.ceil(np.log2(c_max * T)))

        for k in range(k_init, k_max + 1):
            # build sampling weights alpha_i over {k..k_max}
            alpha = {}
            rem = 1.0
            for i in range(k+1, k_max+1):
                alpha[i] = 2 ** (k - i - 1)
                rem -= alpha[i]
            alpha[k] = rem

            # run BASIC to get candidate policy and regrets
            ok, regrets = self.basic_cucb(T, k, k_max, alpha, True, excluded_policy)
            # if excluded, skip this k
            if excluded_policy is not None and set(self.pi_hat) == set(excluded_policy):
                continue

            if ok:
                if return_regrets:
                    return k, regrets
                return k

        # no k succeeded
        if return_regrets:
            return None, np.array([])
        return None

    def gcobe(self):
        self.reset()
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
            alpha = {}
            rem = 1.0
            for i in range(k+1, k_max+1):
                alpha[i] = 2 ** (k - i - 1)
                rem -= alpha[i]
            alpha[k] = rem
            ok, rp1 = self.basic_cucb(L, k, k_max, alpha, return_policy = True)
            self.regrets += rp1.tolist()
            t += L
            if ok:
                # Phase 2: TwoModelSelect
                # challenger = self.cobe(env, T,
                #                 self.beta1, self.beta2, self.beta3,
                #                 return_regrets=True)
                tms = TwoModelSelect(self.env, L, self.delta, self.d, self.pi_hat, self.beta1, 
                                    self.beta2, self.beta3, beta4, self.T, reward_function=self.reward_function)
                rp2 = tms.run_and_record(self.mu_star)
                self.regrets += rp2.tolist() 
                t += len(rp2)
                # Phase 3: COBE fallback
                rem = self.T - t
                if rem>0:
                    _, rp3 = self.cobe(rem, return_regrets=True)
                    self.regrets += rp3.tolist()
                break
        if len(self.regrets)<self.T:
            self.regrets += [0.0]*(self.T-len(self.regrets))
        return np.array(self.regrets)
d=3
def oracle(weights, include=None):
    return simple_greedy_oracle(np.array(weights), d, include)
if __name__ == '__main__':
    # --- Test parameters ---
    means = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    T = 200000
    C = 50000
    env = TestEnv(means, 2*C)
    b_1 = 1

    delta = 0.1
    beta1 = len(means) * np.log(T) * d * b_1
    beta2 = d * b_1
    beta3 = len(means)
    
    # Instantiate and run GCOBE
    algo = GCOBE(oracle, env, T, delta, d, beta1, beta2, beta3, reward_function='cascadian')
    regrets = algo.gcobe()
    # Sanity checks
    assert len(regrets) == T, f"Expected {T} regrets, got {len(regrets)}"
    assert np.all(regrets >= 0), "Regrets should be non-negative"

    # Plot cumulative regret
    cum_regret = np.cumsum(regrets)
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, T+1), cum_regret, label='G-COBE Cumulative Regret')
    plt.xlabel('Round t')
    plt.ylabel('Cumulative Regret')
    plt.title('G-COBE: Aggregated Regret per Round')
    plt.legend()
    plt.grid(True)
    plt.show()