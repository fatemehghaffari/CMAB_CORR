import numpy as np

class TwoModelSelect:
    """
    TwoModelSelect: Adaptive Gap-Tester for CMAB (Algorithm 4 in MSC paper) citeturn5file2.

    Parameters
    ----------
    env: environment supporting step(super_arm) -> (obs, rewards, done, info)
    L: int
        Horizon for this subroutine.
    pi_hat: list[int]
        Candidate super-arm to test.
    challenger: object
        Challenger algorithm B_pi_hat with .select(), .update(), .reset().
    beta1, beta2, beta3, beta4: floats
        Parameters for confidence and scheduling.
    T: int
        Global horizon (for computing log2 T bound).
    linear: bool
        If True, use linear reward aggregation.
    """
    def __init__(self, env, L, pi_hat, challenger, beta1, beta2, beta3, beta4, T, linear=True):
        self.env = env
        self.L = L
        self.pi_hat = pi_hat
        self.challenger = challenger
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.beta4 = beta4
        self.T = T
        self.linear = linear

    def RB(self, t, theta):
        """Confidence function RB(t, θ) = √(β1·t) + β2·θ + β3."""
        return np.sqrt(self.beta1 * t) + self.beta2 * theta + self.beta3

    def reward_agg(self, rewards):
        """Aggregate per-arm rewards into super-arm reward."""
        if self.linear:
            return np.sum(rewards)
        else:
            raise NotImplementedError("Non-linear aggregation not implemented")

    def run(self):
        """Original run() without regret recording."""
        # (Unchanged; see run_and_record for instrumented version.)
        return

    def run_and_record(self, mu_star):
        """
        Execute TwoModelSelect while recording per-round regret = mu_star - reward_t.

        Returns
        -------
        numpy.ndarray of regrets over at most L rounds.
        """
        # Initialize gap estimate and epoch length
        delta_hat = min(np.sqrt(self.beta4 / self.L), 1.0)
        M = int(np.ceil(self.beta4 / (delta_hat ** 2)))
        t = 1
        regrets = []

        # Phase 2 epochs
        max_epochs = int(np.floor(3 * np.log2(self.T)))
        for j in range(1, max_epochs + 1):
            t_j = t
            p_j = self.beta4 / (2 * M * (delta_hat ** 2))
            # # Reset challenger
            # if hasattr(self.challenger, 'reset'):
            #     self.challenger.reset()
            # Statistics
            sum0 = 0.0
            sum1 = 0.0
            count0 = 0
            count1 = 0

            # Play up to M rounds or until L
            while t <= t_j + M - 1 and t <= self.L:
                if np.random.rand() < p_j:
                    S = self.challenger.select()
                    _, rec, _, _ = self.env.step(S)
                    self.challenger.update(rec)
                    r = self.reward_agg(rec)
                    sum1 += r
                    count1 += 1
                else:
                    S = self.pi_hat
                    _, rec, _, _ = self.env.step(S)
                    r = self.reward_agg(rec)
                    sum0 += r
                    count0 += 1
                # record regret
                regrets.append(mu_star - r)
                t += 1

            # Compute empirical averages
            R0 = sum0 / (1 - p_j) if count0 > 0 else 0.0
            R1 = sum1 / p_j if count1 > 0 else 0.0
            duration = t - t_j
            theta_term = p_j * np.sqrt(self.beta1 * self.L / self.beta2)

            # Condition (11)
            if R0 <= R1 + 0.5 * duration * delta_hat - (5 / p_j) * self.RB(p_j * duration, theta_term):
                delta_hat /= 1.25
                M = int(np.ceil(self.beta4 / (delta_hat ** 2)))
                continue
            # Condition (12)
            if R0 >= R1 + 3 * M * delta_hat + 8 * np.sqrt(self.beta1 * self.L):
                delta_hat *= 1.25
                M = int(np.ceil(self.beta4 / (delta_hat ** 2)))
                continue
            # Termination
            if delta_hat < min(np.sqrt(self.beta4 / self.L), 1.0):
                break
            # Update M for next epoch
            M = int(np.ceil(2 * (t - t_j) + self.beta4 / (delta_hat ** 2)))

        return np.array(regrets)
