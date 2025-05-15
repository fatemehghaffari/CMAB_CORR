import numpy as np
from BASIC import GCOBE
from CBARBAR import CBARBAR, simple_greedy_oracle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
                # if (means[ind] == self.bestarmrew) and (rewards[action.index(ind)] == 1):
                #     rewards[action.index(ind)] = 0
                #     self.C -= 1
        if corr_percent > 0:
            # Generate a mask indicating which bits to flip
            flip_mask = np.random.rand(len(rewards)) < corr_percent
            # Flip the bits: 1 becomes 0, 0 becomes 1
            rewards[flip_mask] = 1 - rewards[flip_mask]

        return None, rewards, True, {}

b_1 = 1
d = 3
T = 200000
C = 0
delta = 0.1
num_rounds = 10
GC = np.zeros([num_rounds, T])
CB = np.zeros([num_rounds, T])
# , 5000, 10000, 15000, 20000
for C in [100000]:
    for i in range(num_rounds):
        print("Round ", i)
        # --- Test parameters ---
        # means = np.random.rand(5).tolist()
        means = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        beta1 = len(means) * np.log(T) * d * b_1
        beta2 = (len(means) / d) * (np.log(T))
        beta3 = len(means)
        env = TestEnv(means, C)
        def oracle(weights, include=None):
            return simple_greedy_oracle(np.array(weights), d, include)
        # Instantiate and run GCOBE
        algo = GCOBE(oracle, env, T, delta, d, beta1, beta2, beta3, reward_function='linear')
        regrets = algo.gcobe()
        # Sanity checks
        assert len(regrets) == T, f"Expected {T} regrets, got {len(regrets)}"
        try:
            assert np.all(regrets >= 0), "Regrets should be non-negative"
        except:
            print("Neg Reg: ", regrets[regrets < 0])
        # Plot cumulative regret
        GC[i, :] = np.cumsum(regrets)

        print("GC done!")
        # oracle: pick top-d arms by means
        print(C)
        env = TestEnv(means, C)
        # run CBARBAR
        algo = CBARBAR(env, oracle, T, 1.0, 1.0, delta, d, reward_function = "linear")
        # algo = CBARBAR(env, oracle, T, alpha, beta, delta, d, reward_function = "linear")
        cum_rewards = algo.run()
        regs = algo.compute_regret()
        CB[i, :] = np.cumsum(regs)
        print("CB done!")
    np.save('C'+str(C)+'_MSCUCB_K5_d3_res4.npy', GC)
    np.save('C'+str(C)+'_CBARBAR_K5_d3_res4.npy', CB)
    GC_mean = np.mean(GC, axis=0)
    CB_mean = np.mean(CB, axis=0)

# GC = np.load('res2/C50000_MSCUCB_K5_C15000_res2.npy')
# CB = np.load('res2/C50000_CBARBAR_K5_C15000_res2.npy')

# T = 200000
# mean_cb = np.mean(CB, axis=0)
# std_cb  = np.std(CB,  axis=0)
# mean_gc = np.mean(GC, axis=0)
# std_gc  = np.std(GC,  axis=0)

# rounds = np.arange(1, T+1)

# # Placeholder settings
# LINEWIDTH       = 2.0
# SHADE_ALPHA     = 0.2
# FONTSIZE_AXIS   = 20
# FONTSIZE_TITLE  = 16
# FONTSIZE_LEGEND = 16
# X_LIMIT         = (0, T)
# Y_LIMIT         = (0, 40000)
# TITLE           = 'd = 4'
# XLABEL          = 'Round'
# YLABEL          = 'Cumulative Regret'

# fig, ax = plt.subplots(dpi=300)

# # Plot CB in blue
# ax.plot(rounds, mean_cb, color='blue', linewidth=LINEWIDTH, label=r'$\mathtt{CBARBAR-NL}$')
# ax.fill_between(rounds,
#                 mean_cb - std_cb,
#                 mean_cb + std_cb,
#                 color='blue',
#                 alpha=SHADE_ALPHA)

# # Plot GC in red
# ax.plot(rounds, mean_gc, color='red', linewidth=LINEWIDTH, label=r'$\mathtt{M}^{2}\mathtt{UCB}$')
# ax.fill_between(rounds,
#                 mean_gc - std_gc,
#                 mean_gc + std_gc,
#                 color='red',
#                 alpha=SHADE_ALPHA)

# ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

# ax.tick_params(axis='both', which='major', labelsize=16)

# # Major grid
# ax.grid(True, which='major', linewidth=1.0)
# ax.grid(True, which='minor', linewidth=0.5)
# ax.minorticks_on()
# # # Thinner horizontal lines at y=0.25 and y=0.75
# # for y in [0.25, 0.75]:
# #     ax.axhline(y, linestyle='--', linewidth=0.5, color='gray')

# # Labels, title, legend, limits
# # ax.set_title(TITLE, fontsize=FONTSIZE_TITLE)
# # Suppose you want x-ticks exactly at [0, 2.5, 5, 7.5, 10]:
# ax.set_yticks([0, 10000, 20000, 30000, 40000])
# # And y-ticks at [0, 0.5, 1.0]:
# ax.set_xticks([0, 50000, 100000, 150000, 200000])


# ax.set_xlabel(XLABEL, fontsize=FONTSIZE_AXIS)
# ax.set_ylabel(YLABEL, fontsize=FONTSIZE_AXIS)
# if X_LIMIT is not None:
#     ax.set_xlim(*X_LIMIT)
# if Y_LIMIT is not None:
#     ax.set_ylim(*Y_LIMIT)

# ax.legend(fontsize=FONTSIZE_LEGEND)
# plt.tight_layout()

# fig.savefig('final_2/d3_K5_C50000_res3_final.png',
#         dpi=300,               
#         bbox_inches='tight',   
#         pad_inches=0.02)       
# plt.show()

