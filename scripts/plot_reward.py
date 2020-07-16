import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style('darkgrid')


names = ('XY', 'Slow feature', 'Full obs')

rewards = [0.40, 0.76, 0.97]

plt.bar(names, rewards)
plt.errorbar(np.arange(len(rewards)), rewards, yerr=[0.06, 0.119, 0.002], ls='none', color='k',elinewidth=2,capsize=4)
plt.ylabel("Normalized Mean Distance to Goal")

plt.savefig("../data/ant_reward.pdf")
plt.show()