import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style('darkgrid')


task_list = ['Ant', 'Point', 'Cheetah', 'Humanoid']
names = ('XY', 'Slow feature', 'Full obs')

xy = [0.4, 0.5, 0, 0]
slow = [0.76, 0, 0, 0]
full = [0.97, 0.98, 0, 0]

x = list(np.arange(len(task_list)))
total_width, n = 0.8, len(names)
width = total_width / n

plt.bar(x, xy, width=width, label=names[0])
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, slow, width=width, label=names[1], tick_label=task_list)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, full, width=width, label=names[2])
plt.legend()

plt.errorbar(np.arange(len(xy)), xy, yerr=[0.06, 0.052, 0.0, 0.0], ls='none', color='k', elinewidth=2, capsize=4)
plt.errorbar(np.arange(len(xy)) + width, slow, yerr=[0.119, 0.0, 0.0, 0.0], ls='none', color='k', elinewidth=2, capsize=4)
plt.errorbar(np.arange(len(xy)) + width * 2, full, yerr=[0.002, 0.003, 0.0, 0.0], ls='none', color='k', elinewidth=2, capsize=4)
plt.ylabel("Normalized Mean Distance to Goal")

plt.savefig("../data/rewards.pdf")
plt.show()