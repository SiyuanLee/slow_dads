import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("darkgrid")

def mean_error(Traj_std, name=None):
    Traj_std_mean = np.mean(Traj_std, axis=0)
    error_bar = np.var(Traj_std, axis=0)
    plt.plot(Traj_std_mean, label=name)
    plt.fill_between(np.arange(len(error_bar)), Traj_std_mean - error_bar, Traj_std_mean + error_bar, alpha=0.2)
                     # facecolor='blue')
    return Traj_std_mean, error_bar


file_dir = '../data/ant_std/full_std.pkl'
f = open(file_dir, 'rb')
full_std = pkl.load(f)
full_mean, full_error = mean_error(full_std, name="full")

file_dir = '../data/ant_std/xy_std.pkl'
f = open(file_dir, 'rb')
xy_std = pkl.load(f)
xy_mean, xy_error = mean_error(xy_std, name="XY")

file_dir = '../data/ant_std/slow_std.pkl'
f = open(file_dir, 'rb')
slow_std = pkl.load(f)
slow_mean, slow_error = mean_error(slow_std, name="slow feature")

plt.xlabel("Steps in the Environment")
plt.ylabel("Standard Deviation")
plt.title("Standard Deviation of Trajectories")

plt.legend(loc=0)
plt.savefig("../data/ant_std/ant_std.pdf")
plt.show()