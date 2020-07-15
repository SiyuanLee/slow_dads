import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

file_dir = None
f = open(file_dir, 'rb')
total_traj = pkl.load(f)

xlim = 5
plt.xlim(-xlim, xlim)
plt.ylim(-xlim, xlim)
color_map = ['b', 'g', 'r', 'c', 'm', 'y']
style_map = []
for line_style in ['-', '--', '-.', ':']:
    style_map += [color + line_style for color in color_map]

def select():
    index = 0
    num = 20
    # red
    # select = [3, 11, 17]
    # green
    # select = [1, 3, 11]
    # blue
    select = [1, 5, 19]

    for i in range(num):
        if i in select:
            plt.plot(total_traj[index][i][:, 0], total_traj[index][i][:, 1],
                     style_map[i % len(style_map)],
                     label=str(i))
    plt.legend(loc='best')
    plt.show()

def save():
    Select_traj = []
    Select = [[1, 5, 19], [1, 3, 11], [3, 11, 17]]
    for index, select_idx in enumerate(Select):
        select_traj = []
        for i in select_idx:
            select_traj.append(total_traj[index][i].copy())
            plt.plot(total_traj[index][i][:, 0], total_traj[index][i][:, 1],
                     style_map[index % len(style_map)],
                     label=str(i))
        Select_traj.append(select_traj.copy())
    print("Select_traj", len(Select_traj))
    plt.savefig("../data/slow_trajectory.pdf")
    f = open("../data/separ_no_norm_traj.pkl", 'wb')
    pkl.dump(Select_traj, f)

save()



