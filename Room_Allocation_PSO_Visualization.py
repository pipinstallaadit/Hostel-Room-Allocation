import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

num_runs_to_display = 5

g_best_matrix = np.load(os.path.join('npy Files', 'PSO_RUN_G_Status.npy'))

num_iter, num_runs = g_best_matrix.shape

iters = np.arange(num_iter)
selected_runs = np.argsort(g_best_matrix[-1])[:num_runs_to_display]

# initialize figure to be used for plotting
fig, ax = plt.subplots(1, 1)

figManager = plt.get_current_fig_manager()
figManager.resize(1600, 900)   # manually set window size


# initialising the global best vs iteration plot
g_lines = [ ax.plot(iters[:1], g_best_matrix[:1, i], label=str(i))[0] for i in selected_runs ]

# setting the axes limits
ax.set_xlim([1, num_iter])
ax.set_ylim([500, 1000])

# set axes labels
ax.set_xlabel('Iteration #', fontsize=15)
ax.set_ylabel('Global Best Value', fontsize=15)

# set scale
ax.set_xscale('log')

# display grid
ax.minorticks_on()
# display grid
ax.minorticks_on()

ax.grid(which='minor', grid_color='r', grid_linestyle='--', grid_linewidth=0.5)
ax.grid(which='major', grid_color='grey', grid_linewidth=1)


# set label
# ax.legend()

# set title
ax.set_title('Evolution of Global Best with Iterations', fontsize=15)

def animate(j) :

    for i, g_line in zip(selected_runs, g_lines) :

        # print(type(g_line))
        g_line.set_data(iters[:j+1], g_best_matrix[:j+1, i])

    return g_lines

anim = FuncAnimation(fig, animate, frames=int(num_iter//5), interval = 10 / num_iter, blit=True)

plt.show()

ch = input('Save ?\n')

if ch == 'y' :
    # save the animation
    print('Saving...')
    anim.save(os.path.join('mp4 Files', 'PSO_Original.mp4'), writer = 'ffmpeg', fps = 30)
    print('Done')