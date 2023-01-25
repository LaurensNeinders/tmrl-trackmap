import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


l_x = np.loadtxt('../saved_tracks/tmrl-test/observed_run/observed_track_l_x_small.csv', delimiter=',').tolist()
l_z = np.loadtxt('../saved_tracks/tmrl-test/observed_run/observed_track_l_z_small.csv', delimiter=',').tolist()
r_x = np.loadtxt('../saved_tracks/tmrl-test/observed_run/observed_track_r_x_small.csv', delimiter=',').tolist()
r_z = np.loadtxt('../saved_tracks/tmrl-test/observed_run/observed_track_r_z_small.csv', delimiter=',').tolist()
# pos = np.loadtxt('../saved_tracks/tmrl-test/observed_run/observed_track_car_pos.csv', delimiter=',').tolist()
# l_x,l_z = np.loadtxt('../saved_tracks/tmrl-test/observed_run/track_left_smooth_small.csv', delimiter=',').tolist()
# r_x,r_z = np.loadtxt('../saved_tracks/tmrl-test/observed_run/track_right_smooth_small.csv', delimiter=',').tolist()
# pos = np.loadtxt('../saved_tracks/observed_track_car_pos1.csv', delimiter=',').tolist()

# print(pos)


fig = plt.figure()
plt.xlim(-400, 400)
plt.ylim(-300, 400)
# plt.xlim(200, 850)
# plt.ylim(400, 850)
graph, = plt.plot([], [], 'o')

def animate(i):
    # graph.set_data(l_x[i+1] + r_x[i+1] + [pos[i+1][0]] , l_z[i+1] + r_z[i+1] + [pos[i+1][1]])
    graph.set_data(l_x[i+1] + r_x[i+1], l_z[i+1] + r_z[i+1])
    # graph.set_data(l_x + r_x + [pos[i+1][0]], l_z + r_z + [pos[i+1][1]])
    # graph.set_data(pos[i+1])
    return graph

ani = FuncAnimation(fig, animate, frames=len(l_x)-1, interval=25)
plt.show()
