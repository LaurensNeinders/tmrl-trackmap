import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# l_x = np.loadtxt('../saved_tracks/observed_tracks/observed_track_l_x1_small.csv', delimiter=',').tolist()
# l_z = np.loadtxt('../saved_tracks/observed_tracks/observed_track_l_z1_small.csv', delimiter=',').tolist()
# r_x = np.loadtxt('../saved_tracks/observed_tracks/observed_track_r_x1_small.csv', delimiter=',').tolist()
# r_z = np.loadtxt('../saved_tracks/observed_tracks/observed_track_r_z1_small.csv', delimiter=',').tolist()
l_x = np.loadtxt('../saved_tracks/observed_tracks/observed_track_l_x1_small.csv', delimiter=',').tolist()
l_z = np.loadtxt('../saved_tracks/observed_tracks/observed_track_l_z1_small.csv', delimiter=',').tolist()
r_x = np.loadtxt('../saved_tracks/observed_tracks/observed_track_r_x1_small.csv', delimiter=',').tolist()
r_z = np.loadtxt('../saved_tracks/observed_tracks/observed_track_r_z1_small.csv', delimiter=',').tolist()
pos = np.loadtxt('../saved_tracks/observed_track_car_pos1.csv', delimiter=',').tolist()



fig = plt.figure()
plt.xlim(-400, 400)
plt.ylim(-300, 400)
graph, = plt.plot([], [], 'o')

def animate(i):
    # graph.set_data(l_x[i+1] + r_x[i+1] , l_z[i+1] + r_z[i+1])
    graph.set_data(pos[i+1])
    return graph

ani = FuncAnimation(fig, animate, frames=len(pos)-1, interval=25)
plt.show()
