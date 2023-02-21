import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splprep, splev

l_x,l_z = np.loadtxt('../saved_tracks/tmrl-test/track_left_smooth.csv', delimiter=',')
r_x,r_z = np.loadtxt('../saved_tracks/tmrl-test/track_right_smooth.csv', delimiter=',')
pos = np.loadtxt('../saved_tracks/tmrl-test/observed_run/observed_track_car_pos_human.csv',delimiter=',')

print(pos.T)
plt.plot(r_x, r_z, 'ro')
plt.plot(l_x, l_z, 'ro')
plt.plot(pos.T[0], pos.T[1],'bo')
plt.show()




