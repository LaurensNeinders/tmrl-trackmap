import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splprep, splev

# place that the recorded track was saved
l_x,l_z = np.loadtxt('../saved_tracks/tmrl-test/track_left.csv', delimiter=',')
r_x,r_z = np.loadtxt('../saved_tracks/tmrl-test/track_right.csv', delimiter=',')

# settings
total_points = 1000 # points per side of the road in total

plt.scatter(l_x.tolist()+r_x.tolist(),l_z.tolist()+r_z.tolist())
plt.show()

# calculate a curve through the left-sided points
points_left = [l_x] + [l_z]
tck, u = splprep(points_left, u=None, s=0.5)
u_new = np.linspace(u.min(), u.max(), total_points)
x_new, y_new = splev(u_new, tck, der=0)

# calculate a curve through the right-sided points
points_right = [r_x] + [r_z]
tck, u = splprep(points_right, u=None, s=1)
u_new2 = np.linspace(u.min(), u.max(), total_points)
x_new2, y_new2 = splev(u_new2, tck, der=0)

# when the below 2 lines are not commented out, the smoothed out track will save at that position
# np.savetxt('../saved_tracks/sm_track/track_left_smooth_small.csv', [x_new,y_new], delimiter=',')
# np.savetxt('../saved_tracks/sm_track/track_right_smooth_small.csv', [x_new2,y_new2], delimiter=',')







