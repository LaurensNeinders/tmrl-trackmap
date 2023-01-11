import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splprep, splev

l_x,l_z = np.loadtxt('../saved_tracks/sm_track/track_left.csv', delimiter=',')
r_x,r_z = np.loadtxt('../saved_tracks/sm_track/track_right.csv', delimiter=',')

total_points = 60 # points per side of the road in total
plt.scatter(l_x.tolist()+r_x.tolist(),l_z.tolist()+r_z.tolist())
plt.show()
# calculate a curve through the left-sided points
points_left = [l_x] + [l_z]
tck, u = splprep(points_left, u=None, s=0.5)
u_new = np.linspace(u.min(), u.max(), total_points)
x_new, y_new = splev(u_new, tck, der=0)

# plt.plot(l_x, l_z, 'ro')
# plt.plot(x_new, y_new, 'b--')


# calculate a curve through the right-sided points
points_right = [r_x] + [r_z]
tck, u = splprep(points_right, u=None, s=1)
u_new = np.linspace(u.min(), u.max(), total_points)
x_new2, y_new2 = splev(u_new, tck, der=0)
print(u.min(), u.max())

np.savetxt('../saved_tracks/sm_track/track_left_smooth_small.csv', [x_new,y_new], delimiter=',')
np.savetxt('../saved_tracks/sm_track/track_right_smooth_small.csv', [x_new2,y_new2], delimiter=',')


x_new = x_new.tolist()
y_new = y_new.tolist()
x_new2 = x_new2.tolist()
y_new2 = y_new2.tolist()


plt.plot(r_x, r_z, 'ro')
plt.plot(x_new2, y_new2, 'b--')
plt.plot(l_x, l_z, 'ro')
plt.plot(x_new, y_new, 'b--')
plt.show()

plt.plot(x_new2[:100] + x_new[:100], y_new2[:100] + y_new[:100], 'ro')
plt.show()




