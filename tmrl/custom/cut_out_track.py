from scipy import spatial
import numpy as np
from matplotlib import pyplot as plt

map_left = np.loadtxt('../saved_tracks/track_left_smooth_small.csv', delimiter=',')
map_right = np.loadtxt('../saved_tracks/track_right_smooth_small.csv', delimiter=',')

coordinates = [ 536.2958984375 , 661.7255249023438 ]

look_ahead_distance = 25 # points out of 1000 for total track

nearby_correction = 60

# Find point that is closest to the car, from all the points, both left and right side
entire_map = map_left.T.tolist()+map_right.T.tolist()
tree = spatial.KDTree(entire_map)
(_, i) = tree.query(coordinates)
if i < len(map_left.T): # if the closest point is on the left side
    print("left side is closer")

    i_l = i # this index is the index for the closest point on the left side of the track
    i_l_min = i_l
    i_l_max = min(i_l + look_ahead_distance, len(map_left.T)-1)
    # find the nearest point on the right side of the track, but look for only nearby points
    j_min = max(i_l-nearby_correction,0) # lower bound
    j_max = min(i_l+nearby_correction,len(map_left.T)-1) # upper bound
    tree_r = spatial.KDTree(map_right.T[j_min:j_max]) # look up the index of the closest point
    (_, i_r_min) = tree_r.query(map_left.T[i_l])
    i_r_min = i_r_min + j_min

    #calculate the endpoint for the other side of the track
    j_min = max(i_l+look_ahead_distance-nearby_correction,0) # lower bound
    j_max = min(i_l+look_ahead_distance+nearby_correction,len(map_left.T)-1) # upper bound
    tree_r_far = spatial.KDTree(map_right.T[j_min:j_max]) # look up the index of the closest point
    (_, i_r_max) = tree_r_far.query(map_left.T[i_l_max])
    i_r_max = i_r_max + j_min

else:
    print("right side is closer")
    i_r = i-len(map_left.T) # this index is the index for the closest point on the right side of the track
    i_r_min = i_r
    i_r_max = i_l_max = min(i_r + look_ahead_distance, len(map_right.T)-1)
    # find the nearest point on the left side of the track, but look for only nearby points
    j_min = max(i_r-nearby_correction,0) # lower bound
    j_max = min(i_r+nearby_correction,len(map_right.T)-1) # upper bound
    tree_l = spatial.KDTree(map_left.T[j_min:j_max]) # look up the index of the closest point
    (_, i_l_min) = tree_l.query(map_right.T[i_r])
    i_l_min = i_l_min + j_min

    #calculate the endpoint for the other side of the track
    j_min = max(i_r+look_ahead_distance-nearby_correction,0) # lower bound
    j_max = min(i_r+look_ahead_distance+nearby_correction,len(map_right.T)-1) # upper bound
    tree_l_far = spatial.KDTree(map_left.T[j_min:j_max]) # look up the index of the closest point
    (_, i_l_max) = tree_l_far.query(map_right.T[i_r_max])
    i_l_max = i_l_max + j_min


#make a nice graph
plt.plot(map_left[0][i_l_min:i_l_max],map_left[1][i_l_min:i_l_max])
# print(i_r_min)
plt.plot(map_right[0][i_r_min:i_r_max],map_right[1][i_r_min:i_r_max])
plt.scatter([coordinates[0]],[coordinates[1]])
plt.show()


