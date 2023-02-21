# rtgym interfaces for Trackmania

# standard library imports
import platform
import logging
import time
from collections import deque

# third-party imports
import cv2
import gym.spaces as spaces
import numpy as np
from keyboard import is_pressed
from scipy import spatial

# third-party imports
from rtgym import RealTimeGymInterface

# local imports
import tmrl.config.config_constants as cfg
from tmrl.custom.utils.compute_reward import RewardFunction
from tmrl.custom.utils.control_gamepad import control_gamepad, gamepad_reset, gamepad_close_finish_pop_up_tm20
from tmrl.custom.utils.control_keyboard import apply_control, keyres
from tmrl.custom.utils.control_mouse import mouse_close_finish_pop_up_tm20, mouse_save_replay_tm20
from tmrl.custom.utils.window import WindowInterface
from tmrl.custom.utils.tools import Lidar, TM2020OpenPlanetClient

# Globals ==============================================================================================================

NB_OBS_FORWARD = 500  # this allows (and rewards) 50m cuts

# Custom variables

map_left = np.loadtxt('saved_tracks/tmrl-test/track_left.csv', delimiter=',')
map_right = np.loadtxt('saved_tracks/tmrl-test/track_right.csv', delimiter=',')
all_observed_track_parts = [[],[],[],[],[]]

# Interface for Trackmania 2020 ========================================================================================


def get_all_observed_track_parts():
    return all_observed_track_parts

class TM2020Interface(RealTimeGymInterface):
    """
    This is the API needed for the algorithm to control Trackmania2020
    """
    def __init__(self,
                 img_hist_len: int = 4,
                 gamepad: bool = False,
                 min_nb_steps_before_failure: int = int(3.5 * 20),
                 save_replay: bool = False,
                 grayscale: bool = True,
                 resize_to=(64, 64),
                 finish_reward=cfg.REWARD_END_OF_TRACK,
                 constant_penalty=cfg.CONSTANT_PENALTY):
        """
        Base rtgym interface for TrackMania 2020 (Full environment)

        Args:
            img_hist_len: int: history of images that are part of observations
            gamepad: bool: whether to use a virtual gamepad for control
            min_nb_steps_before_failure: int: episode is done if not receiving reward during this nb of timesteps
            save_replay: bool: whether to save TrackMania replays
            grayscale: bool: whether to output grayscale images or color images
            resize_to: Tuple[int, int]: resize output images to this (width, height)
            finish_reward: float: reward when passing the finish line
            constant_penalty: float: constant reward given at each time-step
        """
        self.last_time = None
        self.img_hist_len = img_hist_len
        self.img_hist = None
        self.img = None
        self.reward_function = None
        self.client = None
        self.gamepad = gamepad
        self.j = None
        self.window_interface = None
        self.small_window = None
        self.min_nb_steps_before_failure = min_nb_steps_before_failure
        self.save_replay = save_replay
        self.grayscale = grayscale
        self.resize_to = resize_to
        self.finish_reward = finish_reward
        self.constant_penalty = constant_penalty

        self.initialized = False

    def initialize_common(self):
        if self.gamepad:
            assert platform.system() == "Windows", "Sorry, Only Windows is supported for gamepad control"
            import vgamepad as vg
            self.j = vg.VX360Gamepad()
            logging.debug(" virtual joystick in use")
        self.window_interface = WindowInterface("Trackmania")
        self.window_interface.move_and_resize()
        self.last_time = time.time()
        self.img_hist = deque(maxlen=self.img_hist_len)
        self.img = None
        self.reward_function = RewardFunction(reward_data_path=cfg.REWARD_PATH,
                                              nb_obs_forward=NB_OBS_FORWARD,
                                              nb_obs_backward=10,
                                              nb_zero_rew_before_failure=10,
                                              min_nb_steps_before_failure=self.min_nb_steps_before_failure)
        self.client = TM2020OpenPlanetClient()

    def initialize(self):
        self.initialize_common()
        self.small_window = True
        self.initialized = True

    def send_control(self, control):
        """
        Non-blocking function
        Applies the action given by the RL policy
        If control is None, does nothing (e.g. to record)
        Args:
            control: np.array: [forward,backward,right,left]
        """
        if self.gamepad:
            if control is not None:
                control_gamepad(self.j, control)
        else:
            if control is not None:
                actions = []
                if control[0] > 0:
                    actions.append('f')
                if control[1] > 0:
                    actions.append('b')
                if control[2] > 0.5:
                    actions.append('r')
                elif control[2] < -0.5:
                    actions.append('l')
                apply_control(actions)

    def grab_data_and_img(self):
        img = self.window_interface.screenshot()[:, :, :3]  # BGR ordering
        if self.resize_to is not None:  # cv2.resize takes dim as (width, height)
            img = cv2.resize(img, self.resize_to)
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = img[:, :, ::-1]  # reversed view for numpy RGB convention
        data = self.client.retrieve_data()
        self.img = img  # for render()
        return data, img

    def reset_race(self):
        if self.gamepad:
            gamepad_reset(self.j)
        else:
            keyres()

    def reset_common(self):
        if not self.initialized:
            self.initialize()
        self.send_control(self.get_default_action())
        self.reset_race()
        time_sleep = max(0, cfg.SLEEP_TIME_AT_RESET - 0.1) if self.gamepad else cfg.SLEEP_TIME_AT_RESET
        time.sleep(time_sleep)  # must be long enough for image to be refreshed

    def reset(self):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        data, img = self.grab_data_and_img()
        speed = np.array([
            data[0],
        ], dtype='float32')
        gear = np.array([
            data[9],
        ], dtype='float32')
        rpm = np.array([
            data[10],
        ], dtype='float32')
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [speed, gear, rpm, imgs]
        self.reward_function.reset()
        return obs, {}

    def close_finish_pop_up_tm20(self):
        if self.gamepad:
            gamepad_close_finish_pop_up_tm20(self.j)
        else:
            mouse_close_finish_pop_up_tm20(small_window=self.small_window)

    def wait(self):
        """
        Non-blocking function
        The agent stays 'paused', waiting in position
        """
        self.send_control(self.get_default_action())
        self.reset_race()
        time.sleep(0.5)
        self.close_finish_pop_up_tm20()

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        data, img = self.grab_data_and_img()
        speed = np.array([
            data[0],
        ], dtype='float32')
        gear = np.array([
            data[9],
        ], dtype='float32')
        rpm = np.array([
            data[10],
        ], dtype='float32')
        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [speed, gear, rpm, imgs]
        end_of_track = bool(data[8])
        info = {}
        if end_of_track:
            terminated = True
            rew += self.finish_reward
            if self.save_replay:
                mouse_save_replay_tm20(True)
        rew += self.constant_penalty
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        gear = spaces.Box(low=0.0, high=6, shape=(1, ))
        rpm = spaces.Box(low=0.0, high=np.inf, shape=(1, ))
        if self.resize_to is not None:
            w, h = self.resize_to
        else:
            w, h = cfg.WINDOW_HEIGHT, cfg.WINDOW_WIDTH
        if self.grayscale:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w))  # cv2 grayscale images are (h, w)
        else:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w, 3))  # cv2 images are (h, w, c)
        return spaces.Tuple((speed, gear, rpm, img))

    def get_action_space(self):
        """
        must return a Box
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(3, ))

    def get_default_action(self):
        """
        initial action at episode start
        """
        return np.array([0.0, 0.0, 0.0], dtype='float32')


class TM2020InterfaceLidar(TM2020Interface):
    def __init__(self, img_hist_len=1, gamepad=False, min_nb_steps_before_failure=int(20 * 3.5), save_replay: bool = False):
        super().__init__(img_hist_len, gamepad, min_nb_steps_before_failure, save_replay)
        self.window_interface = None
        self.lidar = None

    def grab_lidar_speed_and_data(self):
        img = self.window_interface.screenshot()[:, :, :3]
        data = self.client.retrieve_data()
        speed = np.array([
            data[0],
        ], dtype='float32')
        lidar = self.lidar.lidar_20(img=img, show=False)
        return lidar, speed, data

    def initialize(self):
        super().initialize_common()
        self.small_window = False
        self.lidar = Lidar(self.window_interface.screenshot())
        self.initialized = True

    def reset(self):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        img, speed, data = self.grab_lidar_speed_and_data()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        img, speed, data = self.grab_lidar_speed_and_data() # img is lidar
        car_position = [data[2],data[4]]
        all_observed_track_parts[4].append(car_position)
        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]])) # data[2-4] are the position, from that the reward is computed
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        end_of_track = bool(data[8])
        info = {}
        if end_of_track:
            rew += self.finish_reward
            terminated = True
            if self.save_replay:
                mouse_save_replay_tm20()
        rew += self.constant_penalty
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        imgs = spaces.Box(low=0.0, high=np.inf, shape=(
            self.img_hist_len,
            19,
        ))  # lidars
        return spaces.Tuple((speed, imgs))


class TM2020InterfaceLidarProgress(TM2020InterfaceLidar):

    def reset(self):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        img, speed, data = self.grab_lidar_speed_and_data()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        progress = np.array([0], dtype='float32')
        obs = [speed, progress, imgs]
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        img, speed, data = self.grab_lidar_speed_and_data()
        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        progress = np.array([self.reward_function.cur_idx / self.reward_function.datalen], dtype='float32')
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, progress, imgs]
        end_of_track = bool(data[8])
        info = {}
        if end_of_track:
            rew += self.finish_reward
            terminated = True
            if self.save_replay:
                mouse_save_replay_tm20()
        rew += self.constant_penalty
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        progress = spaces.Box(low=0.0, high=1.0, shape=(1,))
        imgs = spaces.Box(low=0.0, high=np.inf, shape=(
            self.img_hist_len,
            19,
        ))  # lidars
        return spaces.Tuple((speed, progress, imgs))


class TM2020InterfaceTrackMap(TM2020InterfaceLidar):
    def __init__(self, img_hist_len=1, gamepad=False, min_nb_steps_before_failure=int(20 * 3.5), record=False, save_replay: bool = False):
        super().__init__(img_hist_len, gamepad, min_nb_steps_before_failure, save_replay)
        self.record = record
        self.window_interface = None
        self.lidar = None
        self.last_pos = [0,0]
        self.index = 0

    def get_observation_space(self):
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        gear = spaces.Box(low=0.0, high=6, shape=(1, ))
        rpm = spaces.Box(low=0.0, high=np.inf, shape=(1, ))
        track_information = spaces.Box(low=-300,high=300, shape=(60,))
        acceleration = spaces.Box(low=-100, high=100.0, shape=(1, ))
        steering_angle = spaces.Box(low=-1, high=1.0, shape=(1, ))
        slipping_tires = spaces.Box(low=0.0, high=1, shape=(4,))
        crash = spaces.Box(low=0.0, high=1, shape=(1, ))
        return spaces.Tuple((speed,gear,rpm,track_information,acceleration,steering_angle,slipping_tires,crash))

    def grab_data(self):
        data = self.client.retrieve_data()
        return data

    def get_obs_rew_terminated_info(self):

        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """


        data = self.grab_data()

        car_position = [data[2],data[4]]
        yaw = data[11]      # angle the car is facing

        self.last_pos = car_position
        # retrieving map information --------------------------------------
        # Cut out a portion directly in front of the car, as input for the ai
        look_ahead_distance = 15 # points to look ahead on the track
        nearby_correction = 60 # one point on a side needs to be at least this close to the same point on the other side
        l_x, l_z, r_x, r_z = self.get_track_in_front(car_position, look_ahead_distance, nearby_correction)


        #normalize the track in front

        l_x, l_z, r_x, r_z = self.normalize_track(l_x, l_z, r_x, r_z,car_position,yaw)
        # save the track in front in a file, so we can play it back later
        all_observed_track_parts[0].append(l_x.tolist())
        all_observed_track_parts[1].append(l_z.tolist())
        all_observed_track_parts[2].append(r_x.tolist())
        all_observed_track_parts[3].append(r_z.tolist())
        all_observed_track_parts[4].append(car_position)
        # ----------------------------------------------------------------------


        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]])) # data[2-4] are the position, from that the reward is computed
        track_information = np.array(np.append(np.append(l_x,r_x),np.append(l_z,r_z)), dtype='float32')
        speed = np.array([
            data[0],
        ], dtype='float32')
        gear = np.array([
            data[9],
        ], dtype='float32')
        rpm = np.array([
            data[10],
        ], dtype='float32')
        acceleration = np.array([
            data[18],
        ], dtype='float32')
        steering_angle = np.array([
            data[19],
        ], dtype='float32')
        slipping_tires = np.array(data[20:24], dtype='float32')
        crash = np.array([
            data[24],
        ], dtype='float32')
        obs = [speed, gear, rpm, track_information,acceleration,steering_angle,slipping_tires,crash]
        end_of_track = bool(data[8])
        info = {}
        crash_penalty = 0 # < 0 to give a penalty
        if crash == 1:
            rew += crash_penalty
        if end_of_track:
            rew += self.finish_reward
            terminated = True
            if self.save_replay:
                mouse_save_replay_tm20()
        rew += self.constant_penalty
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_track_in_front(self, car_position, look_ahead_distance, nearby_correction):
        # Find point that is closest to the car, from all the points, both left and right side
        entire_map = map_left.T.tolist()+map_right.T.tolist()
        tree = spatial.KDTree(entire_map)
        (_, i) = tree.query(car_position)
        if i < len(map_left.T): # if the closest point is on the left side
            # print("left side is closer")

            i_l = i # this index is the index for the closest point on the left side of the track
            i_l_min = i_l
            # find the nearest point on the right side of the track, but look for only nearby points
            j_min = max(i_l_min-nearby_correction,0) # lower bound
            j_max = min(i_l_min+nearby_correction,len(map_left.T)-1) # upper bound
            tree_r = spatial.KDTree(map_right.T[j_min:j_max]) # look up the index of the closest point on the other side of the track
            (_, i_r_min) = tree_r.query(map_left.T[i_l_min])
            i_r_min = i_r_min + j_min



            # #calculate the endpoint for the other side of the track
            # j_min = max(i_l+look_ahead_distance-nearby_correction,0) # lower bound
            # j_max = min(i_l+look_ahead_distance+nearby_correction,len(map_left.T)-1) # upper bound
            # tree_r_far = spatial.KDTree(map_right.T[j_min:j_max]) # look up the index of the closest point
            # (_, i_r_max) = tree_r_far.query(map_left.T[i_l_max])
            # i_r_max = i_r_max + j_min

        else:
            # print("right side is closer")
            i_r = i-len(map_left.T) # this index is the index for the closest point on the right side of the track
            i_r_min = i_r
            # find the nearest point on the left side of the track, but look for only nearby points
            j_min = max(i_r-nearby_correction,0) # lower bound
            j_max = min(i_r+nearby_correction,len(map_right.T)-1) # upper bound
            tree_l = spatial.KDTree(map_left.T[j_min:j_max]) # look up the index of the closest point
            (_, i_l_min) = tree_l.query(map_right.T[i_r])
            i_l_min = i_l_min + j_min

        i_l_max = i_l_min + look_ahead_distance
        i_r_max = i_r_min + look_ahead_distance



        extra = np.full((look_ahead_distance,2),map_left.T[-1])
        map_left_extended = np.append(map_left.T,extra,axis=0).T

        extra = np.full((look_ahead_distance,2),map_right.T[-1])
        map_right_extended = np.append(map_right.T,extra,axis=0).T

        l_x = map_left_extended[0][i_l_min:i_l_max]
        l_z = map_left_extended[1][i_l_min:i_l_max]
        r_x = map_right_extended[0][i_r_min:i_r_max]
        r_z = map_right_extended[1][i_r_min:i_r_max]
        return l_x, l_z, r_x, r_z

    def normalize_track(self, l_x, l_z, r_x, r_z,car_position,yaw):
        angle = yaw
        left = (np.array([l_x,l_z]).T-car_position).T
        right = (np.array([r_x,r_z]).T-car_position).T

        left_normal_x = left[0] * np.cos(angle) - left[1] * np.sin(angle)
        left_normal_y = left[0] * np.sin(angle) + left[1] * np.cos(angle)

        right_normal_x = right[0] * np.cos(angle) - right[1] * np.sin(angle)
        right_normal_y = right[0] * np.sin(angle) + right[1] * np.cos(angle)

        return left_normal_x,left_normal_y,right_normal_x,right_normal_y

    def reset(self):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        data = self.grab_data()
        track_information = np.full((60,), 0,dtype='float32')
        speed = np.array([
            data[0],
        ], dtype='float32')
        gear = np.array([
            data[9],
        ], dtype='float32')
        rpm = np.array([
            data[10],
        ], dtype='float32')

        acceleration = np.array([
            data[18],
        ], dtype='float32')
        steering_angle = np.array([
            data[19],
        ], dtype='float32')
        slipping_tires = np.array(data[20:24], dtype='float32')
        crash = np.array([
            data[24],
        ], dtype='float32')

        obs = [speed, gear, rpm, track_information, acceleration, steering_angle, slipping_tires, crash]
        self.reward_function.reset()
        return obs, {}

if __name__ == "__main__":
    pass
