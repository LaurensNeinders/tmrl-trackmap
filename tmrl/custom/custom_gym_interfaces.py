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

coordinates_x = []
coordinates_z = []
colormap = []

# Interface for Trackmania 2020 ========================================================================================


def get_coordinates():
    return coordinates_x,coordinates_z,colormap

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
        Args:
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
        if self.resize_to is not None:
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
            h, w = self.resize_to
        else:
            h, w = cfg.WINDOW_HEIGHT, cfg.WINDOW_WIDTH
        if self.grayscale:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w))
        else:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w, 3))
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


class TM20InterfaceFull(TM2020Interface):
    """
    Full version of the CV-based interface
    This includes all observations, including the track progress
    Other interfaces don't derive from this one as it is more computationally demanding
    This interface is general-purpose and meant to be tweaked to your needs
    """
    def initialize(self):
        self.initialize_common()
        self.small_window = False
        self.initialized = True
        self.lidar = Lidar(self.window_interface.screenshot())
        self.lidar_hist = deque(maxlen=self.img_hist_len)

    def grab_data_and_img_and_lidar(self):
        img = self.window_interface.screenshot()[:, :, :3]
        lidar = self.lidar.lidar_20(img=img, show=False)
        img = img[:, :, ::-1]  # reversed view for numpy RGB convention
        data = self.client.retrieve_data()
        self.img = img  # for render()
        return data, img, lidar

    def reset(self):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        data, img, lidar = self.grab_data_and_img_and_lidar()
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
            self.lidar_hist.append(lidar)
        imgs = np.array(list(self.img_hist))
        lids = np.array(list(self.lidar_hist))
        progress = np.array([0], dtype='float32')
        obs = [speed, progress, gear, rpm, imgs, lids]
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        data, img, lidar = self.grab_data_and_img_and_lidar()
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
        progress = np.array([self.reward_function.cur_idx / self.reward_function.datalen], dtype='float32')
        self.img_hist.append(img)
        self.lidar_hist.append(lidar)
        imgs = np.array(list(self.img_hist))
        lids = np.array(list(self.lidar_hist))
        obs = [speed, progress, gear, rpm, imgs, lids]
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
        progress = spaces.Box(low=0.0, high=1.0, shape=(1,))
        gear = spaces.Box(low=0.0, high=6, shape=(1, ))
        rpm = spaces.Box(low=0.0, high=np.inf, shape=(1, ))
        imgs = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, cfg.WINDOW_HEIGHT, cfg.WINDOW_WIDTH, 3))
        lids = spaces.Box(low=0.0, high=np.inf, shape=(self.img_hist_len, 19))
        return spaces.Tuple((speed, progress, gear, rpm, imgs, lids))


class TM2020InterfaceLidar(TM2020Interface):
    def __init__(self, img_hist_len=1, gamepad=False, min_nb_steps_before_failure=int(20 * 3.5), record=False, save_replay: bool = False):
        super().__init__(img_hist_len, gamepad, min_nb_steps_before_failure, save_replay)
        self.record = record
        self.window_interface = None
        self.lidar = None


    def grab_lidar_speed_and_data(self):
        img = self.window_interface.screenshot()[:, :, :3]
        data = self.client.retrieve_data()
        speed = np.array([
            data[0],
        ], dtype='float32')
        lidar = self.lidar.lidar_20(img=img, show=False)
        # print("data:", data)
        # print("lidar:", lidar)
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


class TM2020InterfaceLidarTrackMap(TM2020InterfaceLidar):
    def __init__(self, img_hist_len=1, gamepad=False, min_nb_steps_before_failure=int(20 * 3.5), record=False, save_replay: bool = False):
        super().__init__(img_hist_len, gamepad, min_nb_steps_before_failure, save_replay)
        self.record = record
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

    def get_obs_rew_terminated_info(self):

        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """


        img, speed, data = self.grab_lidar_speed_and_data() # img is lidar



        # print("pitch: ", pitch)
        # print("CamPitch",np.rad2deg(np.arcsin(pitch)))
        #Distance calculations
        lidar_arr = np.array(img)

        # variables

        filter = True
        pitch_roll_adjustment = True
        ground_height = 10

        yaw = data[11] # get the angle which the car is facing
        pitch = data[12]
        roll = data[13]

        cam_height = data[16]-ground_height

        filter_list = np.full((len(lidar_arr),),True)
        angles = []
        for angle in range(-90,100, 10):
            angles.append(angle)
        cosine = np.around(np.cos(np.deg2rad(angles)), decimals=5) # the cosine of the angle of a line is neccesary to determine how much the line is forward


        distances_pixels = np.array(lidar_arr)+10 # lidar_arr is that array of lidar observations, +20 is because a line only starts 20 pixels from the starting position, so that needs to be corrected.
        distances_pixels_horizontal_temp = distances_pixels*np.sin(np.deg2rad(angles)) # the amount of pixels in the horizontal axis of a line
        # print("distance pixels" ,distances_pixels[3])
        # print("angle",angles[3])
        if pitch_roll_adjustment:
            distances_pixels_vertical = distances_pixels * cosine + (np.tan(roll)*distances_pixels_horizontal_temp)# amount of pixels in the vertical axis of a line adjusted for roll
            distances_pixels = distances_pixels_vertical/cosine
        else:
            distances_pixels_vertical = distances_pixels * cosine # amount of pixels in the vertical axis of a line
        # print("distance vert pix: ", distances_pixels_vertical[3])


        distances_pixels_horizontal = distances_pixels*np.sin(np.deg2rad(angles)) # the amount of pixels in the horizontal axis of a line
        # print("distance hor pix:", distances_pixels_horizontal[3])
        # horizontal distances
        # angles_2 = (distances_pixels_horizontal/958)*152.1 # distances_pixels_horizontal/958 is the proportion of the screen that the horizontal part of the line takes, *152.1 is the horizontal FOV of the camera
        angles_2 = (distances_pixels_horizontal/958)*152.1 # distances_pixels_horizontal/958 is the proportion of the screen that the horizontal part of the line takes, *152.1 is the horizontal FOV of the camera

        if pitch_roll_adjustment:
            # distances_3d_space_vertical = cam_height *(np.tan(np.deg2rad((((distances_pixels_vertical+49-2)/488)*80)+50+(np.rad2deg(np.arcsin(pitch)))))) # The vertical distance in 3d space of a line
            # distances_3d_space_vertical = cam_height *(np.tan(np.deg2rad((((distances_pixels_vertical+23.626)/440.2)*80)+50-(np.rad2deg(np.arcsin(pitch)))))) # The vertical distance in 3d space of a line
            distances_3d_space_vertical = cam_height *(np.tan(np.deg2rad((((distances_pixels_vertical+21.5)/434)*80)+50+(np.rad2deg(np.arcsin(pitch)))))) # The vertical distance in 3d space of a line

        else:
            distances_3d_space_vertical = 1.514 *(np.tan(np.deg2rad((((distances_pixels_vertical+49-2)/488)*80)+50))) # The vertical distance in 3d space of a line

        print("adjusted: ",((((distances_pixels_vertical+51.7)/488)*80)+50-(np.rad2deg(np.arcsin(pitch))))[10])
        # print("not adjusted: ",((((distances_pixels_vertical+51.7)/488)*80)+50)[9])
        distances_3d_space_horizontal = distances_3d_space_vertical*np.tan(np.deg2rad(angles_2)) # the horizontal distance in 3d space of a line
        distances_3d_space = distances_3d_space_vertical/np.cos(np.deg2rad(angles_2)) # the distance of a line (both horizontal and vertical)
        # print("horizontal distance space",distances_3d_space_horizontal[3])
        # print("angles_2 ", angles_2)
        print("vertical",distances_3d_space_vertical[9])
        print("pixels total length", distances_pixels[9])
        print("actual_length:", 701.18-data[17])
        # print("angles_2", angles_2)
        # print(distances_3d_space_vertical[10])
        # position calculations

        # position of the car
        cam_pos_x = data[15]
        cam_pos_z = data[17]
        pos_player_x = data[2]
        pos_player_z = data[4]


        # vertical position change
        pos_change_vert_x = np.sin(yaw)*distances_3d_space_vertical
        pos_change_vert_z = np.cos(yaw)*distances_3d_space_vertical

        # horizontal position change
        pos_change_hor_x = np.cos(yaw)*distances_3d_space_horizontal
        pos_change_hor_z = np.sin(yaw)*distances_3d_space_horizontal

        # calculate final position
        pos_lidar_x = cam_pos_x + pos_change_vert_x + pos_change_hor_x
        pos_lidar_z = cam_pos_z + pos_change_vert_z - pos_change_hor_z


        # filter out some coordinates that are not accurate for making a map


        if filter:
            # filter out the lidar rays that hit the side of the window
            for i,distance in enumerate(distances_pixels_horizontal):
                if distance > 440 or distance < -440:
                    filter_list[i] = False
            # filter out the lidar rays that hit too far away or negative
            for i, distance in enumerate(distances_3d_space):
                if distance > 40 or distance < 5:
                    filter_list[i] = False
            filter_list[0] = False
            filter_list[1] = False
            filter_list[2] = False
            filter_list[16] = False
            filter_list[17] = False
            filter_list[18] = False
            filter_list[9] = False

        filtered_colormap = np.arange(len(lidar_arr))[np.array(filter_list)]

        filtered_x = np.array(pos_lidar_x)[np.array(filter_list)]
        filtered_z = np.array(pos_lidar_z)[np.array(filter_list)]

        filtered_x = np.append(filtered_x,pos_player_x)
        filtered_z = np.append(filtered_z,pos_player_z)
        filtered_colormap = np.append(filtered_colormap, 19)

        filtered_x = np.append(filtered_x,cam_pos_x)
        filtered_z = np.append(filtered_z,cam_pos_z)
        filtered_colormap = np.append(filtered_colormap, 20)


        # Store all coordinates in a global coordinates list
        coordinates_x.extend(filtered_x)
        coordinates_z.extend(filtered_z)
        colormap.extend(filtered_colormap)
        # coordinates_x.append(pos_player_x)
        # coordinates_z.append(pos_player_z)
        # colormap.append(0)
        # coordinates_x.append((pos_player_x+pos_change_cam_x))
        # coordinates_z.append((pos_player_z+pos_change_cam_z))
        # colormap.append(1)



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







if __name__ == "__main__":
    pass