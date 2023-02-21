# third-party imports
import gym
import cv2
import numpy as np
from rtgym.envs.real_time_env import DEFAULT_CONFIG_DICT

# local imports
from tmrl.custom.custom_gym_interfaces import TM2020Interface, TM2020InterfaceLidar, TM2020InterfaceTrackMap
from tmrl.custom.utils.window import WindowInterface
from tmrl.custom.utils.tools import Lidar
import tmrl.config.config_constants as cfg
import logging

# Custom imports
from tmrl.custom.custom_gym_interfaces import get_all_observed_track_parts
from matplotlib import pyplot as plt
from time import sleep
from scipy.interpolate import splprep, splev


def check_env_tm20_trackmap():
    window_interface = WindowInterface("Trackmania")
    lidar = Lidar(window_interface.screenshot())
    env_config = DEFAULT_CONFIG_DICT.copy()
    env_config["interface"] = TM2020InterfaceTrackMap
    env_config["wait_on_done"] = True
    env_config["interface_kwargs"] = {"img_hist_len": 1, "gamepad": False, "min_nb_steps_before_failure": int(20 * 60), "record": False}
    # env_config["time_step_duration"] = 0.5  # nominal duration of your time-step
    # env_config["start_obs_capture"] = 0.4
    env = gym.make("real-time-gym-v0", config=env_config)
    o, i = env.reset()
    rounds = 1200
    current=0
    while current < rounds:
        current+=1
        o, r, d, t, i = env.step(None)
        logging.info(f"r:{r}, d:{d}")

        if d or t:
            print("d: ",d)
            print("t: ",t)
            o, i = env.reset()
        img = window_interface.screenshot()[:, :, :3]
        lidar.lidar_20(img, True)


def check_env_tm20lidar():
    window_interface = WindowInterface("Trackmania")
    lidar = Lidar(window_interface.screenshot())
    env_config = DEFAULT_CONFIG_DICT.copy()
    env_config["interface"] = TM2020InterfaceLidar
    env_config["wait_on_done"] = True
    env_config["interface_kwargs"] = {"img_hist_len": 1, "gamepad": False, "min_nb_steps_before_failure": int(20 * 60)}
    env = gym.make("real-time-gym-v0", config=env_config)
    o, i = env.reset()
    while True:
        o, r, d, t, i = env.step(None)
        logging.info(f"r:{r}, d:{d}, t:{t}")
        if d or t:
            o, i = env.reset()
        img = window_interface.screenshot()[:, :, :3]
        lidar.lidar_20(img, True)


def show_imgs(imgs):
    imshape = imgs.shape
    if len(imshape) == 3:  # grayscale
        nb, h, w = imshape
        concat = imgs.reshape((nb*h, w))
        cv2.imshow("Environment", concat)
        cv2.waitKey(1)
    elif len(imshape) == 4:  # color
        nb, h, w, c = imshape
        concat = imgs.reshape((nb*h, w, c))
        cv2.imshow("Environment", concat)
        cv2.waitKey(1)


def check_env_tm20full():
    env_config = DEFAULT_CONFIG_DICT.copy()
    env_config["interface"] = TM2020Interface
    env_config["wait_on_done"] = True
    env_config["interface_kwargs"] = {"gamepad": False,
                                      "min_nb_steps_before_failure": int(20 * 60),
                                      "grayscale": cfg.GRAYSCALE,
                                      "resize_to": (cfg.IMG_WIDTH, cfg.IMG_HEIGHT)}
    env = gym.make("real-time-gym-v0", config=env_config)
    o, i = env.reset()
    show_imgs(o[3])
    logging.info(f"o:[{o[0].item():05.01f}, {o[1].item():03.01f}, {o[2].item():07.01f}, imgs({len(o[3])})]")
    while True:
        o, r, d, t, i = env.step(None)
        show_imgs(o[3])
        logging.info(f"r:{r:.2f}, d:{d}, t:{t}, o:[{o[0].item():05.01f}, {o[1].item():03.01f}, {o[2].item():07.01f}, imgs({len(o[3])})]")
        if d or t:
            o, i = env.reset()
            show_imgs(o[3])
            logging.info(f"o:[{o[0].item():05.01f}, {o[1].item():03.01f}, {o[2].item():07.01f}, imgs({len(o[3])})]")


if __name__ == "__main__":
    # check_env_tm20lidar()
    # check_env_tm20_trackmap()
    check_env_tm20full()
