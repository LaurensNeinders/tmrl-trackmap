# third-party imports
import gym
import numpy as np
from rtgym.envs.real_time_env import DEFAULT_CONFIG_DICT

# local imports
from tmrl.custom.custom_gym_interfaces import TM2020Interface, TM2020InterfaceLidar, TM2020InterfaceLidarTrackMap
from tmrl.custom.utils.window import WindowInterface
from tmrl.custom.utils.tools import Lidar
import logging

# Custom imports
from tmrl.custom.custom_gym_interfaces import get_coordinates
from matplotlib import pyplot as plt
from time import sleep


def check_env_tm20lidar():
    window_interface = WindowInterface("Trackmania")
    lidar = Lidar(window_interface.screenshot())
    env_config = DEFAULT_CONFIG_DICT.copy()
    env_config["interface"] = TM2020InterfaceLidarTrackMap
    env_config["wait_on_done"] = True
    env_config["interface_kwargs"] = {"img_hist_len": 1, "gamepad": False, "min_nb_steps_before_failure": int(20 * 60), "record": False}
    # env_config["time_step_duration"] = 0.5  # nominal duration of your time-step
    # env_config["start_obs_capture"] = 0.4
    env = gym.make("real-time-gym-v0", config=env_config)
    o, i = env.reset()
    rounds = 100
    current=0
    while current < rounds:
        current+=1
        o, r, d, t, i = env.step(None)
        # logging.info(f"r:{r}, d:{d}")

        if d or t:
            o, i = env.reset()
        img = window_interface.screenshot()[:, :, :3]
        lidar.lidar_20(img, True)
    x,z,color = get_coordinates()

    sleep(1)

    # color = np.arange(len(x))
    plt.scatter(x,z,c=color)
    plt.show()
    # input("Press Enter to continue...")
if __name__ == "__main__":
    check_env_tm20lidar()
