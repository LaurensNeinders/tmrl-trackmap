import os
import platform
import sys
from setuptools import find_packages, setup
from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import urllib.request
import urllib.error
import socket

if sys.version_info < (3, 7):
    sys.exit('Sorry, Python < 3.7 is not supported. We use dataclasses that have been introduced in 3.7.')


RESOURCES_SRC = "./resources.zip"


def url_retrieve(url: str, outfile: Path, overwrite: bool = False):
    """
    Adapted from https://www.scivision.dev/python-switch-urlretrieve-requests-timeout/
    """
    outfile = Path(outfile).expanduser().resolve()
    if outfile.is_dir():
        raise ValueError("Please specify full filepath, including filename")
    if overwrite or not outfile.is_file():
        outfile.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, str(outfile))
        except (socket.gaierror, urllib.error.URLError) as err:
            raise ConnectionError(f"could not download {url} due to {err}")


# destination folder:
HOME_FOLDER = Path.home()
TMRL_FOLDER = HOME_FOLDER / "TmrlData"

# download relevant items IF THE tmrl FOLDER DOESN'T EXIST:
if not TMRL_FOLDER.exists():
    # print("we create a new tmrldata folder")
    CHECKPOINTS_FOLDER = TMRL_FOLDER / "checkpoints"
    DATASET_FOLDER = TMRL_FOLDER / "dataset"
    REWARD_FOLDER = TMRL_FOLDER / "reward"
    WEIGHTS_FOLDER = TMRL_FOLDER / "weights"
    CONFIG_FOLDER = TMRL_FOLDER / "config"
    CHECKPOINTS_FOLDER.mkdir(parents=True, exist_ok=True)
    DATASET_FOLDER.mkdir(parents=True, exist_ok=True)
    REWARD_FOLDER.mkdir(parents=True, exist_ok=True)
    WEIGHTS_FOLDER.mkdir(parents=True, exist_ok=True)
    CONFIG_FOLDER.mkdir(parents=True, exist_ok=True)

    # download resources:
    RESOURCES_TARGET = TMRL_FOLDER / "resources.zip"
    copy2(RESOURCES_SRC, RESOURCES_TARGET)

    # unzip downloaded resources:
    with ZipFile(RESOURCES_TARGET, 'r') as zip_ref:
        zip_ref.extractall(TMRL_FOLDER)

    # delete zip file:
    RESOURCES_TARGET.unlink()

    # copy relevant files:
    RESOURCES_FOLDER = TMRL_FOLDER / "resources"
    copy2(RESOURCES_FOLDER / "config.json", CONFIG_FOLDER)
    copy2(RESOURCES_FOLDER / "reward.pkl", REWARD_FOLDER)
    copy2(RESOURCES_FOLDER / "SAC_4_LIDAR_pretrained.tmod", WEIGHTS_FOLDER)
    copy2(RESOURCES_FOLDER / "SAC_4_imgs_pretrained.tmod", WEIGHTS_FOLDER)
    copy2(RESOURCES_FOLDER / "SAC_trackmap_pretrained.tmod", WEIGHTS_FOLDER)
    # copy2(RESOURCES_FOLDER / "SAC_trackmap_pretrained_t.tcpt", CHECKPOINTS_FOLDER)


    # on Windows, look for OpenPlanet:
    if platform.system() == "Windows":
        OPENPLANET_FOLDER = HOME_FOLDER / "OpenplanetNext"

        if OPENPLANET_FOLDER.exists():
            # copy the OpenPlanet script:
            try:
                OP_SCRIPTS_FOLDER = OPENPLANET_FOLDER / 'Plugins' / 'GrabData'
                OP_SCRIPTS_FOLDER.mkdir(parents=True, exist_ok=True)
                TM20_SCRIPT_FILE = RESOURCES_FOLDER / 'Plugins' / 'GrabData' / 'Plugin_GrabData_extended.as'
                TM20_SCRIPT_FILE_INFO = RESOURCES_FOLDER / 'Plugins' / 'GrabData' /  'info.toml'
                copy2(TM20_SCRIPT_FILE, OP_SCRIPTS_FOLDER)
                copy2(TM20_SCRIPT_FILE_INFO, OP_SCRIPTS_FOLDER)
            except Exception as e:
                print(
                    f"An exception was caught when trying to copy the OpenPlanet script and signature automatically. \
                    Please copy these files manually for TrackMania 2020 support. The caught exception was: {str(e)}.")
        else:
            # warn the user that OpenPlanet couldn't be found:
            print(f"The OpenPlanet folder was not found at {OPENPLANET_FOLDER}. \
            Please copy the OpenPlanet script and signature manually for TrackMania 2020 support.")


install_req = [
    'numpy',
    'torch',
    'imageio',
    'imageio-ffmpeg',
    'pandas',
    'gym>=0.26.0',
    'rtgym>=0.7',
    'pyyaml',
    'wandb',
    'requests',
    'opencv-python',
    'scikit-image',
    'keyboard',
    'pyautogui',
    'pyinstrument',
    'tlspyo>=0.2.5',
    'matplotlib'
]

if platform.system() == "Windows":
    install_req.append('pywin32>=303')
    install_req.append('vgamepad')

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

# The text of the README file
with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

setup(
    name='tmrl',
    version='0.4.1',
    description='Network-based framework for real-time robot learning',
    long_description=README,
    long_description_content_type='text/markdown',
    keywords='reinforcement learning, robot learning, trackmania, self driving, roborace',
    url='https://github.com/trackmania-rl/tmrl',
    download_url='https://github.com/trackmania-rl/tmrl/archive/refs/tags/v0.4.1.tar.gz',
    author='Yann Bouteiller, Edouard Geze',
    author_email='yann.bouteiller@polymtl.ca, edouard.geze@hotmail.fr',
    license='MIT',
    install_requires=install_req,
    classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Information Technology',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python',
            'Topic :: Games/Entertainment',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    include_package_data=True,
    extras_require={},
    scripts=[],
    packages=find_packages(exclude=("tests", )))
