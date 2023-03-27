import pathlib

import gymnasium
from moderngl_window import resources

from .self_driving_car import STATE_WIDTH, STATE_HEIGHT, STATE_CHANNELS, STATE_SHAPE
from .self_driving_car import SelfDrivingCar

resources.register_scene_dir(pathlib.Path('self_driving_car/res/scenes').resolve())
resources.register_program_dir(pathlib.Path('self_driving_car/res/shaders').resolve())
resources.register_texture_dir(pathlib.Path('self_driving_car/res/textures').resolve())

gymnasium.envs.registration.register(id='SelfDrivingCar-v0', entry_point='self_driving_car:SelfDrivingCar')
