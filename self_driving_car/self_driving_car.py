import enum
import time
from math import degrees, radians

import gymnasium
import moderngl
import moderngl_window
import numpy
from gymnasium import spaces
from moderngl_window import geometry
from moderngl_window import scene
from moderngl_window.meta import SceneDescription, ProgramDescription
from moderngl_window.resources import programs
from moderngl_window.resources import scenes
from numpy import random
from pyrr import Matrix44
from pyrr import Vector3

STATE_WIDTH = 96
STATE_HEIGHT = 96
STATE_CHANNELS = 3
STATE_SHAPE = (STATE_HEIGHT, STATE_WIDTH, STATE_CHANNELS)


class AckermanVehicle:
    def __init__(self, position=Vector3(), angle=0.0, axle_length=2.0, wheel_distance=1.0, steering_range=45.0,
                 top_speed=1):
        self.position = position
        self.angle = angle
        self.current_speed = 0.0
        self.target_speed = 0.0
        self.current_steer = 0.0
        self.target_steer = 0.0
        self.axle_length = axle_length
        self.wheel_distance = wheel_distance
        self.steering_range = steering_range
        self.top_speed = top_speed

    def throttle(self, value):
        self.target_speed = value

    def steer(self, value):
        self.target_steer = value

    def step(self, dt):
        self.current_steer += (self.target_steer - self.current_steer) * dt
        self.current_speed += (self.target_speed - self.current_speed) * dt

        self.angle += numpy.abs(self.current_speed) * (self.current_steer * self.steering_range) * dt
        self.position += (self.current_speed * self.top_speed * dt) * Vector3(
            (numpy.sin(self.angle), 0.0, numpy.cos(self.angle)), dtype='f4')

    def transform_point(self, p):
        return self.position + Matrix44.from_y_rotation(-self.angle) * p

    @property
    def front_right_wheel(self):
        return self.transform_point(Vector3((0.5 * self.wheel_distance, 0.0, self.axle_length), dtype='f4'))

    @property
    def front_left_wheel(self):
        return self.transform_point(Vector3((-0.5 * self.wheel_distance, 0.0, self.axle_length), dtype='f4'))

    @property
    def rear_right_wheel(self):
        return self.transform_point(Vector3((0.5 * self.wheel_distance, 0.0, 0.0), dtype='f4'))

    @property
    def rear_left_wheel(self):
        return self.transform_point(Vector3((-0.5 * self.wheel_distance, 0.0, 0.0), dtype='f4'))

    @property
    def wheel_positions(self):
        return self.front_left_wheel, self.front_right_wheel, self.rear_left_wheel, self.rear_right_wheel


class SelfDrivingCar(gymnasium.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array', 'state_pixels'],
        'render_fps': 60,
    }

    class Action(enum.Enum):
        REVERSE = 0
        LOW_SPEED = 1
        MID_SPEED = 2
        HIGH_SPEED = 3
        STEER_LEFT = 4
        STEER_CENTER = 5
        STEER_RIGHT = 6
        BRAKE = 7

    def __init__(self, render_mode=None, scene_name=None, timestep=0.1):
        self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_HEIGHT, STATE_WIDTH, STATE_CHANNELS),
                                            dtype=numpy.uint8)
        self.action_space = spaces.Discrete(n=len(self.Action))
        self.timestep = timestep

        self.state = None
        self.start_time = time.time()
        self.vehicle = AckermanVehicle()
        self.onboard_camera = scene.Camera(fov=45.0, aspect_ratio=STATE_WIDTH / STATE_HEIGHT, near=0.001, far=100.0)

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        if self.render_mode == 'human':
            moderngl_window.settings.WINDOW['class'] = 'moderngl_window.context.glfw.Window'
            moderngl_window.settings.WINDOW['glVersion'] = (4, 1)
            moderngl_window.settings.WINDOW['size'] = (640, 480)
            self.window = moderngl_window.create_window_from_settings()
            self.context = self.window.ctx

            self.camera = scene.Camera(aspect_ratio=self.window.aspect_ratio, fov=45.0, near=0.01, far=1000.0)
            self.camera.set_position(-3, 5, 5)
            self.camera.set_rotation(-45.0, -30.0)

            self.vehicle_shader = programs.load(ProgramDescription('cube_simple.glsl'))
            self.vehicle_shader['color'].value = 1.0, 1.0, 1.0, 1.0
            self.vehicle_model = scenes.load(SceneDescription('vehicle.obj')).root_nodes[0].mesh.vao.instance(self.vehicle_shader)
        else:
            self.context = moderngl.create_standalone_context()

        self.context.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.framebuffer = self.window.ctx.simple_framebuffer((STATE_WIDTH, STATE_HEIGHT))

        self.scene = scenes.load(SceneDescription(scene_name))

    def _render(self, render_mode):
        assert render_mode in self.metadata['render_modes']

        self.framebuffer.use()

        p = self.vehicle.transform_point(Vector3((0, 1.0, 1.3), dtype='f4'))
        self.onboard_camera.set_position(p.x, p.y, -p.z)
        self.onboard_camera.set_rotation(degrees(self.vehicle.angle) - 90.0, 0.0)

        self.framebuffer.clear()
        self.scene.draw(self.onboard_camera.projection.matrix, self.onboard_camera.matrix)

        buffer = self.framebuffer.read()
        return numpy.flipud(numpy.frombuffer(buffer, dtype=numpy.uint8).reshape(STATE_SHAPE))

    def _get_obs(self):
        self.state = self._render('state_pixels')
        return self.state

    def _get_info(self):
        return {
            'vehicle': {
                'position': self.vehicle.position,
                'angle': self.vehicle.angle,
                'speed': self.vehicle.current_speed * self.vehicle.top_speed,
                'steer': self.vehicle.current_steer * self.vehicle.steering_range,
            }
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Initialize car
        self.vehicle = AckermanVehicle()
        self.vehicle.position = Vector3((1.0 + random.uniform(-0.5, 0.5), 0.0, 0.0), dtype='f4')
        self.vehicle.angle = random.uniform(radians(-30.0), radians(30.0))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            # self._render_frame()
            pass

        return observation, info

    def step(self, action):
        # Apply action
        assert self.vehicle is not None
        if action is not None:
            action = self.Action(action)
            self.vehicle.steer(
                -1.0 * (action == self.Action.STEER_LEFT) +
                0.0 * (action == self.Action.STEER_CENTER) +
                1.0 * (action == self.Action.STEER_RIGHT))
            self.vehicle.throttle(
                -0.25 * (action == self.Action.REVERSE) +
                0.0 * (action == self.Action.BRAKE) +
                0.25 * (action == self.Action.LOW_SPEED) +
                0.5 * (action == self.Action.MID_SPEED) +
                1.0 * (action == self.Action.HIGH_SPEED)
            )

        self.vehicle.step(self.timestep)
        self.start_time = time.time()

        reward = 0.0
        for p in self.vehicle.wheel_positions:
            reward += -0.01 * (p.x < 0.0)  # Replace with "Wrong-Way Driving" reward

        done = False
        if abs(self.vehicle.position.x) > 2.0:  # Replace with road width
            done = True
            reward += -10.0  # Replace with "Out of Bounds" reward
        elif self.vehicle.position.z > 10.0:
            done = True
            reward += self.vehicle.position.z

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, False, info

    def render(self):
        assert self.render_mode == 'human' and self.window is not None
        self.window.use()

        position = self.vehicle.position
        translation = Matrix44.from_translation(
            Vector3((self.vehicle.position.x, 0.0, -self.vehicle.position.z), dtype='f4'), dtype='f4')
        rotation = Matrix44.from_y_rotation(self.vehicle.angle, dtype='f4')
        model = translation * rotation

        view_matrix = self.camera.matrix

        self.window.clear()
        self.scene.draw(self.camera.projection.matrix, view_matrix)

        self.vehicle_shader['m_proj'].write(self.camera.projection.matrix)
        self.vehicle_shader['m_camera'].write(self.camera.matrix)

        self.vehicle_shader['color'].value = 1, 1, 1, 1
        self.vehicle_shader['m_model'].write(model)
        self.vehicle_model.render()

        self.vehicle_shader['color'].value = 1, 0, 0, 1
        for wheel in self.vehicle.wheel_positions:
            self.vehicle_shader['m_model'].write(
                Matrix44.from_translation(Vector3((wheel.x, wheel.y, -wheel.z), dtype='f4'), dtype='f4'))
            geometry.sphere(radius=0.5).render(self.vehicle_shader)

        self.window.swap_buffers()

    def close(self):
        # self.framebuffer.release()
        if self.window is not None:
            self.window.close()
