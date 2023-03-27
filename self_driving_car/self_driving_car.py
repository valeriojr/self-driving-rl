import inspect
import time
from math import degrees, radians

import PIL.Image
import gymnasium
import moderngl
import moderngl_window
import numpy
from gymnasium import spaces
from moderngl_window import geometry
from moderngl_window import scene
from moderngl_window.meta import ProgramDescription
from moderngl_window.meta import SceneDescription
from moderngl_window.meta import TextureDescription
from moderngl_window.resources import programs
from moderngl_window.resources import scenes
from moderngl_window.resources import textures
from numpy import random
from pyrr import Matrix44
from pyrr import Vector3

STATE_WIDTH = 96
STATE_HEIGHT = 96
STATE_CHANNELS = 3
STATE_SHAPE = (STATE_HEIGHT, STATE_WIDTH, STATE_CHANNELS)
ATTR_NAMES = geometry.AttributeNames(position='in_position', normal='in_normal', texcoord_0='in_tex_coord')


def action(f):
    f.is_action = True

    return f


def set_uniforms(shader, **kwargs):
    for key, value in kwargs.items():
        if key in shader:
            shader[key].write(value)


class AckermanVehicle:
    def __init__(self, shader, position=Vector3(), angle=0.0, axle_length=2.0, wheel_distance=1.0,
                 steering_range=radians(45.0), top_speed=1, wheel_radius=0.5, wheel_width=0.25):
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

        self.actions = [value for name, value in inspect.getmembers(self, lambda member: hasattr(member, 'is_action'))]

        self.chassi_vao = scenes.load(SceneDescription('vehicle.obj', attr_names=ATTR_NAMES)).root_nodes[
            0].mesh.vao.instance(shader)
        self.wheel_radius = wheel_radius
        self.wheel_width = wheel_width
        self.wheel_scale = Vector3((wheel_width, wheel_radius, wheel_radius), dtype='f4')
        self.wheel_vao = scenes.load(SceneDescription('wheel.obj', attr_names=ATTR_NAMES)).root_nodes[
            0].mesh.vao.instance(shader)
        wheel_image = PIL.Image.open('self_driving_car/res/textures/wheel.png')
        self.wheel_texture = textures.load(TextureDescription(image=wheel_image))

    def reset(self):
        self.current_speed = 0.0
        self.target_speed = 0.0
        self.current_steer = 0.0
        self.target_steer = 0.0

    def throttle(self, value):
        self.target_speed = value

    def steer(self, value):
        self.target_steer += value
        if self.target_steer < -self.steering_range:
            self.target_steer = -self.steering_range
        if self.target_steer > self.steering_range:
            self.target_steer = self.steering_range

    def step(self, dt):
        self.current_steer += (self.target_steer - self.current_steer) * dt
        self.current_speed += (self.target_speed - self.current_speed) * dt

        self.angle += numpy.abs(self.current_speed) * (self.current_steer * self.steering_range) * dt
        self.position += (self.current_speed * self.top_speed * dt) * Vector3(
            (numpy.sin(self.angle), 0.0, numpy.cos(self.angle)), dtype='f4')

    def render(self, shader, proj_view):
        p = Vector3((self.position.x, self.position.y + self.wheel_radius / 2, -self.position.z), dtype='f4')
        model = Matrix44.from_translation(p, dtype='f4') * Matrix44.from_y_rotation(self.angle, dtype='f4')
        set_uniforms(shader, model=model, mvp=proj_view * model, color=numpy.zeros(4, dtype=numpy.float32))
        self.chassi_vao.render()

        self.wheel_texture.use()

        for i, wheel in enumerate(self.wheel_positions):
            steering_rotation = Matrix44.from_y_rotation(
                self.angle if i >= 2 else self.angle + self.current_steer * self.steering_range, dtype='f4')

            model = Matrix44.from_translation(Vector3((wheel.x, wheel.y + self.wheel_radius, -wheel.z), dtype='f4')) * \
                    steering_rotation * \
                    Matrix44.from_x_rotation(-self.current_speed * time.time(), dtype='f4') * \
                    Matrix44.from_scale(self.wheel_scale, dtype='f4')
            set_uniforms(shader, model=model, mvp=proj_view * model, color=numpy.ones(4, dtype=numpy.float32))
            self.wheel_vao.render()

    def transform_point(self, p):
        return self.position + Matrix44.from_y_rotation(-self.angle) * p

    @property
    def camera_position(self):
        return self.transform_point(Vector3((0.0, 1.0, 1.3), dtype='f4'))

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

    @action
    def do_nothing(self, timestep):
        pass

    @action
    def reverse(self, timestep):
        self.throttle(-0.25)

    @action
    def low_speed(self, timestep):
        self.throttle(0.25)

    @action
    def mid_speed(self, timestep):
        self.throttle(0.5)

    @action
    def high_speed(self, timestep):
        self.throttle(1.0)

    @action
    def steer_left_slightly(self, timestep):
        self.steer(-0.5 * timestep)

    @action
    def steer_left(self, timestep):
        self.steer(-1.0 * timestep)

    @action
    def steer_left_sharply(self, timestep):
        self.steer(-2.0 * timestep)

    @action
    def steer_right_slightly(self, timestep):
        self.steer(0.5 * timestep)

    @action
    def steer_right(self, timestep):
        self.steer(1.0 * timestep)

    @action
    def steer_right_sharply(self, timestep):
        self.steer(2.0 * timestep)


class Road:
    def __init__(self, context, shader, texture, length=10.0, lane_width=3.75):
        self.lane_width = lane_width
        self.length = length

        position = numpy.array([
            -lane_width, 0.0, 0.0,
            lane_width, 0.0, 0.0,
            lane_width, 0.0, -length,

            -lane_width, 0.0, 0.0,
            lane_width, 0.0, -length,
            -lane_width, 0.0, -length,
        ], dtype=numpy.float32)

        normal = numpy.array([
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
        ])

        tex_coord = numpy.array([
            0.0, 0.0,
            1.0, 0.0,
            1.0, length / (2 * self.lane_width),

            0.0, 0.0,
            1.0, length / (2 * self.lane_width),
            0.0, length / (2 * self.lane_width),
        ], dtype=numpy.float32)

        self.vao = context.vertex_array(shader, [
            (context.buffer(position), '3f', 'in_position'),
            (context.buffer(normal), '3f', 'in_normal'),
            (context.buffer(tex_coord), '2f', 'in_tex_coord'),
        ])
        self.texture = texture

        position = numpy.array([
            -1000.0, -0.1, -1000.0,
            1000.0, -0.1, -1000.0,
            1000.0, -0.1, -1000.0,

            -1000.0, -0.1, 1000.0,
            1000.0, -0.1, -1000.0,
            -1000.0, -0.1, -1000.0,
        ], dtype=numpy.float32)

        tex_coord = numpy.array([
            -50.0, -50.0,
            50.0, -50.0,
            50.0, 50.0,

            -50.0, -50.0,
            50.0, 50.0,
            -50.0, 50.0,
        ], dtype=numpy.float32)

        grass_image = PIL.Image.open('self_driving_car/res/textures/grass.png')
        self.grass_texture = textures.load(TextureDescription(image=grass_image))
        self.grass_vao = context.vertex_array(shader, [
            (context.buffer(position), '3f', 'in_position'),
            (context.buffer(normal), '3f', 'in_normal'),
            (context.buffer(tex_coord), '2f', 'in_tex_coord'),
        ])

    def render(self):
        self.texture.use()
        self.vao.render(moderngl.TRIANGLES)

        self.grass_texture.use()
        self.grass_vao.render(moderngl.TRIANGLES)


class SelfDrivingCar(gymnasium.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array', 'state_pixels'],
        'render_fps': 60,
    }

    def __init__(self, render_mode=None, scene_name=None, timestep=0.1):
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        if self.render_mode == 'human':
            moderngl_window.settings.WINDOW['class'] = 'moderngl_window.context.glfw.Window'
            moderngl_window.settings.WINDOW['glVersion'] = (4, 1)
            moderngl_window.settings.WINDOW['size'] = (640, 480)
            self.window = moderngl_window.create_window_from_settings()
            self.context = self.window.ctx

            self.camera = scene.Camera(aspect_ratio=self.window.aspect_ratio, fov=45.0, near=0.01, far=1000.0)
            self.camera.set_position(0, 4, 0)
            self.camera.set_rotation(-90.0, -25.0)

        else:
            self.context = moderngl.create_standalone_context()

        self.shader = programs.load(ProgramDescription('cube_simple.glsl'))
        self.shader['color'].value = 1.0, 1.0, 1.0, 1.0

        self.state = None
        self.start_time = time.time()
        self.onboard_camera = scene.Camera(fov=45.0, aspect_ratio=STATE_WIDTH / STATE_HEIGHT, near=0.001, far=100.0)
        self.position_history = []
        self.angle_history = []
        self.vehicle = AckermanVehicle(self.shader)

        self.observation_space = spaces.Tuple([
            spaces.Box(low=0, high=255, shape=STATE_SHAPE, dtype=numpy.uint8),
            spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=numpy.float32)
        ])
        self.action_space = spaces.Discrete(n=len(self.vehicle.actions))
        self.timestep = timestep

        self.context.enable(moderngl.DEPTH_TEST)

        grid_lines = 50
        grid_start = -grid_lines / 2
        self.grid_shader = programs.load(ProgramDescription('lines.glsl'))
        grid_vbo = self.context.buffer(numpy.array([
            [
                grid_start + j, 0.0, -grid_start,
                grid_start + j, 0.0, grid_start,
                -grid_start, 0.0, grid_start + j,
                grid_start, 0.0, grid_start + j
            ] for j in range(grid_lines)
        ], dtype=numpy.float32).ravel())
        self.grid = self.context.vertex_array(self.grid_shader, grid_vbo, 'in_position')

        road_image = PIL.Image.open('self_driving_car/res/textures/road.png')
        road_texture = textures.load(TextureDescription(image=road_image))
        self.road = Road(self.context, self.shader, road_texture, length=100)

        self.framebuffer = self.context.simple_framebuffer((STATE_WIDTH, STATE_HEIGHT))

    def _render(self, target, camera):
        target.use()
        target.clear(0.529, 0.808, 0.922, 1.0)

        proj = camera.projection.matrix
        view = camera.matrix
        proj_view = proj * view
        set_uniforms(self.shader, view=view, proj=proj)

        model = Matrix44.identity(dtype='f4')
        mvp = proj_view * model
        set_uniforms(self.shader, model=model, mvp=mvp)
        self.road.render()

    def _get_obs(self):
        p = self.vehicle.camera_position
        self.onboard_camera.set_position(p.x, p.y, -p.z)
        self.onboard_camera.set_rotation(degrees(self.vehicle.angle) - 90.0, 0.0)

        self._render(self.framebuffer, self.onboard_camera)

        buffer = self.framebuffer.read()
        self.state = numpy.flipud(numpy.frombuffer(buffer, dtype=numpy.uint8).reshape(STATE_SHAPE))

        return (
            self.state,
            numpy.array([self.vehicle.current_speed, self.vehicle.current_steer], dtype=numpy.float32)
        )

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
        self.vehicle.reset()
        r = numpy.random.uniform(-1.0, 1.0)
        offset = self.road.lane_width - 0.5 * self.vehicle.wheel_distance
        self.vehicle.position = Vector3((r * offset, 0.0, 4.0), dtype='f4')
        self.vehicle.angle = (1 - abs(r)) * random.uniform(radians(-10.0), radians(10.0))

        self.position_history.clear()
        self.angle_history.clear()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Apply action
        assert self.vehicle is not None
        if action is not None:
            self.vehicle.actions[action](self.timestep)

        self.vehicle.step(self.timestep)
        plate_position = self.vehicle.transform_point(Vector3((0.0, 0.0, self.vehicle.axle_length), dtype='f4'))
        self.start_time = time.time()

        reward = 0.0
        if plate_position.x >= 0.0:
            reward += 0.01 * numpy.cos(self.vehicle.angle)

        if len(self.position_history) > 0:
            if plate_position.x < 0.0 < self.position_history[-1].x:
                reward += -20.0
            elif self.position_history[-1].x < 0.0 < plate_position.x:
                reward += 5.0

        done = False
        if abs(plate_position.x) > self.road.lane_width:
            done = True
            if plate_position.x < 0.0:
                reward += self.vehicle.position.z - 100.0
            else:
                reward += self.vehicle.position.z - 30.0  # Replace with "Out of Bounds" reward

        observation = self._get_obs()
        info = self._get_info()

        self.position_history.append(self.vehicle.camera_position)
        self.angle_history.append(self.angle_history)

        return observation, reward, done, False, info

    def render(self):
        assert self.render_mode == 'human' and self.window is not None
        proj_view = self.camera.projection.matrix * self.camera.matrix

        self._render(self.window, self.camera)

        set_uniforms(self.grid_shader, color=numpy.array([1.0, 1.0, 1.0, 1.0], dtype=numpy.float32), mvp=proj_view,
                     view=self.camera.matrix)
        self.grid.render(moderngl.LINES)

        position = self.vehicle.position
        translation = Matrix44.from_translation(
            Vector3((self.vehicle.position.x, 0.0, -self.vehicle.position.z), dtype='f4'), dtype='f4')
        rotation = Matrix44.from_y_rotation(self.vehicle.angle, dtype='f4')
        vehicle_model = translation * rotation

        self.vehicle.render(self.shader, proj_view)

        self.window.swap_buffers()

    def close(self):
        # self.framebuffer.release()
        if self.window is not None:
            self.window.close()
