import time

import cv2
import glm
import moderngl
import moderngl_window
from moderngl_window.conf import settings
import numpy

import obj_loader

settings.WINDOW['class'] = 'moderngl_window.context.glfw.Window'
settings.WINDOW['gl_version'] = (4, 1)

car_vertices = obj_loader.load_obj('res/models/car.obj')
cone_vertices = obj_loader.load_obj('res/models/cone.obj')
tree_vertices = obj_loader.load_obj('res/models/tree.obj')

road_vertices = numpy.dstack([
    [1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [10.0, -10.0, -10.0, 10.0, 10.0, -10.0],
    [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    [10.0, -10.0, -10.0, 10.0, 10.0, -10.0],
]).astype(numpy.float32)


def load_texture(path, moderngl_context):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return moderngl_context.texture(image.shape[:-1], image.shape[-1], image)


class Environment:
    checkpoint_distance = 1.0
    sky_color = (0.3, 0.5, 0.7, 1.0)

    def __init__(self, agents=1, window_size=(800, 600)):
        self.agents = agents
        self.state = numpy.rec.array([(0.0, 0.0, 0.0, self.checkpoint_distance) for _ in range(agents)], dtype=[
            ('x', numpy.float32),
            ('z', numpy.float32),
            ('angle', numpy.float32),
            ('checkpoint', numpy.float32),
        ])
        self.next_checkpoint = numpy.full(agents, self.checkpoint_distance)
        self.done = numpy.random.random(agents) > 0.5
        self.state_image_buffer = numpy.zeros((agents, 96, 96, 3))

        # Rendering
        self.frame_time = time.time()
        self.context = moderngl.create_standalone_context()
        self.context.enable(moderngl.DEPTH_TEST)

        settings.WINDOW['size'] = window_size
        settings.WINDOW['aspect_ratio'] = window_size[0] / window_size[1]
        self.window = moderngl_window.create_window_from_settings()
        self.window.ctx.enable(moderngl.DEPTH_TEST)
        self.window.ctx.front_face = 'cw'
        self.offscreen_framebuffer = self.context.simple_framebuffer((96, 96))

        with open('res/shaders/unlit.vert') as vertex_shader, open('res/shaders/unlit.frag') as fragment_shader:
            self.shader = self.context.program(vertex_shader=vertex_shader.read(),
                                               fragment_shader=fragment_shader.read())

        self.car_vbo = self.context.buffer(car_vertices.tobytes())
        self.car_vao = self.context.vertex_array(self.shader, self.car_vbo, 'in_vert', 'in_uv', mode=moderngl.TRIANGLES)
        self.car_texture = load_texture('res/textures/car.png', self.context)

        self.road_vbo = self.context.buffer(road_vertices.tobytes())
        self.road_vao = self.context.vertex_array(self.shader, self.road_vbo, 'in_vert', 'in_uv',
                                                  mode=moderngl.TRIANGLES)
        self.road_texture = load_texture('res/textures/road.png', self.context)

        self.cone_vbo = self.context.buffer(cone_vertices.tobytes())
        self.cone_vao = self.context.vertex_array(self.shader, self.cone_vbo, 'in_vert', 'in_uv',
                                                  mode=moderngl.TRIANGLES)
        self.tree_vbo = self.context.buffer(tree_vertices.tobytes())
        self.tree_vao = self.context.vertex_array(self.shader, self.tree_vbo, 'in_vert', 'in_uv',
                                                  mode=moderngl.TRIANGLES)
        self.props_texture = load_texture('res/textures/props.png', self.context)

        self.proj = glm.perspective(0.25 * numpy.pi, window_size[0] / window_size[1], 0.001, 100.0)
        self.view = glm.lookAt((1, 1, 5), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))

    def _render(self, view_matrix, render_agents):
        if 'proj' in self.shader:
            self.shader['proj'].write(self.proj)
        if 'view' in self.shader:
            self.shader['view'].write(view_matrix)

        model = glm.mat4(1.0)
        if 'model' in self.shader:
            self.shader['model'].write(model)
        if 'mvp' in self.shader:
            self.shader['mvp'].write(self.proj * view_matrix * model)
        if 'tint' in self.shader:
            self.shader['tint'].write(glm.vec4(1.0, 1.0, 1.0, 1.0))

        self.road_texture.use()
        self.road_vao.render()

        if render_agents:
            self.car_texture.use()
            for i in range(self.agents):
                x, z, angle = self.state.x[i], self.state.z[i], self.state.angle[i]
                model = glm.translate(glm.vec3(x, 0.0, z)) * glm.rotate(angle, glm.vec3(0.0, 1.0, 0.0))

                if 'model' in self.shader:
                    self.shader['model'].write(model)
                if 'mvp' in self.shader:
                    self.shader['mvp'].write(self.proj * view_matrix * model)
                if 'tint' in self.shader:
                    self.shader['tint'].write(glm.vec4(1.0))

                self.car_vao.render()

        self.props_texture.use()
        for i in range(10):
            x, z, angle = 2.0 if i % 2 == 1 else -2.0, -10 + 2 * i, i * (numpy.pi / 10)
            model = glm.translate(glm.vec3(x, 0.0, z)) * glm.rotate(angle, glm.vec3(0.0, 1.0, 0.0))

            if 'model' in self.shader:
                self.shader['model'].write(model)
            if 'mvp' in self.shader:
                self.shader['mvp'].write(self.proj * view_matrix * model)
            if 'tint' in self.shader:
                self.shader['tint'].write(glm.vec4(1.0))

            self.tree_vao.render()

        for i in range(10):
            x, z, angle = 2.0 if i % 2 == 0 else -2.0, -10 + 2 * i, i * (numpy.pi / 10)
            model = glm.translate(glm.vec3(x, 0.0, z)) * glm.rotate(angle, glm.vec3(0.0, 1.0, 0.0))

            if 'model' in self.shader:
                self.shader['model'].write(model)
            if 'mvp' in self.shader:
                self.shader['mvp'].write(self.proj * view_matrix * model)
            if 'tint' in self.shader:
                self.shader['tint'].write(glm.vec4(1.0))

            self.cone_vao.render()

    def reset(self):
        self.state.x = -1.0 + 2.0 * numpy.random.random(self.agents)
        self.state.z = 0.0
        self.state.angle = numpy.pi
        self.state.checkpoint = numpy.full(self.agents, self.checkpoint_distance)

    def step(self, action, timestep):
        # previous_state = self.state.copy()
        # Simulate
        self.state.angle += action.speed * action.steering * (numpy.pi / 8) * timestep
        self.state.x += action.speed * numpy.cos(self.state.angle + 0.5 * numpy.pi) * timestep
        self.state.z -= action.speed * numpy.sin(self.state.angle + 0.5 * numpy.pi) * timestep

        r = numpy.zeros(self.agents)

        checkpoint_reached = self.state.z > self.next_checkpoint
        self.next_checkpoint += self.checkpoint_distance
        r += 10.0 * checkpoint_reached

        self.offscreen_framebuffer.use()
        for i in range(self.agents):
            angle = self.state.angle[i] + (0.5 * numpy.pi)
            view_matrix = glm.lookAt(
                glm.vec3(self.state.x[i], 0.25, self.state.z[i]),
                glm.vec3(self.state.x[i] + numpy.cos(angle), 0.0, self.state.z[i] - numpy.sin(angle)),
                glm.vec3(0.0, 1.0, 0.0)
            )

            self.offscreen_framebuffer.clear(*self.sky_color)
            self._render(view_matrix, False)

            image = numpy.frombuffer(self.offscreen_framebuffer.read(), dtype=numpy.uint8).reshape(
                (96, 96, 3))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.state_image_buffer[i, :, :, :] = image / 255

        return self.state_image_buffer, self.done

    def render(self):
        fps = 1.0 / (time.time() - self.frame_time)
        self.frame_time = time.time()
        self.window.use()
        self.window.title = f'{fps:.2f} FPS'
        self.window.clear(*self.sky_color)
        target_z = float(self.state.z.min())
        self.view = glm.lookAt((0, 1, target_z + 5), (0.0, 0.0, target_z), (0.0, 1.0, 0.0))
        self._render(self.view, True)
        self.window.swap_buffers()

    def close(self):
        self.window.close()
