import time

import gymnasium
import numpy
from tensorflow.python import keras
import tensorflow as tf

import self_driving_car


env = gymnasium.make('SelfDrivingCar-v0', render_mode='human', scene_name='env_basic.obj', timestep=0.1)
model = keras.models.load_model('results/27-Mar-2023_00-14-37/train/model.h5')

while not env.window.is_closing:
    (camera, vehicle_state), info = env.reset()
    for i in range(1000):
        start_time = time.time()

        action, _ = model([numpy.expand_dims(camera, axis=0), numpy.expand_dims(vehicle_state, axis=0)])
        action = tf.random.categorical(action, 1)[0, 0]

        (camera, vehicle_state), _, done, truncated, info = env.step(action)
        env.render()

        if done or truncated:
            break
        if env.window.is_closing:
            break

        if i % 10 == 9:
            end_time = time.time()

            step_time = end_time - start_time
            env.window.title = f'{int(1.0 / step_time)} FPS ({env.timestep / step_time:.2f}x)'
