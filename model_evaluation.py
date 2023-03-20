import gymnasium
import numpy
from tensorflow.python import keras

import self_driving_car


env = gymnasium.make('SelfDrivingCar-v0', render_mode='human', scene_name='env_basic.obj', timestep=0.1)
model = keras.models.load_model('model.h5')

while True:
    state, info = env.reset()
    for i in range(200):
        action, _ = model(numpy.expand_dims(state, axis=0))
        state, _, done, truncated, info = env.step(action.numpy().argmax())
        env.render()

        if done or truncated:
            break

