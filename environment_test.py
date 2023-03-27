import random

import cv2
import gymnasium

import self_driving_car


env = gymnasium.make('SelfDrivingCar-v0', render_mode='human', scene_name='env_basic.obj', max_episode_steps=200)

(camera, state), info = env.reset()
while not env.window.is_closing:
    action = env.action_space.sample()
    (camera, state), reward, truncated, done, info = env.step(action)

    env.render()
    cv2.imshow('Camera', cv2.cvtColor(cv2.medianBlur(camera, 5), cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)
    print(f'Speed: {state[0]}\tSteering: {state[1]}')

    if done:
        env.reset()
