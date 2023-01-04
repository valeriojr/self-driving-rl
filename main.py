import cv2
import numpy

import environment

FRAMEBUFFER_SIZE = (1280, 720)

if __name__ == '__main__':
    agents = 10
    speed_action_count = 5
    steering_action_count = 5

    env = environment.Environment(agents, FRAMEBUFFER_SIZE)

    env.reset()

    while not env.window.is_closing:
        a = numpy.random.randint(low=(0, 0), high=(speed_action_count, steering_action_count),
                                 size=(agents, 2)).transpose()
        a = (a - 2) / 2
        action = numpy.rec.fromarrays(a, dtype=[('speed', numpy.float32), ('steering', numpy.float32)])

        state, done = env.step(action, timestep=0.032)

        env.render()

        # for i in range(agents):
        #     cv2.imshow(f'Agent {i}', state[i])
        # key = cv2.waitKey(16)
        # if key == 27:
        #     env.close()
        #     break

