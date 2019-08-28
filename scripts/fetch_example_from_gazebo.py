#!/usr/bin/env python
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from link_bot_gazebo.gazebo_utils import GazeboServices
from link_bot_gazebo.srv import LinkBotStateRequest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir')
    parser.add_argument('--show', action='store_true')

    args = parser.parse_args()

    np.set_printoptions(suppress=True, precision=4, linewidth=250)

    services = GazeboServices()
    state_req = LinkBotStateRequest()
    s = services.get_state(state_req)
    first_context_image = np.copy(np.frombuffer(s.camera_image.data, dtype=np.uint8)).reshape([64, 64, 3])
    first_context_state = [s.points[-1].x, s.points[-1].y]
    s = services.get_state(state_req)
    second_context_image = np.copy(np.frombuffer(s.camera_image.data, dtype=np.uint8)).reshape([64, 64, 3])
    second_context_state = [s.points[-1].x, s.points[-1].y]
    context_states = np.stack((first_context_state, second_context_state))
    context_actions = np.array([[s.gripper1_velocity.x, s.gripper1_velocity.y]])

    if args.show or not args.outdir:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].imshow(first_context_image)
        axes[0].set_title('state: ({:0.3f}, {:0.3f})'.format(first_context_state[0], first_context_state[1]))
        axes[1].imshow(second_context_image)
        axes[1].set_title('state: ({:0.3f}, {:0.3f})'.format(second_context_state[0], second_context_state[1]))
        print('actions:', context_actions)
        plt.show()

    if args.outdir:
        plt.imsave(os.path.join(args.outdir, '0.png'), first_context_image)
        plt.imsave(os.path.join(args.outdir, '1.png'), second_context_image)
        np.savetxt(os.path.join(args.outdir, 'states.csv'), context_states, delimiter=',')
        np.savetxt(os.path.join(args.outdir, 'context_actions.csv'), context_actions, delimiter=',')


if __name__ == '__main__':
    main()
