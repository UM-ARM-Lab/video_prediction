#!/usr/bin/env python
import argparse
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf

from video_prediction import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('dataset')
    parser.add_argument('dataset_hparams_dict')
    parser.add_argument('--traj-idx', type=int, default=0)
    parser.add_argument('--time-idx', type=int, default=0)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--outdir')

    np.random.seed(0)
    tf.random.set_random_seed(0)
    tf.logging.set_verbosity(tf.logging.FATAL)

    args = parser.parse_args()

    np.set_printoptions(suppress=True, precision=4, linewidth=250)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2

    sess = tf.Session(config=config)

    VideoDataset = datasets.get_dataset_class(args.dataset)
    with open(args.dataset_hparams_dict, 'r') as hparams_f:
        hparams_dict = json.loads(hparams_f.read())
    dataset = VideoDataset(args.input_dir, mode="test", seed=0, num_epochs=1, hparams_dict=hparams_dict)

    inputs = dataset.make_batch(1, shuffle=False)
    runs = 0
    while True:
        outputs = sess.run(inputs)
        if runs == args.traj_idx:
            break
        runs += 1

    if args.interactive:
        first_context_image = outputs['images'][0, 0]
        second_context_image = outputs['images'][0, 1]

        fig, axes = plt.subplots(nrows=1, ncols=2)
        first_image_handle = axes[0].imshow(first_context_image)
        first_title_handle = axes[0].set_title("t=0")
        second_image_handle = axes[1].imshow(second_context_image)
        second_title_handle = axes[1].set_title("t=1")

        t = 0

        def on_key_release(event):
            nonlocal t
            if event.key == 'right':
                if t < outputs['images'].shape[1] - 2:
                    t += 1
            if event.key == 'left':
                if t > 0:
                    t -= 1
            first_context_image = outputs['images'][0, t]
            second_context_image = outputs['images'][0, t + 1]
            first_image_handle.set_data(first_context_image)
            first_title_handle.set_text("t={}".format(t))
            second_image_handle.set_data(second_context_image)
            second_title_handle.set_text("t={}".format(t + 1))

            fig.canvas.draw()
            fig.canvas.flush_events()

        fig.canvas.mpl_connect('key_release_event', on_key_release)
        plt.show()

        first_context_image = outputs['images'][0, t]
        second_context_image = outputs['images'][0, t + 1]
        context_states = outputs['states'][0, t:t + 2]
        actions = outputs['actions'][0, t:]

    else:
        first_context_image = outputs['images'][0, args.time_idx]
        second_context_image = outputs['images'][0, args.time_idx + 1]
        context_states = outputs['states'][0, args.time_idx:args.time_idx + 2]
        actions = outputs['actions'][0, args.time_idx:]

        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].imshow(first_context_image)
        axes[0].set_title("t=0")
        axes[1].imshow(second_context_image)
        axes[1].set_title("t=1")
        plt.show()

    print(context_states)
    print("=======")
    print(actions)

    if args.outdir:
        plt.imsave(os.path.join(args.outdir, '0.png'), first_context_image)
        plt.imsave(os.path.join(args.outdir, '1.png'), second_context_image)
        np.savetxt(os.path.join(args.outdir, 'states.csv'), context_states, delimiter=',')
        np.savetxt(os.path.join(args.outdir, 'actions.csv'), actions, delimiter=',')


if __name__ == '__main__':
    main()
