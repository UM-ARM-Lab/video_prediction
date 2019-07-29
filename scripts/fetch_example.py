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
    outputs = None
    runs = 0
    while True:
        outputs = sess.run(inputs)
        if runs == args.traj_idx:
            break
        runs += 1

    images = outputs['images'][0, args.time_idx:args.time_idx + 2]
    states = outputs['states'][0, args.time_idx:args.time_idx + 2]
    actions = outputs['actions'][0, args.time_idx:]

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].imshow(images[0])
    axes[0].set_title("t=0")
    axes[1].imshow(images[1])
    axes[1].set_title("t=1")
    plt.show()

    if args.outdir:
        plt.imsave(os.path.join(args.outdir, '0.png'), images[0])
        plt.imsave(os.path.join(args.outdir, '1.png'), images[1])

    print(states)
    if args.outdir:
        np.savetxt(os.path.join(args.outdir, 'states.csv'), states, delimiter=',')

    print("=======")

    print(actions)
    if args.outdir:
        np.savetxt(os.path.join(args.outdir, 'actions.csv'), actions, delimiter=',')


if __name__ == '__main__':
    main()
