#!/usr/bin/env python
import argparse
import json
import os

import imageio
import numpy as np
import tensorflow as tf

from video_prediction import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('dataset')
    parser.add_argument('dataset_hparams_dict')
    parser.add_argument('outfile')
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='train')
    parser.add_argument('--fps', type=int, default=2)
    parser.add_argument('--first-n', type=int, default=-1)

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
    dataset = VideoDataset(args.input_dir, mode=args.mode, seed=0, num_epochs=1, hparams_dict=hparams_dict)

    inputs = dataset.make_batch(16, shuffle=False)
    batch_idx = 0
    outfile = "{}_{}.mp4".format(args.outfile, args.mode)
    print("writing to {}".format(outfile))
    frame_idx = 0
    with imageio.get_writer(outfile, fps=args.fps) as writer:
        while True:
            try:
                outputs = sess.run(inputs)
            except tf.errors.OutOfRangeError:
                break
            except KeyboardInterrupt:
                print("Keyboard Interrupt. Saving and stopping!")
                break

            images = outputs['images']
            images = np.clip((images * 255), 0, 255).astype(np.uint8)

            done, frame_idx = write_images(args, frame_idx, images, writer)
            if done:
                print("Stopping early!")
                break

            print("batch {}".format(batch_idx))
            batch_idx += 1


def write_images(args, frame_idx, images, writer):
    for traj in images:
        for image in traj:
            if 0 < args.first_n <= frame_idx:
                return True, frame_idx
            writer.append_data(image)
            frame_idx += 1
    return False, frame_idx


if __name__ == '__main__':
    main()
