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
    parser.add_argument('outdir')
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='train')
    parser.add_argument('--fps', type=int, default=2)

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
    outfile = os.path.join(args.outdir, "{}_video.mp4".format(args.mode))
    print("writing to {}".format(outfile))
    with imageio.get_writer(outfile, fps=args.fps) as writer:
        while True:
            try:
                outputs = sess.run(inputs)
            except tf.errors.OutOfRangeError:
                break

            images = outputs['images']
            images = np.clip((images * 255), 0, 255).astype(np.uint8)

            for traj in images:
                for image in traj:
                    writer.append_data(image)

            print("batch {}".format(batch_idx))
            batch_idx += 1


if __name__ == '__main__':
    main()
