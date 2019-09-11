#!/usr/bin/env python
import argparse
import pathlib

import numpy as np
import tensorflow as tf

from video_prediction.rollout_utils import setup_and_rollout_from_individual_files, rollout_main, setup_and_rollout_from_gazebo


def main():
    np.set_printoptions(suppress=True, linewidth=250, precision=4, threshold=64 * 64 * 3)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser()
    parser.add_argument("example_dir_path", type=pathlib.Path)
    parser.add_argument("actions", help='filename')
    parser.add_argument("checkpoint", help="directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000)")
    parser.add_argument("--outdir", help="ignored if output_gif_dir is specified")
    parser.add_argument("--model", type=str, help="model class name", default='sna')
    parser.add_argument("--model-hparams", type=str, help="a string of comma separated list of model hyperparameters")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    context_length = 2
    results = setup_and_rollout_from_gazebo(args.actions, context_length, args.checkpoint, args.model, args.model_hparams)
    rollout_main(*results, args.outdir, args.fps)


if __name__ == '__main__':
    main()
