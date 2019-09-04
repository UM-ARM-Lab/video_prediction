#!/usr/bin/env python
import argparse

import numpy as np
import tensorflow as tf

from video_prediction import visualization
from video_prediction.visualization import setup_and_run_from_individual_files


def main():
    np.set_printoptions(suppress=True, linewidth=250, precision=4, threshold=64 * 64 * 3)
    tf.logging.set_verbosity(tf.logging.FATAL)

    context_length = 2
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs=context_length, help='filename')
    parser.add_argument("states", help='filename')
    parser.add_argument("context_actions", help='filename')
    parser.add_argument("actions", help='filename')
    parser.add_argument("checkpoint", help="directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000)")
    parser.add_argument("--outdir", help="ignored if output_gif_dir is specified")
    parser.add_argument("--model", type=str, help="model class name", default='sna')
    parser.add_argument("--model_hparams", type=str, help="a string of comma separated list of model hyperparameters")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--show-combined-masks", type=str, default='', help='comma seperated list of integers from 0-9')
    parser.add_argument("-t", type=int, default=0)
    parser.add_argument("-s", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    context_length = 2
    results, has_made_up = setup_and_run_from_individual_files(args.images,
                                                               args.states,
                                                               args.context_actions,
                                                               args.actions,
                                                               context_length,
                                                               args.checkpoint,
                                                               args.model,
                                                               args.model_hparams)

    visualization.visualize_main(results, has_made_up, context_length, args.t, args.outdir, args.show_combined_masks)


if __name__ == '__main__':
    main()
