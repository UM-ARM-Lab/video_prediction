#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from colorama import Fore

from video_prediction.model import visualize_pixel_rollout, build_model, build_placeholders
from visual_mpc.numpy_point import NumpyPoint


def main():
    np.set_printoptions(suppress=True, linewidth=250, precision=4, threshold=64 * 64 * 3)
    tf.logging.set_verbosity(tf.logging.FATAL)

    context_length = 2

    parser = argparse.ArgumentParser()
    parser.add_argument("row", type=int)
    parser.add_argument("col", type=int)
    parser.add_argument("actions", help='filename')
    parser.add_argument("checkpoint", help="directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000)")
    parser.add_argument("--results_dir", default='results', help="ignored if output_gif_dir is specified")
    parser.add_argument("--model", type=str, help="model class name")
    parser.add_argument("--model_hparams", type=str, help="a string of comma separated list of model hyperparameters")
    parser.add_argument("-s", type=int, default=64)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(Fore.YELLOW + "results directory does not exist. Aborting." + Fore.RESET)
        return

    if args.seed is not None:
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    actions = np.genfromtxt(args.actions, delimiter=',', dtype=np.float32)
    context_pixel_distribs = np.zeros((1, context_length, args.s, args.s, 1), dtype=np.float32)
    source_pixel = NumpyPoint(args.col, args.row)
    context_pixel_distribs[0, 0, args.row, args.col] = 1.0
    context_pixel_distribs[0, 1, args.row, args.col] = 1.0
    res = np.array([1.0 / args.s, 1.0 / args.s])
    origin = np.array([args.s // 2, args.s // 2])
    vx, vy = actions[0]
    dt = 1.0
    context_states = (np.array([[args.col, args.row], [args.col + dt * vx / res[0], args.row + dt * vy / res[0]]]) - origin) * res

    future_length, action_dim = actions.shape
    _, state_dim = context_states.shape
    image_dim = [args.s, args.s, 3]
    total_length = context_length + future_length
    inputs_placeholders = build_placeholders(total_length, state_dim, action_dim, image_dim)

    model = build_model(args.checkpoint, args.model, args.model_hparams, context_length, inputs_placeholders, total_length)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.graph.as_default()

    model.restore(sess, args.checkpoint)

    padded_context_states = np.zeros([1, total_length, state_dim], np.float32)
    padded_context_images = np.zeros([1, total_length, args.s, args.s, 3], np.float32)
    padded_context_pixel_distribs = np.zeros([1, total_length, args.s, args.s, 1], np.float32)
    padded_actions = np.zeros([1, total_length, action_dim], np.float32)
    padded_context_images[0, : context_length] = context_images
    padded_context_states[0, : context_length] = context_states
    padded_context_pixel_distribs[0, : context_length] = context_pixel_distribs
    padded_actions[0, context_length - 1: -1] = actions

    feed_dict = {
        inputs_placeholders['states']: padded_context_states,
        inputs_placeholders['images']: padded_context_images,
        inputs_placeholders['pix_distribs']: padded_context_pixel_distribs,
        inputs_placeholders['actions']: padded_actions,
    }

    fetches = OrderedDict({
        'gen_states': model.outputs['gen_states'],
        'pix_distribs': model.inputs['pix_distribs'],
        'gen_pix_distribs': model.outputs['gen_pix_distribs'],
        'input_states': model.inputs['states'],
    })
    results = sess.run(fetches, feed_dict=feed_dict)

    _ = visualize_pixel_rollout(results, context_length, source_pixel, args)
    plt.show()


if __name__ == '__main__':
    main()
