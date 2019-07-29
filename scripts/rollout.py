#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import warnings
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf

from video_prediction import load_data
from video_prediction.model import visualize, build_model, build_placeholders


def main():
    np.set_printoptions(suppress=True, linewidth=250, precision=4, threshold=64 * 64 * 3)
    tf.logging.set_verbosity(tf.logging.FATAL)

    context_length = 2

    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs=context_length, help='filename')
    parser.add_argument("states", help='filename')
    parser.add_argument("actions", help='filename')
    parser.add_argument("checkpoint", help="directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000)")
    parser.add_argument("--results_dir", default='results', help="ignored if output_gif_dir is specified")
    parser.add_argument("--model", type=str, help="model class name")
    parser.add_argument("--model_hparams", type=str, help="a string of comma separated list of model hyperparameters")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.seed is not None:
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    context_states, context_images, actions = load_data(args.images, args.states, args.actions)

    future_length, action_dim = actions.shape
    _, state_dim = context_states.shape
    image_dim = context_images.shape[1:]
    total_length = context_length + future_length
    inputs_placeholders = build_placeholders(total_length, state_dim, action_dim, image_dim)

    model = build_model(args.checkpoint, args.model, args.model_hparams, context_length, inputs_placeholders, total_length)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.graph.as_default()

    model.restore(sess, args.checkpoint)

    padded_context_states = np.zeros([1, total_length, state_dim], np.float32)
    padded_context_images = np.zeros([1, total_length, *image_dim], np.float32)
    padded_actions = np.zeros([1, total_length, action_dim], np.float32)
    padded_context_states[0, : context_length] = context_states
    padded_context_images[0, : context_length] = context_images
    padded_actions[0, context_length - 1: -1] = actions

    feed_dict = {
        inputs_placeholders['states']: padded_context_states,
        inputs_placeholders['images']: padded_context_images,
        inputs_placeholders['actions']: padded_actions,
    }

    fetches = OrderedDict({
        'gen_images': model.outputs['gen_images'],
        'gen_states': model.outputs['gen_states'],
        'input_images': model.inputs['images'],
        'input_states': model.inputs['states'],
    })
    results = sess.run(fetches, feed_dict=feed_dict)

    visualize(results, context_length, args)
    plt.show()


if __name__ == '__main__':
    main()
