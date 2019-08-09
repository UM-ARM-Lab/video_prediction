#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import warnings
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore

from visual_mpc import gui_tools

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf

from video_prediction import load_data
from video_prediction.model import visualize_image_rollout, build_model, build_placeholders, visualize_pixel_rollout


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
    parser.add_argument("--model", type=str, help="model class name", default='sna')
    parser.add_argument("--model_hparams", type=str, help="a string of comma separated list of model hyperparameters")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(Fore.YELLOW + "results directory does not exist. Aborting." + Fore.RESET)
        return

    if args.seed is not None:
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    context_states, context_images, actions = load_data(args.images, args.states, args.actions)
    source_pixel = gui_tools.get_source_pixel(context_images[1])

    actions_length, action_dim = actions.shape
    _, state_dim = context_states.shape
    image_dim = context_images.shape[1:]
    # NOTE: We don't add context_length because that is not how action are fed in. The first action is supposed to be the action
    # that transitions from the first context image to the second context image. Therefore, regardless of the context length,
    # the total number of predicted images will always be 1+ the number of actions. This of course means there's two ways
    # to construct an output sequestion:
    # 1) take the first context image and the rest of the generate images
    # 1) all the context images and only the images generate after the context images (i.e. warm start done)
    total_length = 1 + context_length
    inputs_placeholders = build_placeholders(total_length, actions_length, state_dim, action_dim, image_dim)

    context_pixel_distribs = np.zeros((1, context_length, image_dim[0], image_dim[1], 1), dtype=np.float32)
    context_pixel_distribs[0, 0, source_pixel.row, source_pixel.col] = 1.0
    context_pixel_distribs[0, 1, source_pixel.row, source_pixel.col] = 1.0

    model = build_model(args.checkpoint, args.model, args.model_hparams, context_length, inputs_placeholders, total_length)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.graph.as_default()

    model.restore(sess, args.checkpoint)

    padded_context_states = np.zeros([1, total_length, state_dim], np.float32)
    padded_context_images = np.zeros([1, total_length, *image_dim], np.float32)
    padded_context_pixel_distribs = np.zeros([1, total_length, image_dim[0], image_dim[1], 1], np.float32)
    padded_actions = np.zeros([1, actions_length, action_dim], np.float32)

    padded_context_states[0, :context_length] = context_states
    padded_context_images[0, :context_length] = context_images
    padded_context_pixel_distribs[0, :context_length] = context_pixel_distribs
    padded_actions[0] = actions

    feed_dict = {
        inputs_placeholders['states']: padded_context_states,
        inputs_placeholders['images']: padded_context_images,
        inputs_placeholders['actions']: padded_actions,
        inputs_placeholders['pix_distribs']: padded_context_pixel_distribs,
    }

    fetches = OrderedDict({
        'gen_images': model.outputs['gen_images'],
        'gen_states': model.outputs['gen_states'],
        'gen_pix_distribs': model.outputs['gen_pix_distribs'],
        'pix_distribs': model.inputs['pix_distribs'],
        'input_images': model.inputs['images'],
        'input_states': model.inputs['states'],
    })
    results = sess.run(fetches, feed_dict=feed_dict)

    handles = []
    handles.extend(visualize_image_rollout(results, context_length, args))
    handles.extend(visualize_pixel_rollout(results, context_length, source_pixel, args))
    plt.show()


if __name__ == '__main__':
    main()
