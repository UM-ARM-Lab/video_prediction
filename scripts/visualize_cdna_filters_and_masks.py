#!/usr/bin/env python
import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from video_prediction import load_data
from video_prediction.model import build_model, build_placeholders
from visual_mpc import gui_tools


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
    parser.add_argument("-s", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    context_states, context_images, actions = load_data(args.images, args.states, args.actions)

    future_length, action_dim = actions.shape
    state_dim = 2
    image_dim = [args.s, args.s, 3]
    total_length = context_length + future_length
    inputs_placeholders = build_placeholders(total_length, state_dim, action_dim, image_dim)
    model = build_model(args.checkpoint, args.model, args.model_hparams, context_length, inputs_placeholders, total_length)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.graph.as_default()

    model.restore(sess, args.checkpoint)

    source_pixel = gui_tools.get_source_pixel(context_images[1])

    context_pixel_distribs = np.zeros((1, context_length, args.s, args.s, 1), dtype=np.float32)
    context_pixel_distribs[0, 0, source_pixel.row, source_pixel.col] = 1.0
    context_pixel_distribs[0, 1, source_pixel.row, source_pixel.col] = 1.0

    padded_context_states = np.zeros([1, total_length, state_dim], np.float32)
    padded_context_images = np.zeros([1, total_length, args.s, args.s, 3], np.float32)
    padded_context_pixel_distribs = np.zeros([1, total_length, args.s, args.s, 1], np.float32)
    padded_actions = np.zeros([1, total_length, action_dim], np.float32)
    padded_context_states[0, :context_length] = context_states
    padded_context_images[0, :context_length] = context_images
    padded_context_pixel_distribs[0, : context_length] = context_pixel_distribs
    padded_actions[0, context_length - 1: -1] = actions

    feed_dict = {
        inputs_placeholders['states']: padded_context_states,
        inputs_placeholders['images']: padded_context_images,
        inputs_placeholders['pix_distribs']: padded_context_pixel_distribs,
        inputs_placeholders['actions']: padded_actions,
    }

    fetches = {
        'cdna_kernels': model.outputs['gen_cdna_kernels'],
        'masks': model.outputs['gen_masks'],
        'extra_image': model.outputs['gen_extra_image'],
    }
    results = sess.run(fetches, feed_dict=feed_dict)
    # time steps, 1, h, w, n_kernels, 1
    initial_kernels = results['cdna_kernels'][0, 0].squeeze()
    extra_image = results['extra_image'].squeeze()
    # time steps, n masks, 1, h, w, 1
    initial_masks = results['masks'][0].squeeze()
    initial_pixel_distrib = context_pixel_distribs[0, 0].squeeze()
    initial_image = context_images[1].squeeze()

    plt.figure()
    plt.title("initial image")
    plt.imshow(initial_image)

    plt.figure()
    plt.title("initial pixel distrib")
    plt.imshow(initial_pixel_distrib, cmap='rainbow')

    n_kernels = initial_kernels.shape[-1]
    fig, axes = plt.subplots(nrows=2, ncols=n_kernels)
    for i in range(n_kernels):
        kernel = initial_kernels[:, :, i]
        axes[0, i].set_title("kernel #{}".format(i))
        axes[0, i].imshow(kernel, cmap='Blues')

    plt.figure()
    plt.title("initial image mask")
    plt.imshow(initial_masks[0])

    plt.figure()
    plt.title("made-up image")
    plt.imshow(extra_image)

    plt.figure()
    plt.title("made-up pixels mask")
    plt.imshow(initial_masks[1], cmap='gray')

    plt.figure()
    plt.title("masked made up image")
    plt.imshow(extra_image * np.expand_dims(initial_masks[1], axis=2))

    n_masks = initial_masks.shape[0]
    # the first mask is the one for the background image
    # the second mask is the made-up pixels
    for i in range(n_masks - 2):
        mask = initial_masks[i + 2]
        axes[1, i].set_title("mask #{}".format(i))
        axes[1, i].imshow(mask, cmap='gray')

    plt.show()


if __name__ == '__main__':
    main()
