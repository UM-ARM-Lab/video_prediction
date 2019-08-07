#!/usr/bin/env python
import argparse
import os

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
    parser.add_argument("--outdir", default='results', help="ignored if output_gif_dir is specified")
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
        'cdna_outputs': model.outputs['gen_cdna_outputs'],
        'masks': model.outputs['gen_masks'],
        'extra_image': model.outputs['gen_extra_image'],
    }
    results = sess.run(fetches, feed_dict=feed_dict)
    # time steps, 1, h, w, n_kernels, 1
    initial_kernels = results['cdna_kernels'][0, 0].squeeze()
    initial_outputs = results['cdna_outputs'][0].squeeze()
    extra_image = results['extra_image'].squeeze()
    # time steps, n masks, 1, h, w, 1
    initial_masks = results['masks'][0].squeeze()
    initial_pixel_distrib = context_pixel_distribs[0, 0].squeeze()
    initial_image = context_images[1].squeeze()

    n_outputs = initial_outputs.shape[0]
    n_kernels = initial_kernels.shape[-1]
    fig, axes = plt.subplots(nrows=3, ncols=n_outputs)
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(n_kernels):
        kernel = initial_kernels[:, :, i]
        axes[0, i].set_title("kernel #{}".format(i))
        # Don't set vmin/vmax here, we let matplotlib normalize these here since they only show relative "motion"
        axes[0, i].imshow(kernel, cmap='Blues')
    axes[0, -1].set_title("made-up image")
    axes[0, -1].imshow(extra_image, vmin=0, vmax=255)

    n_masks = initial_masks.shape[0]
    # the first mask is the one for the background image
    # the second mask is the made-up 'extra' image
    for i in range(n_masks - 2):
        mask = initial_masks[i + 2]
        axes[1, i].set_title("mask #{}".format(i))
        axes[1, i].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[1, -1].set_title("made-up image mask")
    axes[1, -1].imshow(initial_masks[1], cmap='gray', vmin=0, vmax=1)

    for i in range(n_outputs):
        output = initial_outputs[i]
        axes[2, i].set_title("transformed #{}".format(i))
        axes[2, i].imshow(output)
    axes[2, -1].set_title("made up transformed")
    axes[2, -1].imshow(initial_outputs[-1])
    plt.savefig(os.path.join(args.outdir, 'transformations.png'))

    fig, axes = plt.subplots(nrows=2, ncols=3)
    axes[0, 0].set_title("untransformed background image")
    axes[0, 0].imshow(initial_image)
    axes[0, 1].title("background mask")
    axes[0, 1].imshow(initial_masks[0], cmap='gray', vmin=0, vmax=1)
    axes[0, 2].title("background image, transformed then masked")
    background_image_masked = initial_outputs[0]
    axes[0, 2].imshow(background_image_masked, cmap='gray', vmin=0, vmax=1)

    axes[1, 0].set_title("background pixel distrib")
    axes[1, 0].imshow(initial_pixel_distrib, cmap='rainbow', vmin=0, vmax=1)
    axes[1, 1].set_title("background mask")
    axes[1, 1].imshow(initial_masks[0], cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title("background pixel distrib masked")
    background_pix_masked = initial_outputs[0]
    axes[0, 2].imshow(background_pix_masked, cmap='gray', vmin=0, vmax=1)

    axes[0, 2].set_title("made-up image")
    axes[0, 2].imshow(extra_image, vmin=0, vmax=255)

    axes[1, 2].set_title("masked made up image")
    axes[1, 2].imshow(extra_image * np.expand_dims(initial_masks[1], axis=2), vmin=0, vmax=255)

    plt.show()


if __name__ == '__main__':
    main()
