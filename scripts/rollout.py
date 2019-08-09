#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation

from video_prediction import load_data
from video_prediction.model_for_planning import build_model, build_placeholders, build_feed_dict
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
    parser.add_argument("--outdir", help="ignored if output_gif_dir is specified")
    parser.add_argument("--model", type=str, help="model class name", default='sna')
    parser.add_argument("--model-hparams", type=str, help="a string of comma separated list of model hyperparameters")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.seed is not None:
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    context_states, context_images, actions = load_data(args.images, args.states, args.actions)
    source_pixel0 = gui_tools.get_source_pixel(context_images[0])
    source_pixel1, target_pixel = gui_tools.get_pixels(context_images[1])

    actions_length, action_dim = actions.shape
    _, state_dim = context_states.shape
    _, h, w, d = context_images.shape
    placeholders, sequence_length = build_placeholders(context_length, actions_length, h, w, d, state_dim, action_dim)

    context_pixel_distribs = np.zeros((1, context_length, h, w, 1), dtype=np.float32)
    context_pixel_distribs[0, 0, source_pixel0.row, source_pixel0.col] = 1.0
    context_pixel_distribs[0, 1, source_pixel1.row, source_pixel1.col] = 1.0

    model = build_model(args.checkpoint, args.model, args.model_hparams, placeholders, context_length, sequence_length)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.graph.as_default()

    model.restore(sess, args.checkpoint)

    feed_dict = build_feed_dict(placeholders, context_images, context_states, context_pixel_distribs, actions,
                                sequence_length)
    fetches = {
        'gen_images': model.outputs['gen_images'],
        'gen_states': model.outputs['gen_states'],
        'gen_pix_distribs': model.outputs['gen_pix_distribs'],
        'pix_distribs': model.inputs['pix_distribs'],
        'input_images': model.inputs['images'],
        'input_states': model.inputs['states'],
    }
    results = sess.run(fetches, feed_dict=feed_dict)

    # there is a choice of how to combine context and generate frames, but I'm going to
    # choose the prettier one which is to show all the context images and leave our the gen_images
    # which are for the same time steps
    # EX: take the first two frames from context, then skip the 1st frame of output
    # since that corresponds to the second context image, and take all the rest of the generated images
    context_images = results['input_images'][0, :context_length].squeeze()
    context_pix_distribs = results['pix_distribs'][0, :context_length].squeeze()
    gen_images = results['gen_images'][0, context_length - 1:].squeeze()
    gen_pix_distribs = results['gen_pix_distribs'][0, context_length - 1:].squeeze()

    full_image_sequence = np.concatenate((context_images, gen_images))
    full_pix_distrib_sequence = np.concatenate((context_pix_distribs, gen_pix_distribs))

    time_data = np.arange(full_pix_distrib_sequence.shape[0])
    probability_of_designated_pixel = np.zeros(full_pix_distrib_sequence.shape[0])
    for t, pix_distrib in enumerate(full_pix_distrib_sequence):
        probability_of_designated_pixel[t] = pix_distrib[source_pixel1.row, source_pixel1.col]

    ##########
    # Plotting
    ##########
    fig, axes = plt.subplots(nrows=1, ncols=3)

    axes[0].set_title("prediction [image]")
    image_handle = axes[0].imshow(full_image_sequence[0], cmap='rainbow')

    axes[1].set_title("prediction [pix distrib]")
    axes[1].scatter(target_pixel.col, target_pixel.row, marker='D', c='y', s=5)
    pix_distrib_handle = axes[1].imshow(full_pix_distrib_sequence[0], cmap='rainbow')
    axes[2].set_title("P(selected pixel)")
    axes[2].set_xlabel("time (step #)")
    axes[2].set_ylabel("probability of designated pixel")
    axes[2].plot(time_data, probability_of_designated_pixel)
    axes[2].set_xlim(0, sequence_length - 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_aspect(sequence_length - 1)
    current_probability_scatter_handle = axes[2].scatter(0, 1, s=20, c='r')

    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].axis("off")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].axis("off")

    def update(t):
        pix_distrib_handle.set_data(full_pix_distrib_sequence[t])
        image_handle.set_data(np.clip(full_image_sequence[t], 0, 1))

        current_probability_scatter_handle.set_offsets([time_data[t], probability_of_designated_pixel[t]])

    t = 0

    def on_key_release(event):
        nonlocal t
        if event.key == 'right':
            if t < sequence_length - 1:
                t += 1
        if event.key == 'left':
            if t > 0:
                t -= 1

        update(t)

        fig.canvas.draw()
        fig.canvas.flush_events()

    fig.canvas.mpl_connect('key_release_event', on_key_release)

    plt.show()

    if args.outdir:
        anim = FuncAnimation(fig, update, frames=sequence_length, interval=1000 / args.fps, repeat=True)
        anim.save(os.path.join(args.outdir, 'rollout.gif'), writer='imagemagick')


if __name__ == '__main__':
    main()
