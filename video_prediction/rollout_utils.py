#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation

from link_bot_gazebo import gazebo_utils
from video_prediction.datasets.dataset_utils import load_data
from video_prediction.model_for_planning import build_model, build_placeholders, build_feed_dict, rollouts_from_results
from visual_mpc import gui_tools


def setup_and_rollout_from_gazebo(actions_filename, context_length, checkpoint, model, model_hparams):
    state_dim = 6
    sna_model_action_dim = 2
    services = gazebo_utils.GazeboServices()

    context_images, context_states, context_actions = services.get_context(context_length, state_dim, sna_model_action_dim)

    actions = np.atleast_2d(np.genfromtxt(actions_filename, delimiter=',', dtype=np.float32))

    return setup_and_rollout(context_images, context_states, context_actions, actions, context_length, checkpoint, model,
                             model_hparams)


def setup_and_rollout_from_individual_files(images, states, context_actions, actions, context_length, checkpoint, model,
                                            model_hparams):
    context_states, context_images, context_actions, actions = load_data(images, states, context_actions, actions)
    return setup_and_rollout(context_images, context_states, context_actions, actions, context_length, checkpoint, model,
                             model_hparams)


def setup_and_rollout(context_images, context_states, context_actions, actions, context_length, checkpoint, model, model_hparams):
    source_pixel0 = gui_tools.get_source_pixel(context_images[0])
    assert source_pixel0 is not None

    source_pixel1 = gui_tools.get_source_pixel(context_images[1])
    assert source_pixel1 is not None

    actions_length, action_dim = actions.shape
    _, state_dim = context_states.shape
    _, image_h, image_w, image_d = context_images.shape
    placeholders, sequence_length = build_placeholders(context_length, actions_length, image_h, image_w, image_d, state_dim,
                                                       action_dim)

    context_pixel_distribs = np.zeros((context_length, image_h, image_w, 1), dtype=np.float32)
    context_pixel_distribs[0, source_pixel0.row, source_pixel0.col] = 1.0
    context_pixel_distribs[1, source_pixel1.row, source_pixel1.col] = 1.0

    model = build_model(checkpoint, model, model_hparams, placeholders, context_length, sequence_length)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.graph.as_default()

    model.restore(sess, checkpoint)

    feed_dict = build_feed_dict(placeholders, context_images, context_states, context_pixel_distribs, context_actions, actions,
                                sequence_length)
    fetches = {
        'input_images': model.inputs['images'],
        'input_states': model.inputs['states'],
        'input_pix_distribs': model.inputs['pix_distribs'],

        'gen_images': model.outputs['gen_images'],
        'gen_states': model.outputs['gen_states'],
        'gen_pix_distribs': model.outputs['gen_pix_distribs'],
    }
    results = sess.run(fetches, feed_dict=feed_dict)

    pix_distrib_sequence, image_sequence = rollouts_from_results(results, context_length)

    probability_of_designated_pixel = np.zeros(pix_distrib_sequence.shape[0])
    for t, pix_distrib in enumerate(pix_distrib_sequence):
        probability_of_designated_pixel[t] = pix_distrib[source_pixel1.row, source_pixel1.col]

    return image_sequence, pix_distrib_sequence, sequence_length, image_h, image_w


def rollout_main(image_sequence, pix_distrib_sequence, sequence_length, image_h, image_w, outdir, fps):
    gui_tools.configure_matplotlib()

    fig, axes = plt.subplots(nrows=1, ncols=2)

    axes[0].set_title("prediction [image]")
    image_handle = axes[0].imshow(image_sequence[0])

    axes[1].set_title("prediction [pix distrib]")
    pix_distrib_handle = axes[1].imshow(pix_distrib_sequence[0], cmap='rainbow')

    cc, rr = np.meshgrid(np.arange(image_h), np.arange(image_w))
    pixel_positions = np.tile(np.stack((rr, cc), axis=2), [pix_distrib_sequence.shape[0], 1, 1, 1])
    weighted_positions = np.expand_dims(pix_distrib_sequence, axis=3) * pixel_positions
    expected_positions = np.sum(np.sum(weighted_positions, axis=2), axis=1)
    expected_pos_pix_distrib_handle = axes[1].scatter(expected_positions[0, 1], expected_positions[0, 0], c='orange', marker='*',
                                                      s=1)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].axis("off")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].axis("off")

    def update(t):
        pix_distrib_handle.set_data(pix_distrib_sequence[t])
        # TODO: clip or normalize?
        image_handle.set_data(np.clip(image_sequence[t], 0, 1))
        expected_position = np.flip(expected_positions[t])
        expected_pos_pix_distrib_handle.set_offsets(expected_position)

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

    if outdir:
        anim = FuncAnimation(fig, update, frames=sequence_length, interval=1000 / fps, repeat=True)
        anim.save(os.path.join(outdir, 'rollout.gif'), writer='imagemagick')
        del anim  # this makes it so the animation will stop running, so that the arrow keys can work
        t = 0

    plt.show()
