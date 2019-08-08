#!/usr/bin/env python
import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation

from video_prediction import load_data
from video_prediction.model import build_model, build_placeholders
# noinspection PyUnresolvedReferences
from visual_mpc import gui_tools
# noinspection PyUnresolvedReferences
from visual_mpc.numpy_point import NumpyPoint


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

    results = setup_and_run(args, context_length)

    for k, v in results.items():
        print("{:40s}: {}".format(k, v.shape))

    # the occasional first 0 indexing here is due to the batch size of 1
    # second index is the time step, in this case the last "context" time step
    second_context_image = results['input_images'][0, 1].squeeze()
    second_context_pix_distrib = results['input_pix_distribs'][0, 1].squeeze()
    kernels = results['gen_cdna_kernels'][0, 1].squeeze()
    masks = results['gen_masks'][0, 1].squeeze()
    transformed_images = results['gen_transformed_images'][0, 1].squeeze()
    masked_images = results['gen_masked_images'][0, 1].squeeze()
    transformed_pix_distribs = results['gen_transformed_pix_distribs'][0, 1].squeeze()
    masked_pix_distribs = results['gen_masked_pix_distribs'][0, 1].squeeze()
    extra_image = results['gen_made_up_images'][0, 1].squeeze()
    gen_images = results['gen_images'][0, 1].squeeze()
    gen_pix_distribs = results['gen_pix_distribs'][0, 1].squeeze()

    background_pix_masked = masked_pix_distribs[0]

    # first dimension of gen_masked_images is the different output images
    # the first one is the background image masked, the second is the made-up image masked,
    # and the rest correspond to the different CDNA kernels
    background_image_masked = masked_images[0]

    # Configure matplotlib
    mpl.rcParams['figure.subplot.wspace'] = 0.1
    mpl.rcParams['figure.subplot.hspace'] = 0.1
    mpl.rcParams['figure.titlesize'] = 7
    mpl.rcParams['figure.figsize'] = (12.8, 9.6)
    mpl.rcParams['figure.dpi'] = 100
    mpl.rcParams['axes.formatter.useoffset'] = False
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['figure.titlesize'] = 7
    mpl.rcParams['legend.facecolor'] = 'white'
    mpl.rcParams['font.size'] = 7

    ############################################
    # Background and Made-Up Image Visualization
    ############################################
    # the first mask is the one for the background image
    # the second mask is the made-up 'extra' image
    fig, axes = plt.subplots(nrows=3, ncols=3, gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    # the background image is never transformed, just masked
    axes[0, 0].set_title("background image")
    axes[0, 0].imshow(second_context_image, vmin=0, vmax=1)
    axes[0, 1].set_title("background mask")
    axes[0, 1].imshow(masks[0], cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title("background image, masked")
    axes[0, 2].imshow(background_image_masked, vmin=0, vmax=1)

    axes[1, 0].set_title("background pix distrib")
    axes[1, 0].imshow(second_context_pix_distrib, cmap='rainbow', vmin=0, vmax=1)
    axes[1, 1].set_title("background mask")
    axes[1, 1].imshow(masks[0], cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title("background pix distrib masked")
    axes[1, 2].imshow(background_pix_masked, cmap='gray', vmin=0, vmax=1)

    axes[2, 0].set_title("made-up image")
    axes[2, 0].imshow(extra_image, vmin=0, vmax=1)
    axes[2, 1].set_title("made-up image mask")
    axes[2, 1].imshow(masks[1], cmap='gray', vmin=0, vmax=1)
    axes[2, 2].set_title("masked made up image")
    axes[2, 2].imshow(masked_images[1], vmin=0, vmax=1)

    plt.savefig(os.path.join(args.outdir, 'other_figure.png'))

    ############################################
    # CDNA Visualization on the real images
    ############################################

    n_outputs = masked_images.shape[0]
    fig, axes = plt.subplots(nrows=4, ncols=n_outputs - 2, gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(n_outputs - 2):
        kernel = kernels[i]
        kernel = kernel / np.linalg.norm(kernel)
        axes[0, i].set_title("kernel #{}".format(i))
        # Don't set vmin/vmax here, we let matplotlib normalize these here since they only show relative "motion"
        axes[0, i].imshow(kernel, cmap='Blues', vmin=0, vmax=1)

    transformed_image_handles = []
    step_text_handles = []
    for i in range(n_outputs - 2):
        axes[1, i].set_title("transformed #{}".format(i))
        transformed_image_handle = axes[1, i].imshow(second_context_image, vmin=0, vmax=1)
        transformed_image_handles.append(transformed_image_handle)
        step_text_handle = axes[1, i].text(5, 8, 't=0', fontdict={'color': 'white', 'size': 5},
                                           bbox=dict(facecolor='black', alpha=0.5))
        step_text_handles.append(step_text_handle)

    def transformed_image_update(j):
        for i in range(n_outputs - 2):
            # the background image is not transformed, so just skip the first image which is the transformed "made-up" image
            step_text_handles[i].set_text("t={}".format(j))
            if j == 0:
                transformed_image_handles[i].set_data(second_context_image)
            else:
                transformed_image_handles[i].set_data(np.clip(transformed_images[i + 1], 0, 1))

    transformed_image_anim = FuncAnimation(fig, transformed_image_update, frames=2, interval=1000, repeat=True)

    for i in range(n_outputs - 2):
        mask = masks[i + 2]
        axes[2, i].set_title("mask #{}".format(i))
        axes[2, i].imshow(mask, cmap='gray', vmin=0, vmax=1)

    for i in range(n_outputs - 2):
        output = masked_images[i + 2]
        axes[3, i].set_title("masked #{}".format(i))
        axes[3, i].imshow(output, vmin=0, vmax=1)

    ############################################
    # CDNA Visualization on the pixel distributions
    ############################################

    fig, axes = plt.subplots(nrows=4, ncols=n_outputs - 2, gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(n_outputs - 2):
        kernel = kernels[i]
        kernel = kernel / np.linalg.norm(kernel)
        axes[0, i].set_title("kernel #{}".format(i))
        # Don't set vmin/vmax here, we let matplotlib normalize these here since they only show relative "motion"
        axes[0, i].imshow(kernel, cmap='Blues', vmin=0, vmax=1)

    transformed_pix_distrib_handles = []
    step_text_handles = []
    for i in range(n_outputs - 2):
        axes[1, i].set_title("transformed #{}".format(i))
        transformed_pix_distrib_handle = axes[1, i].imshow(second_context_pix_distrib, vmin=0, vmax=1)
        transformed_pix_distrib_handles.append(transformed_pix_distrib_handle)
        step_text_handle = axes[1, i].text(5, 8, 't=0', fontdict={'color': 'white', 'size': 5},
                                           bbox=dict(facecolor='black', alpha=0.5))
        step_text_handles.append(step_text_handle)

    def transformed_pix_distrib_update(j):
        for i in range(n_outputs - 2):
            # Note: there's no "made-up" image or background image in transformed_pix_distribs
            step_text_handles[i].set_text("t={}".format(j))
            if j == 0:
                transformed_pix_distrib_handles[i].set_data(second_context_pix_distrib)
            else:
                transformed_pix_distrib_handles[i].set_data(np.clip(transformed_pix_distribs[i], 0, 1))

    transformed_pix_distrib_anim = FuncAnimation(fig, transformed_pix_distrib_update, frames=2, interval=1000, repeat=True)

    for i in range(n_outputs - 2):
        mask = masks[i + 2]
        axes[2, i].set_title("mask #{}".format(i))
        axes[2, i].imshow(mask, cmap='gray', vmin=0, vmax=1)

    for i in range(n_outputs - 2):
        output = masked_pix_distribs[i + 2]
        axes[3, i].set_title("masked #{}".format(i))
        axes[3, i].imshow(output, vmin=0, vmax=1)

    ############################################
    # Animate total effect of transforming and masking on pixel distrib and real images
    ############################################
    fig, axes = plt.subplots(nrows=2, ncols=n_outputs - 1, gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    masked_image_handles = []
    masked_pix_distrib_handles = []
    image_step_text_handles = []
    pix_distrib_step_text_handles = []
    for i in range(n_outputs - 2):
        axes[1, i].set_title("masked #{}".format(i))
        masked_image_handle = axes[0, i].imshow(second_context_image, vmin=0, vmax=1)
        masked_image_handles.append(masked_image_handle)
        image_step_text_handle = axes[0, i].text(5, 8, 't=0', fontdict={'color': 'white', 'size': 5},
                                                 bbox=dict(facecolor='black', alpha=0.5))
        image_step_text_handles.append(image_step_text_handle)

        masked_pix_distrib_handle = axes[1, i].imshow(second_context_pix_distrib, vmin=0, vmax=1)
        masked_pix_distrib_handles.append(masked_pix_distrib_handle)
        pix_distrib_step_text_handle = axes[1, i].text(5, 8, 't=0', fontdict={'color': 'white', 'size': 5},
                                                       bbox=dict(facecolor='black', alpha=0.5))
        pix_distrib_step_text_handles.append(pix_distrib_step_text_handle)

        combined_image_handle = axes[0, i].imshow(second_context_image, vmin=0, vmax=1)
        combined_pix_distrib_handle = axes[0, i].imshow(second_context_pix_distrib, vmin=0, vmax=1)

    def masked_update(j):
        for i in range(n_outputs - 2):
            image_step_text_handles[i].set_text("t={}".format(j))
            pix_distrib_step_text_handles[i].set_text("t={}".format(j))
            if j == 0:
                masked_image_handles[i].set_data(second_context_image)
                masked_pix_distrib_handles[i].set_data(second_context_pix_distrib)
                combined_image_handle.set_data(second_context_image)
                combined_pix_distrib_handle.set_data(second_context_pix_distrib)
            else:
                masked_image_handles[i].set_data(np.clip(masked_images[i + 2], 0, 1))
                masked_pix_distrib_handles[i].set_data(np.clip(masked_pix_distribs[i + 2], 0, 1))
                combined_pix_distrib_handle.set_data(np.clip(gen_pix_distribs, 0, 1))
                combined_image_handle.set_data(np.clip(gen_images, 0, 1))

    masked_anim = FuncAnimation(fig, masked_update, frames=2, interval=1000, repeat=True)

    ############################################
    # Show and save
    ############################################
    if args.outdir:
        transformed_image_anim.save(os.path.join(args.outdir, 'transformed_image_figure.gif'), writer='imagemagick', dpi=fig.dpi)
        masked_anim.save(os.path.join(args.outdir, 'masked_image_figure.gif'), writer='imagemagick', dpi=fig.dpi)
        transformed_pix_distrib_anim.save(os.path.join(args.outdir, 'transformed_pix_distrib_figure.gif'), writer='imagemagick',
                                          dpi=fig.dpi)

    plt.show()


def setup_and_run(args, context_length):
    context_states, context_images, actions = load_data(args.images, args.states, args.actions)
    future_length, action_dim = actions.shape
    state_dim = 2
    image_dim = [args.s, args.s, 3]
    total_length = context_length + future_length
    print('Total Time Length:', total_length)
    inputs_placeholders = build_placeholders(total_length, state_dim, action_dim, image_dim)
    model = build_model(args.checkpoint, args.model, args.model_hparams, context_length, inputs_placeholders, total_length)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.graph.as_default()
    model.restore(sess, args.checkpoint)

    # source_pixel = gui_tools.get_source_pixel(context_images[1])
    source_pixel = NumpyPoint(32, 32)

    context_pix_distribs = np.zeros((1, context_length, args.s, args.s, 1), dtype=np.float32)
    context_pix_distribs[0, 0, source_pixel.row, source_pixel.col] = 1.0
    context_pix_distribs[0, 1, source_pixel.row, source_pixel.col] = 1.0
    padded_context_states = np.zeros([1, total_length, state_dim], np.float32)
    padded_context_images = np.zeros([1, total_length, args.s, args.s, 3], np.float32)
    padded_context_pix_distribs = np.zeros([1, total_length, args.s, args.s, 1], np.float32)
    padded_actions = np.zeros([1, total_length, action_dim], np.float32)
    padded_context_states[0, :context_length] = context_states
    padded_context_images[0, :context_length] = context_images
    padded_context_pix_distribs[0, : context_length] = context_pix_distribs
    padded_actions[0, context_length - 1: -1] = actions
    feed_dict = {
        inputs_placeholders['states']: padded_context_states,
        inputs_placeholders['images']: padded_context_images,
        inputs_placeholders['pix_distribs']: padded_context_pix_distribs,
        inputs_placeholders['actions']: padded_actions,
    }
    fetches = {
        'gen_images': model.outputs['gen_images'],
        'gen_pix_distribs': model.outputs['gen_pix_distribs'],
        'gen_cdna_kernels': model.outputs['gen_cdna_kernels'],
        'gen_masked_images': model.outputs['gen_masked_images'],
        'gen_masks': model.outputs['gen_masks'],
        'gen_made_up_images': model.outputs['gen_made_up_images'],
        'gen_masked_pix_distribs': model.outputs['gen_masked_pix_distribs'],
        'gen_transformed_pix_distribs': model.outputs['gen_transformed_pix_distribs'],
        'gen_transformed_images': model.outputs['gen_transformed_images'],
        'input_images': model.inputs['images'],
        'input_pix_distribs': model.inputs['pix_distribs'],
    }
    results = sess.run(fetches, feed_dict=feed_dict)
    return results


if __name__ == '__main__':
    main()
