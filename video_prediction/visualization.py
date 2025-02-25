import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation

from link_bot_gazebo import gazebo_utils
from video_prediction.datasets.dataset_utils import load_data
from video_prediction.model_for_planning import build_model, build_placeholders, build_feed_dict, make_context_pix_distribs
from visual_mpc import gui_tools


def visualize_main(results, has_made_up, context_length, t, outdir, show_combined_masks):
    for k, v in results.items():
        print("{:40s}: {}".format(k, v.shape))

    # the occasional first 0 indexing here is due to the batch size of 1
    # second index is the time step, in this case the last "context" time step
    # t=0 is the first generates image, and so the first context image would be t=-1
    # by writing t=n we are asking to visualize how image at t=n was generated
    assert t >= 0
    kernels = results['cdna_kernels'][0, t].squeeze()
    masks = results['masks'][0, t].squeeze()

    background_image = results['background_images'][0, t].squeeze()
    images_transformed = results['transformed_images'][0, t].squeeze()
    masked_images = results['masked_images'][0, t].squeeze()
    images_fused = results['fused_images'][0, t].squeeze()

    background_pix_distrib = results['background_pix_distribs'][0, t].squeeze()
    transformed_pix_distribs = results['transformed_pix_distribs'][0, t].squeeze()
    masked_pix_distribs = results['masked_pix_distribs'][0, t].squeeze()
    pix_distribs_fused = results['fused_pix_distribs'][0, t].squeeze()

    if has_made_up:
        made_up_image = results['made_up_images'][0, t].squeeze()
        made_up_pix_distrib = results['made_up_pix_distribs'][0, t].squeeze()
    else:
        made_up_image = None
        made_up_pix_distrib = None

    # NOTE: first dimension of masked_images is the different output images
    #  the first one is the background image masked, args.the second is the made-up image masked,
    #  and the rest correspond to the different CDNA kernels
    background_masked_image = masked_images[0]
    background_pix_distrib_masked = masked_pix_distribs[0]
    extra_masks = 2 if has_made_up else 1
    if has_made_up:
        made_up_masked_images = masked_images[1]
        made_up_pix_distrib_masked = masked_pix_distribs[1]
    else:
        made_up_masked_images = None
        made_up_pix_distrib_masked = None

    if t < context_length:
        prev_image = results['input_images'][0, t].squeeze()
        prev_pix_distrib = results['input_pix_distribs'][0, t].squeeze()
    else:
        prev_image = results['fused_images'][0, t - 1].squeeze()
        prev_pix_distrib = results['fused_pix_distribs'][0, t - 1].squeeze()

    n_kernels = kernels.shape[0]

    gui_tools.configure_matplotlib()

    ############################################
    # Background and Made-Up Image Visualization
    ############################################
    non_motion_viz(background_masked_image,
                   background_pix_distrib_masked,
                   made_up_pix_distrib,
                   made_up_image,
                   made_up_masked_images,
                   masks,
                   made_up_pix_distrib_masked,
                   background_image,
                   background_pix_distrib,
                   has_made_up)

    if outdir:
        plt.savefig(os.path.join(outdir, 'non_background_fig.png'))

    #######################################
    # CDNA Visualization on the real images
    #######################################
    transformed_image_anim = cdna_image_viz(kernels,
                                            masked_images,
                                            masks,
                                            prev_image,
                                            images_transformed,
                                            n_kernels,
                                            extra_masks,
                                            )

    ###############################################
    # CDNA Visualization on the pixel distributions
    ###############################################
    transformed_pix_distrib_anim = cdna_pix_distrib_viz(kernels,
                                                        masked_pix_distribs,
                                                        masks,
                                                        transformed_pix_distribs,
                                                        prev_pix_distrib,
                                                        n_kernels,
                                                        extra_masks,
                                                        )

    ################################################################################
    # Show total effect of transforming and masking on pixel distrib and real images
    ################################################################################
    combination_viz(background_masked_image,
                    background_pix_distrib_masked,
                    prev_image,
                    prev_pix_distrib,
                    images_fused,
                    pix_distribs_fused,
                    masked_images,
                    masked_pix_distribs,
                    made_up_pix_distrib_masked,
                    background_image,
                    background_pix_distrib,
                    n_kernels,
                    has_made_up,
                    extra_masks,
                    )

    ##############################################################
    # Show the context image and the made-up masked image overlaid
    ##############################################################
    if has_made_up:
        prev_vs_made_up_viz(prev_image, made_up_masked_images)

    ##########################
    # Show some masks combined
    ##########################
    if show_combined_masks != '':
        indeces = [int(i) for i in show_combined_masks.split(",")]
        combined_masks_viz(prev_image, masks, indeces, has_made_up)

    ###############
    # Show and save
    ###############
    if outdir:
        transformed_image_anim.save(os.path.join(outdir, 'transformed_image_figure.gif'), writer='imagemagick')
        transformed_pix_distrib_anim.save(os.path.join(outdir, 'transformed_pix_distrib_figure.gif'), writer='imagemagick')

    plt.show()


def imshow_madeup_if_has(ax, image, has_made_up, **kwargs):
    if has_made_up:
        ax.imshow(image, **kwargs)
    else:
        ax.plot([0, 64], [0, 64], c='r')
        ax.plot([0, 0, 64, 64, 0], [0, 64, 64, 0, 0], c='gray')
        ax.set_xlim(0, 64)
        ax.set_ylim(0, 64)
        ax.set_aspect(1)


def non_motion_viz(background_masked_image: np.ndarray,
                   background_pix_distrib_masked: np.ndarray,
                   made_up_pix_distrib: np.ndarray,
                   made_up_image: np.ndarray,
                   made_up_masked_image: np.ndarray,
                   masks: np.ndarray,
                   made_up_pix_distrib_masked: np.ndarray,
                   background_image: np.ndarray,
                   background_pix_distrib: np.ndarray,
                   has_made_up: bool,
                   ):
    # the first mask is the one for the background image
    # the second mask is the made-up 'extra' image
    fig, axes = plt.subplots(nrows=4, ncols=3, gridspec_kw={'wspace': 0.1, 'hspace': 0.2})
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
    # the background image is never transformed, just masked
    axes[0, 0].set_title("background image")
    axes[0, 0].imshow(background_image, vmin=0, vmax=1)
    axes[0, 1].set_title("background mask")
    axes[0, 1].imshow(masks[0], cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title("background image, masked")
    axes[0, 2].imshow(background_masked_image, vmin=0, vmax=1)

    axes[1, 0].set_title("made-up image")
    imshow_madeup_if_has(axes[1, 0], made_up_image, has_made_up, vmin=0, vmax=1)
    axes[1, 1].set_title("made-up mask")
    imshow_madeup_if_has(axes[1, 1], masks[1], has_made_up, vmin=0, vmax=1, cmap='gray')
    axes[1, 2].set_title("masked made up image")
    imshow_madeup_if_has(axes[1, 2], made_up_masked_image, has_made_up, vmin=0, vmax=1)

    axes[2, 0].set_title("background pix distrib")
    axes[2, 0].imshow(background_pix_distrib, vmin=0, vmax=1)
    axes[2, 1].set_title("background mask")
    axes[2, 1].imshow(masks[0], cmap='gray', vmin=0, vmax=1)
    axes[2, 2].set_title("background pix distrib masked")
    axes[2, 2].imshow(background_pix_distrib_masked, vmin=0, vmax=1)

    axes[3, 0].set_title("made-up pix_distrib")
    imshow_madeup_if_has(axes[3, 0], made_up_pix_distrib, has_made_up, vmin=0, vmax=1)
    axes[3, 1].set_title("made-up mask")
    imshow_madeup_if_has(axes[3, 1], masks[1], has_made_up, vmin=0, vmax=1, cmap='gray')
    axes[3, 2].set_title("masked made up pix distrib")
    imshow_madeup_if_has(axes[3, 2], made_up_pix_distrib_masked, has_made_up, vmin=0, vmax=1)


def prev_vs_made_up_viz(prev_image: np.ndarray,
                        made_up_masked_image: np.ndarray,
                        ):
    fig, axes = plt.subplots(nrows=1, ncols=3, gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
    axes[0].set_title("previous image image")
    # TODO: clip or normalize?
    axes[0].imshow(np.clip(prev_image, 0, 1), vmin=0, vmax=1)
    axes[1].set_title("made-up masked image")
    axes[1].imshow(made_up_masked_image, vmin=0, vmax=1)
    axes[2].set_title("combined")
    combined = prev_image + made_up_masked_image
    norm_combined = combined / np.max(combined)
    axes[2].imshow(norm_combined, vmin=0, vmax=1)


def combination_viz(background_masked_image: np.ndarray,
                    background_pix_distrib_masked: np.ndarray,
                    prev_image: np.ndarray,
                    prev_pix_distrib: np.ndarray,
                    gen_images: np.ndarray,
                    gen_pix_distribs: np.ndarray,
                    masked_images: np.ndarray,
                    pix_distribs_masked: np.ndarray,
                    made_up_pix_distrib_masked: np.ndarray,
                    background_image: np.ndarray,
                    background_pix_distrib: np.ndarray,
                    n_kernels: int,
                    has_made_up: bool,
                    extra_masks: int,
                    ):
    fig, axes = plt.subplots(nrows=3, ncols=n_kernels + 5, gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    axes[0, 0].set_title("background\nimage")
    axes[0, 0].imshow(background_image, vmin=0, vmax=1)
    axes[1, 0].set_title("background pix\ndistrib")
    axes[1, 0].imshow(background_pix_distrib, vmin=0, vmax=1)
    axes[2, 0].set_title("background pix\ndistrib scaled")
    axes[2, 0].imshow(background_pix_distrib, cmap='RdYlBu_r')

    axes[0, 1].set_title("previous\nimage")
    axes[0, 1].imshow(np.clip(prev_image, 0, 1), vmin=0, vmax=1)
    axes[1, 1].set_title("previous pix\ndistrib")
    axes[1, 1].imshow(prev_pix_distrib, vmin=0, vmax=1)
    axes[2, 1].set_title("previous pix\ndistrib scaled")
    axes[2, 1].imshow(prev_pix_distrib, cmap='RdYlBu_r')

    for i in range(n_kernels):
        axes[0, i + 2].set_title("masked #{}".format(i))
        axes[0, i + 2].imshow(np.clip(masked_images[i + extra_masks], 0, 1), vmin=0, vmax=1)
        axes[1, i + 2].set_title("masked #{}".format(i))
        axes[1, i + 2].imshow(np.clip(pix_distribs_masked[i + extra_masks], 0, 1), vmin=0, vmax=1)
        axes[2, i + 2].set_title("masked #{}\nscaled".format(i))
        axes[2, i + 2].imshow(np.clip(pix_distribs_masked[i + extra_masks], 0, 1), cmap='RdYlBu_r')

    axes[0, -3].set_title('background\nmasked')
    axes[0, -3].imshow(background_masked_image, vmin=0, vmax=1)
    axes[1, -3].set_title('background\nmasked')
    axes[1, -3].imshow(background_pix_distrib_masked, vmin=0, vmax=1)
    axes[2, -3].set_title('background\nmasked scaled')
    axes[2, -3].imshow(background_pix_distrib_masked, cmap='RdYlBu_r')

    axes[0, -2].set_title('made-up\nmasked')
    imshow_madeup_if_has(axes[0, -2], masked_images[1], has_made_up, vmin=0, vmax=1)
    axes[1, -2].set_title('made-up\nmasked')
    imshow_madeup_if_has(axes[1, -2], made_up_pix_distrib_masked, has_made_up, vmin=0, vmax=1)
    axes[2, -2].set_title('made-up\nmasked scaled')
    imshow_madeup_if_has(axes[2, -2], made_up_pix_distrib_masked, has_made_up, cmap='rainbow')

    axes[0, -1].set_title("combined image")
    axes[0, -1].imshow(np.clip(gen_images, 0, 1), vmin=0, vmax=1)
    axes[1, -1].set_title("combined pix\ndistrib")
    axes[1, -1].imshow(np.clip(gen_pix_distribs, 0, 1), vmin=0, vmax=1)
    axes[2, -1].set_title("combined pix\ndistrib scaled")
    axes[2, -1].imshow(np.clip(gen_pix_distribs, 0, 1), cmap='RdYlBu_r')


def cdna_pix_distrib_viz(kernels: np.ndarray,
                         pix_distribs_masked: np.ndarray,
                         masks: np.ndarray,
                         transformed_pix_distribs: np.ndarray,
                         prev_pix_distrib: np.ndarray,
                         n_kernels: int,
                         extra_masks: int,
                         ):
    fig, axes = plt.subplots(nrows=4, ncols=n_kernels + 1, gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
    for i in range(n_kernels):
        kernel = kernels[i]
        axes[0, i].set_title("kernel #{}".format(i))
        # Don't set vmin/vmax here, we let matplotlib normalize these here since they only show relative "motion"
        axes[0, i].imshow(kernel, cmap='Blues', vmin=0, vmax=1)
    transformed_pix_distrib_handles = []
    step_text_handles = []
    for i in range(n_kernels):
        axes[1, i].set_title("transformed #{}".format(i))
        transformed_pix_distrib_handle = axes[1, i].imshow(prev_pix_distrib, vmin=0, vmax=1)
        transformed_pix_distrib_handles.append(transformed_pix_distrib_handle)
        step_text_handle = axes[1, i].text(5, 8, 't=0', fontdict={'color': 'white', 'size': 5},
                                           bbox=dict(facecolor='black', alpha=0.5))
        step_text_handles.append(step_text_handle)

    def transformed_pix_distrib_update(j):
        for i in range(n_kernels):
            # Note: there's no "made-up" image or background image in transformed_pix_distribs
            step_text_handles[i].set_text("t={}".format(j))
            if j == 0:
                transformed_pix_distrib_handles[i].set_data(prev_pix_distrib)
            else:
                transformed_pix_distrib_handles[i].set_data(np.clip(transformed_pix_distribs[i], 0, 1))

    transformed_pix_distrib_anim = FuncAnimation(fig, transformed_pix_distrib_update, frames=2, interval=1000, repeat=True)
    for i in range(n_kernels):
        mask = masks[i + extra_masks]
        axes[2, i].set_title("mask #{}".format(i))
        axes[2, i].imshow(mask, cmap='gray', vmin=0, vmax=1)
    for i in range(n_kernels):
        output = pix_distribs_masked[i + extra_masks]
        axes[3, i].set_title("masked #{}".format(i))
        axes[3, i].imshow(output, vmin=0, vmax=1)
    return transformed_pix_distrib_anim


def cdna_image_viz(kernels: np.ndarray,
                   masked_images: np.ndarray,
                   masks: np.ndarray,
                   prev_image: np.ndarray,
                   images_transformed: np.ndarray,
                   n_kernels: int,
                   extra_masks: int,
                   ):
    fig, axes = plt.subplots(nrows=4, ncols=n_kernels + 1, gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
    for i in range(n_kernels):
        kernel = kernels[i]
        axes[0, i].set_title("kernel #{}".format(i))
        axes[0, i].imshow(kernel, cmap='Blues', vmin=0, vmax=1)
    transformed_image_handles = []
    step_text_handles = []
    for i in range(n_kernels):
        axes[1, i].set_title("transformed #{}".format(i))
        transformed_image_handle = axes[1, i].imshow(np.clip(prev_image, 0, 1), vmin=0, vmax=1)
        transformed_image_handles.append(transformed_image_handle)
        step_text_handle = axes[1, i].text(5, 8, 't=0', fontdict={'color': 'white', 'size': 5},
                                           bbox=dict(facecolor='black', alpha=0.5))
        step_text_handles.append(step_text_handle)

    def transformed_image_update(j):
        for i in range(n_kernels):
            # the background image is not transformed, so just skip the first image which is the transformed "made-up" image
            step_text_handles[i].set_text("t={}".format(j))
            if j == 0:
                transformed_image_handles[i].set_data(np.clip(prev_image, 0, 1))
            else:
                transformed_image_handles[i].set_data(np.clip(images_transformed[i + (extra_masks - 1)], 0, 1))

    transformed_image_anim = FuncAnimation(fig, transformed_image_update, frames=2, interval=1000, repeat=True)
    for i in range(n_kernels):
        mask = masks[i + extra_masks]
        axes[2, i].set_title("mask #{}".format(i))
        axes[2, i].imshow(mask, cmap='gray', vmin=0, vmax=1)
    for i in range(n_kernels):
        output = masked_images[i + extra_masks]
        axes[3, i].set_title("masked #{}".format(i))
        # TODO: clip or normalize?
        axes[3, i].imshow(np.clip(output, 0, 1), vmin=0, vmax=1)
    return transformed_image_anim


def combined_masks_viz(prev_image: np.ndarray,
                       masks: np.ndarray,
                       indeces: List[int],
                       has_made_up: bool):
    fig, axes = plt.subplots(nrows=3, ncols=int(masks.shape[0] / 2), gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    axes = axes.flatten()
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    # first image is background
    background_mask = masks[0]

    if has_made_up:
        first_mask = 2
    else:
        first_mask = 1

    for i, mask in enumerate(masks[first_mask:]):
        if i in indeces:
            axes[i].plot([0, 0, 63, 63, 0], [0, 63, 63, 0, 0], c='green', linewidth=4)
        axes[i].set_title("mask #{}".format(i))
        axes[i].imshow(mask, cmap='gray', vmin=0, vmax=1)

    axes[i + 1].set_title("supposed background mask")
    axes[i + 1].imshow(background_mask, cmap='gray', vmin=0, vmax=1)

    indeces = np.array(indeces)
    shifted_indeces = indeces + first_mask
    masks_to_combine = masks[shifted_indeces]
    combined_masks = np.sum(masks_to_combine, axis=0)
    combined_masks_3d = np.atleast_3d(combined_masks)
    axes[i + 2].set_title("combined masks {}".format(indeces))
    axes[i + 2].imshow(combined_masks, cmap='gray', vmin=0, vmax=1)

    axes[i + 3].imshow(np.clip(combined_masks_3d * prev_image, 0, 1), vmin=0, vmax=1)
    axes[i + 3].set_title("masks [image]")

    other_indeces = np.setdiff1d(np.arange(1, masks.shape[0]), shifted_indeces)
    other_masks_to_combine = masks[other_indeces]
    combined_other_masks = np.sum(other_masks_to_combine, axis=0)
    combined_other_masks_3d = np.atleast_3d(combined_other_masks)
    axes[i + 4].set_title("other masks combined")
    axes[i + 4].imshow(combined_other_masks, cmap='gray', vmin=0, vmax=1)

    axes[i + 5].imshow(np.clip(combined_other_masks_3d * prev_image, 0, 1), vmin=0, vmax=1)
    axes[i + 5].set_title("other masks [image]")


def setup_and_run_from_example_folder(example_dir_path, context_length, checkpoint, model, model_hparams):
    images_file = [example_dir_path / '0.png', example_dir_path / '1.png']
    states_file = example_dir_path / 'states.csv'
    context_actions_file = example_dir_path / 'context_actions.csv'
    actions_file = example_dir_path / 'actions.csv'
    context_states, context_images, context_actions, actions = load_data(images_file, states_file, context_actions_file,
                                                                         actions_file)
    return setup_and_run(context_states, context_images, context_actions, actions, context_length, checkpoint, model,
                         model_hparams)


def setup_and_run_from_individual_files(images, states, context_actions, actions, context_length, checkpoint, model,
                                        model_hparams):
    context_states, context_images, context_actions, actions = load_data(images, states, context_actions, actions)
    return setup_and_run(context_states, context_images, context_actions, actions, context_length, checkpoint, model,
                         model_hparams)


def setup_and_run_from_gazebo(actions_filename, context_length, checkpoint, model, model_hparams):
    state_dim = 6
    sna_model_action_dim = 2
    services = gazebo_utils.GazeboServices()
    context_images, context_states, context_actions = services.get_context(context_length, state_dim, sna_model_action_dim)

    actions = np.atleast_2d(np.genfromtxt(actions_filename, delimiter=',', dtype=np.float32))

    return setup_and_run(context_states, context_images, context_actions, actions, context_length, checkpoint, model,
                         model_hparams)


def setup_and_run(context_states, context_images, context_actions, actions, context_length, checkpoint, model, model_hparams):
    actions_length, action_dim = actions.shape
    _, image_h, image_w, image_d = context_images.shape
    _, state_dim = context_states.shape

    placeholders, sequence_length = build_placeholders(context_length, actions_length, image_h, image_w, image_d, state_dim,
                                                       action_dim)

    model = build_model(checkpoint, model, model_hparams, placeholders, context_length, sequence_length)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.graph.as_default()
    model.restore(sess, checkpoint)

    source_pixel0 = gui_tools.get_source_pixel(context_images[0])
    source_pixel1 = gui_tools.get_source_pixel(context_images[1])

    # FIXME: debugging!
    context_pix_distribs = make_context_pix_distribs(context_length, image_h, image_w, [source_pixel0, source_pixel1])

    feed_dict = build_feed_dict(placeholders=placeholders,
                                context_images=context_images,
                                context_states=context_states,
                                context_pix_distribs=context_pix_distribs,
                                context_actions=context_actions,
                                actions=actions,
                                sequence_length=sequence_length)

    fetches = {
        'input_images': model.inputs['images'],
        'input_pix_distribs': model.inputs['pix_distribs'],

        'cdna_kernels': model.outputs['gen_cdna_kernels'],
        'masks': model.outputs['gen_masks'],

        'background_images': model.outputs['gen_background_images'],
        'transformed_images': model.outputs['gen_transformed_images'],
        'masked_images': model.outputs['gen_masked_images'],
        'fused_images': model.outputs['gen_images'],

        'background_pix_distribs': model.outputs['gen_background_pix_distribs'],
        'transformed_pix_distribs': model.outputs['gen_transformed_pix_distribs'],
        'masked_pix_distribs': model.outputs['gen_masked_pix_distribs'],
        'fused_pix_distribs': model.outputs['gen_pix_distribs'],
    }

    if model.hparams.generate_scratch_image:
        fetches['made_up_images'] = model.outputs['gen_made_up_images']
        fetches['made_up_pix_distribs'] = model.outputs['gen_made_up_pix_distribs']

    results = sess.run(fetches, feed_dict=feed_dict)
    return results, model.hparams.generate_scratch_image
