#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pyplot as plt
import errno
from collections import OrderedDict

from PIL import Image
import json
import os
import random

import cv2
import numpy as np
import tensorflow as tf

from video_prediction import models
from video_prediction.utils.ffmpeg_gif import save_gif


def main():
    np.set_printoptions(suppress=True, linewidth=250, precision=4, threshold=64 * 64 * 3)

    context_frames = 2

    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs=context_frames, help='filename')
    parser.add_argument("states", nargs=context_frames, help='should look like "s1,s2"')
    parser.add_argument("actions", help='should look like "vx1,vy1,vx2,vy2,..."')
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

    model_hparams_dict = {}
    checkpoint_dir = os.path.normpath(args.checkpoint)
    if not os.path.isdir(args.checkpoint):
        checkpoint_dir, _ = os.path.split(checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_dir)
    with open(os.path.join(checkpoint_dir, "options.json")) as f:
        print("loading options from checkpoint %s" % args.checkpoint)
        options = json.loads(f.read())
        args.model = args.model or options['model']
    try:
        with open(os.path.join(checkpoint_dir, "model_hparams.json")) as f:
            model_hparams_dict = json.loads(f.read())
    except FileNotFoundError:
        print("model_hparams.json was not loaded because it does not exist")

    # all arrays must be the length of the entire prediction, but only the first two will be used
    actions = np.fromstring(args.actions, dtype=np.float32, sep=',').reshape([1, -1, 2])
    sequence_length = actions.shape[1]
    states = np.zeros([1, sequence_length, 2], np.float32)
    image_bytes = np.zeros([1, sequence_length, 64, 64, 3], np.float32)
    for time_step_idx in range(context_frames):
        states[0, time_step_idx] = np.fromstring(args.states[time_step_idx], dtype=np.float32, sep=',')
        rgba_image_uint8 = np.array(Image.open(args.images[time_step_idx]), dtype=np.uint8)
        rgb_image_float = rgba_image_uint8[:, :, :3].astype(np.float32) / 255.0
        image_bytes[0, time_step_idx] = rgb_image_float

    VideoPredictionModel = models.get_model_class(args.model)
    hparams_dict = dict(model_hparams_dict)
    hparams_dict.update({
        'context_frames': context_frames,
        'sequence_length': sequence_length,
    })
    model = VideoPredictionModel(
        mode='test',
        hparams_dict=hparams_dict,
        hparams=args.model_hparams)

    sequence_length = model.hparams.sequence_length
    context_frames = model.hparams.context_frames
    future_length = sequence_length - context_frames

    dataset = tf.data.Dataset.from_tensor_slices((OrderedDict({
        'images': image_bytes,
        'states': states,
        'actions': actions,
    })))
    batched_dataset = dataset.batch(1, drop_remainder=True)

    iterator = batched_dataset.make_one_shot_iterator()
    inputs = iterator.get_next()
    input_phs = {k: tf.placeholder(v.dtype, v.shape, '%s_ph' % k) for k, v in inputs.items()}

    with tf.variable_scope(''):
        model.build_graph(input_phs)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.graph.as_default()

    model.restore(sess, args.checkpoint)

    input_results = sess.run(inputs)

    feed_dict = {input_ph: input_results[name] for name, input_ph in input_phs.items()}
    fetches = OrderedDict({
        'images': model.outputs['gen_images'],
        'states': model.outputs['gen_states'],
    })
    results = sess.run(fetches, feed_dict=feed_dict)
    gen_images = results['images']
    gen_states = results['states']
    print(states)
    print(gen_states)

    plt.scatter(states[0, :context_frames, 0], states[0, :context_frames, 1], label='input_states')
    # plt.scatter(gen_states[:], label='gen_states')
    plt.axis("equal")

    # only keep the future frames
    gen_images = gen_images[0, -future_length:]
    context_images_ = (input_results['images'][0] * 255.0).astype(np.uint8)
    gen_images = (gen_images * 255.0).astype(np.uint8)

    # Save gif of input images
    input_images_fname = 'input_image.gif'
    save_gif(os.path.join(args.results_dir, input_images_fname), context_images_, fps=args.fps)

    # Save gif of generated images
    gen_images_fname = 'gen_image.gif'
    context_and_gen_images = list(context_images_[:context_frames]) + list(gen_images)
    save_gif(os.path.join(args.results_dir, gen_images_fname),
             context_and_gen_images, fps=args.fps)

    # Save gif of errors over time
    error_images_fname = 'error_image.gif'
    error_images = context_images_ - context_and_gen_images
    save_gif(os.path.join(args.results_dir, error_images_fname), error_images, fps=args.fps)

    # Save individual images
    gen_image_fname_pattern = 'gen_image_%%05d_%%0%dd.png' % max(2, len(str(len(gen_images) - 1)))
    for time_step_idx, gen_image in enumerate(gen_images):
        gen_image_fname = gen_image_fname_pattern % (time_step_idx, time_step_idx)
        if gen_image.shape[-1] == 1:
            gen_image = np.tile(gen_image, (1, 1, 3))
        else:
            gen_image = cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.results_dir, gen_image_fname), gen_image)

    states_plot_filename = os.path.join(args.results_dir, "states_plot.png")
    plt.savefig(fname=states_plot_filename)
    plt.show()

if __name__ == '__main__':
    main()
