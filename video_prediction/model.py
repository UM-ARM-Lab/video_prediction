from __future__ import absolute_import, division, print_function

import errno
import json
import os
import warnings
from collections import OrderedDict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from video_prediction import models
from video_prediction.utils.ffmpeg_gif import save_gif


class VisualPredictionModel:

    def __init__(self, checkpoint: str,
                 context_length: int,
                 future_length: int,
                 state_dim: int = 2,
                 action_dim: int = 2,
                 image_dim: List[int] = None):
        if image_dim is None:
            image_dim = [64, 64, 3]
        self.image_dim = image_dim
        self.h, self.w, _ = self.image_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.checkpoint = checkpoint
        self.context_length = context_length
        self.prediction_length = future_length
        self.total_length = context_length + future_length
        self.placeholders = build_placeholders(self.total_length, state_dim, action_dim, image_dim)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.model = build_model(checkpoint, 'sna', None, context_length, self.placeholders, self.total_length)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.sess.graph.as_default()

        self.model.restore(self.sess, self.checkpoint)

    def rollout_pixel_distributions(self, context_pixel_images, context_states, actions):
        """
        Convert the data needed to make a prediction into a TensorFlow Dataset Iterator
        :param context_pixel_images: a numpy array [context_length, width, height, channels]
        :param context_states: a numpy array [context_length, state_dimension]
        :param actions: a numpy array [future_length, action_dimension]
        """
        padded_context_states = np.zeros([1, self.total_length, self.state_dim], np.float32)
        nop_images = np.zeros([1, self.total_length, *self.image_dim], np.float32)
        padded_context_pixel_images = np.zeros([1, self.total_length, self.h, self.w, 1], np.float32)
        padded_actions = np.zeros([1, self.total_length, self.action_dim], np.float32)
        padded_context_states[0, : self.context_length] = context_states
        padded_context_pixel_images[0, : self.context_length] = context_pixel_images
        padded_actions[0, self.context_length - 1: -1] = actions

        feed_dict = {
            self.placeholders['states']: padded_context_states,
            self.placeholders['images']: nop_images,
            self.placeholders['pix_distribs']: padded_context_pixel_images,
            self.placeholders['actions']: padded_actions,
        }
        fetches = OrderedDict({
            'gen_pix_distribs': self.model.outputs['gen_pix_distribs'],
            'gen_states': self.model.outputs['gen_states'],
        })
        results = self.sess.run(fetches, feed_dict=feed_dict)

        # 0 indexes the batch, which is always of size 1
        # time indexing here means that we include the last context image only
        gen_states = np.concatenate((context_states[[1]], results['gen_states'][0, self.context_length:]))
        gen_images = np.concatenate((context_pixel_images[[1]], results['gen_pix_distribs'][0, self.context_length:]))
        return gen_images, gen_states

    def rollout(self, context_images, context_states, actions):
        """
        Convert the data needed to make a prediction into a TensorFlow Dataset Iterator
        :param context_images: a numpy array [context_length, width, height, channels]
        :param context_states: a numpy array [context_length, state_dimension]
        :param actions: a numpy array [future_length, action_dimension]
        """
        padded_context_states = np.zeros([1, self.total_length, self.state_dim], np.float32)
        padded_context_images = np.zeros([1, self.total_length, *self.image_dim], np.float32)
        nop_pixel_images = np.zeros([1, self.total_length, self.h, self.w, 1], np.float32)
        padded_actions = np.zeros([1, self.total_length, self.action_dim], np.float32)
        padded_context_states[0, : self.context_length] = context_states
        padded_context_images[0, : self.context_length] = context_images
        padded_actions[0, self.context_length - 1: -1] = actions

        feed_dict = {
            self.placeholders['states']: padded_context_states,
            self.placeholders['images']: padded_context_images,
            self.placeholders['pix_distribs']: nop_pixel_images,
            self.placeholders['actions']: padded_actions,
        }
        fetches = OrderedDict({
            'gen_images': self.model.outputs['gen_images'],
            'gen_states': self.model.outputs['gen_states'],
        })
        results = self.sess.run(fetches, feed_dict=feed_dict)

        # 0 indexes the batch, which is always of size 1
        # time indexing here means that we include the last context image only
        gen_states = np.concatenate((context_states[[1]], results['gen_states'][0, self.context_length:]))
        gen_images = np.concatenate((context_images[[1]], results['gen_images'][0, self.context_length:]))
        return gen_images, gen_states


def build_placeholders(total_length, state_dim, action_dim, image_dim):
    # all arrays must be the length of the entire prediction, but only the first context_length will be used
    # we also don't fill the last action because it is not used.
    # Ths way the last action given is actually used to make the final prediction
    h, w, d = image_dim
    placeholders = {
        'states': tf.placeholder(tf.float32, [1, total_length, state_dim]),
        'images': tf.placeholder(tf.float32, [1, total_length, *image_dim]),
        'pix_distribs': tf.placeholder(tf.float32, [1, total_length, h, w, 1]),
        'actions': tf.placeholder(tf.float32, [1, total_length, action_dim]),
    }
    return placeholders


def build_model(checkpoint, model_str, model_hparams, context_length, input_placeholders, sequence_length):
    model_hparams_dict = {}
    checkpoint_dir = os.path.normpath(checkpoint)
    if not os.path.isdir(checkpoint):
        checkpoint_dir, _ = os.path.split(checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_dir)
    with open(os.path.join(checkpoint_dir, "options.json")) as f:
        print("loading options from checkpoint %s" % checkpoint)
        options = json.loads(f.read())
        model_str = model_str or options['model']
    try:
        with open(os.path.join(checkpoint_dir, "model_hparams.json")) as f:
            model_hparams_dict = json.loads(f.read())
    except FileNotFoundError:
        print("model_hparams.json was not loaded because it does not exist")

    VideoPredictionModel = models.get_model_class(model_str)
    hparams_dict = dict(model_hparams_dict)
    hparams_dict.update({
        'context_frames': context_length,
        'sequence_length': sequence_length,
    })
    model = VideoPredictionModel(
        mode='test',
        hparams_dict=hparams_dict,
        hparams=model_hparams)
    with tf.variable_scope(''):
        model.build_graph(input_placeholders)
    return model


def visualize(results, context_length, args):
    # first dimension is batch size, second is time
    context_images = (results['input_images'][0, :context_length] * 255.0).astype(np.uint8)
    context_states = results['input_states'][0, :context_length]
    gen_images = (results['gen_images'][0, context_length:] * 255.0).astype(np.uint8)
    gen_states = results['gen_states'][0, context_length:]

    # Plot trajectory in states space
    # FIXME: this assumes 2d state
    plt.scatter(context_states[:, 0], context_states[:, 1], label='context_states', c='b')
    plt.scatter(gen_states[:, 0], gen_states[:, 1], label='gen_states', c='r')
    states_path = np.concatenate((context_states, gen_states))
    plt.plot(states_path[:, 0], states_path[:, 1], c='r')
    plt.legend()
    plt.axis("equal")

    states_plot_filename = os.path.join(args.results_dir, "states_plot.png")
    plt.savefig(fname=states_plot_filename)

    # Save gif of input images
    input_images_filename = 'input_image.gif'
    save_gif(os.path.join(args.results_dir, input_images_filename), context_images, fps=args.fps)

    # Save gif of generated images
    gen_images_filename = 'gen_image.gif'
    context_and_gen_images = np.concatenate((context_images, gen_images), axis=0)
    save_gif(os.path.join(args.results_dir, gen_images_filename), context_and_gen_images, fps=args.fps)

    # Save individual frames of the prediction
    gen_image_filename_pattern = 'gen_image_%%05d_%%0%dd.png' % max(2, len(str(len(gen_images) - 1)))
    for time_step_idx, gen_image in enumerate(gen_images):
        gen_image_filename = gen_image_filename_pattern % (time_step_idx, time_step_idx)
        plt.imsave(os.path.join(args.results_dir, gen_image_filename), gen_image)
