from __future__ import absolute_import, division, print_function

import errno
import json
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from video_prediction import models
from video_prediction.utils.ffmpeg_gif import save_gif


class VisualPredictionModel:

    def __init__(self, checkpoint, future_length, context_length=2, state_dim=2, action_dim=2, image_dim=None):
        if image_dim is None:
            image_dim = [64, 64, 3]
        self.checkpoint = checkpoint
        self.context_length = context_length
        self.prediction_length = future_length
        self.total_length = context_length + future_length
        self.placeholders, self.inputs_op = build_iterators(context_length, future_length, state_dim, action_dim, image_dim)
        self.model = build_model(checkpoint, 'sna', None, context_length, self.placeholders, self.total_length)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.sess.graph.as_default()

        self.model.restore(self.sess, self.checkpoint)

    def rollout(self, context_states, context_images, actions):
        """
        Convert the data needed to make a prediction into a TensorFlow Dataset Iterator
        :param context_states: a numpy array [context_length, state_dimension]
        :param context_images: a numpy array [context_length, width, height, channels]
        :param actions: a numpy array [future_length, action_dimension]
        :return: dictionary for feeding, iterator, and total_length
        """
        # inputs_placeholders, inputs_op, total_length = build_dataset(context_states, context_images, actions)
        input_results = self.sess.run(self.inputs_op)

        feed_dict = {input_ph: input_results[name] for name, input_ph in self.placeholders.items()}
        feed_dict['states'][0, :self.context_length] = context_states
        feed_dict['images'][0, :self.context_length] = context_images
        feed_dict['actions'][0, self.context_length - 1:-1] = actions
        fetches = OrderedDict({
            'images': self.model.outputs['gen_images'],
            'states': self.model.outputs['gen_states'],
        })
        results = self.sess.run(fetches, feed_dict=feed_dict)

        gen_images = (results['images'][0, self.context_length:] * 255.0).astype(np.uint8)
        gen_states = results['states'][0, self.context_length:]


def build_placeholders(total_length, state_dim, action_dim, image_dim):
    # all arrays must be the length of the entire prediction, but only the first context_length will be used
    # we also don't fill the last action because it is not used.
    # Ths way the last action given is actually used to make the final prediction
    placeholders = {
        'states': tf.placeholder(tf.float32, [1, total_length, state_dim]),
        'images': tf.placeholder(tf.float32, [1, total_length, *image_dim]),
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
    # plt.plot(context_states, c='b')
    plt.scatter(gen_states[:, 0], gen_states[:, 1], label='gen_states', c='r')
    # plt.plot(gen_states, c='r')
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
