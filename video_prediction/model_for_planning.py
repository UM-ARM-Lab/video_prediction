from __future__ import absolute_import, division, print_function

import errno
import json
import os
from collections import OrderedDict
from typing import List

import numpy as np
import tensorflow as tf

from video_prediction import models


class ModelForPlanning:

    def __init__(self, checkpoint: str,
                 context_length: int,
                 actions_length: int,
                 state_dim: int = 2,
                 action_dim: int = 2,
                 image_dim: List[int] = None):
        self.context_length = context_length
        if image_dim is None:
            image_dim = [64, 64, 3]
        h, w, d = image_dim

        self.placeholders, self.sequence_length = build_placeholders(context_length,
                                                                     actions_length,
                                                                     h, w, d,
                                                                     state_dim,
                                                                     action_dim)
        self.model = build_model(checkpoint, 'sna', None, self.placeholders, context_length, self.sequence_length)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.sess.graph.as_default()

        self.model.restore(self.sess, checkpoint)

    def rollout_all(self, context_images, context_pixel_images, context_states, actions):
        feed_dict = build_feed_dict(self.placeholders,
                                    context_images,
                                    context_states,
                                    context_pixel_images,
                                    actions,
                                    self.sequence_length)

        fetches = OrderedDict({
            'gen_images': self.model.outputs['gen_images'],
            'gen_pix_distribs': self.model.outputs['gen_pix_distribs'],
            'gen_states': self.model.outputs['gen_states'],
        })

        results = self.sess.run(fetches, feed_dict=feed_dict)

        # 0 indexes the batch, which is always of size 1
        # time indexing here means that we include the last context image only
        gen_states = np.concatenate((context_states[[1]], results['gen_states'][0, self.context_length:]))
        gen_pix_distribs = np.concatenate((context_pixel_images[[1]], results['gen_pix_distribs'][0, self.context_length:]))
        gen_images = np.concatenate((context_images[[1]], results['gen_images'][0, self.context_length:]))
        return gen_pix_distribs, gen_images, gen_states


def build_placeholders(context_length, actions_length, h, w, d, state_dim, action_dim):
    # NOTE: We don't add context_length because that is not how action are fed in. The first action is supposed to be the action
    # that transitions from the first context image to the second context image. Therefore, regardless of the context length,
    # the total number of predicted images will always be 1+ the number of actions. This of course means there's two ways
    # to construct an output sequence:
    # 1) take the first context image and the rest of the generate images
    # 2) all the context images and only the images generate after the context images (i.e. warm start done)

    sequence_length = actions_length + 1
    placeholders = {
        'states': tf.placeholder(tf.float32, [1, sequence_length, state_dim]),
        'images': tf.placeholder(tf.float32, [1, sequence_length, h, w, d]),
        'pix_distribs': tf.placeholder(tf.float32, [1, context_length, h, w, 1]),
        'actions': tf.placeholder(tf.float32, [1, actions_length, action_dim]),
    }

    return placeholders, sequence_length


def build_feed_dict(placeholders, context_images, context_states, context_pix_distribs, actions, sequence_length):
    _, action_dim = actions.shape
    _, state_dim = context_states.shape
    _, h, w, d = context_images.shape
    # FIXME: this is gross
    context_length = context_pix_distribs.shape[1]
    padded_context_states = np.zeros([1, sequence_length, state_dim], dtype=np.float32)
    padded_context_images = np.zeros([1, sequence_length, h, w, d], dtype=np.float32)
    batched_context_pix_distribs = context_pix_distribs
    batched_actions = np.expand_dims(actions, axis=0)

    padded_context_images[0, :context_length] = context_images
    padded_context_states[0, :context_length] = context_states

    feed_dict = {
        placeholders['states']: padded_context_states,
        placeholders['images']: padded_context_images,
        placeholders['pix_distribs']: batched_context_pix_distribs,
        placeholders['actions']: batched_actions,
    }

    return feed_dict


def build_model(checkpoint, model_str, model_hparams, input_placeholders, context_length, sequence_length):
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
