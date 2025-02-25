#!/usr/bin/env python
from __future__ import division, print_function

import glob
import os
import random
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from google.protobuf.json_format import MessageToDict
from tensorflow.contrib.training import HParams


def make_mask(T, S):
    P = T - S + 1
    mask = np.zeros((T, T))
    for i in range(S):
        mask += np.diag(np.ones(T - i), -i)
    return mask[:, :P]


class BaseVideoDataset(object):
    def __init__(self, input_dir, mode='train', num_epochs=None, seed=None,
                 hparams_dict=None, hparams=None):
        """
        Args:
            input_dir: either a directory containing subdirectories train,
                val, test, etc, or a directory containing the tfrecords.
            mode: either train, val, or test
            num_epochs: if None, dataset is iterated indefinitely.
            seed: random seed for the op that samples subsequences.
            hparams_dict: a dict of `name=value` pairs, where `name` must be
                defined in `self.get_default_hparams()`.
            hparams: a string of comma separated list of `name=value` pairs,
                where `name` must be defined in `self.get_default_hparams()`.
                These values overrides any values in hparams_dict (if any).

        Note:
            self.input_dir is the directory containing the tfrecords.
        """
        self.input_dir = os.path.normpath(os.path.expanduser(input_dir))
        self.mode = mode
        self.num_epochs = num_epochs
        self.seed = seed
        self._max_sequence_length = None

        if self.mode not in ('train', 'val', 'test'):
            raise ValueError('Invalid mode %s' % self.mode)

        if not os.path.exists(self.input_dir):
            raise FileNotFoundError("input_dir %s does not exist" % self.input_dir)
        self.filenames = None
        # look for tfrecords in input_dir and input_dir/mode directories
        for input_dir in [self.input_dir, os.path.join(self.input_dir, self.mode)]:
            filenames = glob.glob(os.path.join(input_dir, '*.tfrecord*'))
            if filenames:
                self.input_dir = input_dir
                self.filenames = sorted(filenames)  # ensures order is the same across systems
                break
        if not self.filenames:
            raise FileNotFoundError('No tfrecords were found in %s.' % self.input_dir)
        self.dataset_name = os.path.basename(os.path.split(self.input_dir)[0])

        self.state_like_names_and_shapes = OrderedDict()
        self.action_like_names_and_shapes = OrderedDict()
        self.trajectory_constant_names_and_shapes = OrderedDict()

        self.hparams = self.parse_hparams(hparams_dict, hparams)
        self.start_mask = None

    def get_default_hparams_dict(self):
        """
        Returns:
            A dict with the following hyperparameters.

            crop_size: crop image into a square with sides of this length.
            scale_size: resize image to this size after it has been cropped.
            context_frames: the number of ground-truth frames to pass in at
                start.
            sequence_length: the number of frames in the video sequence, so
                state-like sequences are of length sequence_length and
                action-like sequences are of length sequence_length - 1.
                This number includes the context frames.
            long_sequence_length: the number of frames for the long version.
                The default is the same as sequence_length.
            frame_skip: number of frames to skip in between outputted frames,
                so frame_skip=0 denotes no skipping.
            time_shift: shift in time by multiples of this, so time_shift=1
                denotes all possible shifts. time_shift=0 denotes no shifting.
                It is ignored (equiv. to time_shift=0) when mode != 'train'.
            force_time_shift: whether to do the shift in time regardless of
                mode.
            shuffle_on_val: whether to shuffle the samples regardless if mode
                is 'train' or 'val'. Shuffle never happens when mode is 'test'.
            use_state: whether to load and return state and actions.
            free_space_only: when True, only sequences which contain data where the 'constraint' feature is true will be selected
        """
        hparams = dict(
            crop_size=0,
            scale_size=0,
            context_frames=1,
            sequence_length=0,
            long_sequence_length=0,
            frame_skip=0,
            time_shift=1,
            force_time_shift=False,
            shuffle_on_val=False,
            use_state=False,
            free_space_only=False,
            compression_type='',
            dt=0.1,
            env_w=1.0,
            env_h=1.0,
        )
        return hparams

    def get_default_hparams(self):
        return HParams(**self.get_default_hparams_dict())

    def parse_hparams(self, hparams_dict, hparams):
        parsed_hparams = self.get_default_hparams().override_from_dict(hparams_dict or {})
        if hparams:
            if not isinstance(hparams, (list, tuple)):
                hparams = [hparams]
            for hparam in hparams:
                parsed_hparams.parse(hparam)
        if parsed_hparams.long_sequence_length == 0:
            parsed_hparams.long_sequence_length = parsed_hparams.sequence_length
        return parsed_hparams

    @property
    def jpeg_encoding(self):
        raise NotImplementedError

    def set_sequence_length(self, sequence_length):
        self.hparams.sequence_length = sequence_length

    def filter(self, serialized_example):
        return tf.convert_to_tensor(True)

    def parser(self, serialized_example):
        """
        Parses a single tf.train.Example or tf.train.SequenceExample into
        images, states, actions, etc tensors.
        """
        raise NotImplementedError

    def make_dataset(self, batch_size, use_batches=True, shuffle=True, num_parallel_calls=None):
        filenames = self.filenames
        if shuffle:
            shuffle = (self.mode == 'train') or (self.mode == 'val' and self.hparams.shuffle_on_val)
            if shuffle:
                random.shuffle(filenames)

        dataset = tf.data.TFRecordDataset(filenames, buffer_size=8 * 1024 * 1024, compression_type=self.hparams.compression_type)
        # filter to keep only examples which are as long as the requested sequence_length
        dataset = dataset.filter(self.filter)
        if shuffle:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1024, count=self.num_epochs))
        else:
            dataset = dataset.repeat(self.num_epochs)

        def _parser(serialized_example):
            state_like_seqs, action_like_seqs = self.parser(serialized_example)
            return state_like_seqs, action_like_seqs

        def has_valid_index(constraints_seq):
            valid_start_onehot = constraints_seq.squeeze() @ self.start_mask
            valid_start_indeces = np.argwhere(valid_start_onehot == 0).squeeze()
            # Handle the case where there is no such sequence
            return valid_start_indeces.size > 0

        def _filter_free_space_only(state_like_seqs, action_like_seqs):
            del action_like_seqs
            is_valid = tf.py_func(has_valid_index,
                                  [state_like_seqs['constraints']],
                                  tf.bool, name='has_valid_index')
            return is_valid

        def _flatten(state_like_sliced_seqs, action_like_sliced_seqs):
            flat_input_dict = state_like_sliced_seqs
            flat_input_dict.update(action_like_sliced_seqs)
            return flat_input_dict

        def _slice_sequences(state_like_seqs, action_like_seqs):
            example_sequence_length = self._max_sequence_length
            return self.slice_sequences(state_like_seqs, action_like_seqs, example_sequence_length)

        if self.hparams.free_space_only:
            if use_batches:
                dataset = dataset.map(
                    _parser, num_parallel_calls=num_parallel_calls).filter(
                    _filter_free_space_only).map(
                    _slice_sequences, num_parallel_calls=num_parallel_calls).map(
                    _flatten, num_parallel_calls=num_parallel_calls).batch(
                    batch_size, drop_remainder=True).prefetch(
                    batch_size)
            else:
                dataset = dataset.map(
                    _parser, num_parallel_calls=num_parallel_calls).filter(
                    _filter_free_space_only).map(
                    _slice_sequences, num_parallel_calls=num_parallel_calls).map(
                    _flatten, num_parallel_calls=num_parallel_calls)
        else:
            if use_batches:
                dataset = dataset.map(
                    _parser, num_parallel_calls=num_parallel_calls).map(
                    _slice_sequences, num_parallel_calls=num_parallel_calls).map(
                    _flatten, num_parallel_calls=num_parallel_calls).batch(
                    batch_size, drop_remainder=True).prefetch(
                    batch_size)
            else:
                dataset = dataset.map(
                    _parser, num_parallel_calls=num_parallel_calls).map(
                    _slice_sequences, num_parallel_calls=num_parallel_calls).map(
                    _flatten, num_parallel_calls=num_parallel_calls)
        return dataset

    def make_batch(self, batch_size, shuffle=True):
        dataset = self.make_dataset(batch_size, shuffle)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def decode_and_preprocess_images(self, image_buffers, image_shape):
        def decode_and_preprocess_image(image_buffer):
            image_buffer = tf.reshape(image_buffer, [])
            if self.jpeg_encoding:
                image = tf.image.decode_jpeg(image_buffer)
            else:
                image = tf.decode_raw(image_buffer, tf.uint8)
            image = tf.reshape(image, image_shape)
            crop_size = self.hparams.crop_size
            scale_size = self.hparams.scale_size
            if crop_size or scale_size:
                if not crop_size:
                    crop_size = min(image_shape[0], image_shape[1])
                image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
                image = tf.reshape(image, [crop_size, crop_size, 3])
                if scale_size:
                    # upsample with bilinear interpolation but downsample with area interpolation
                    if crop_size < scale_size:
                        image = tf.image.resize_images(image, [scale_size, scale_size],
                                                       method=tf.image.ResizeMethod.BILINEAR)
                    elif crop_size > scale_size:
                        image = tf.image.resize_images(image, [scale_size, scale_size],
                                                       method=tf.image.ResizeMethod.AREA)
                    else:
                        # image remains unchanged
                        pass
            return image

        if not isinstance(image_buffers, (list, tuple)):
            image_buffers = tf.unstack(image_buffers)
        images = [decode_and_preprocess_image(image_buffer) for image_buffer in image_buffers]
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        return images

    def convert_to_sequences(self, state_like_seqs, action_like_seqs, example_sequence_length):
        # Convert anything which is a list of tensors along time dimension to one tensor where the first dimension is time
        state_like_tensors = {}
        action_like_tensors = {}
        for example_name, seq in state_like_seqs.items():
            seq = tf.convert_to_tensor(seq)
            seq.set_shape([example_sequence_length] + seq.shape.as_list()[1:])
            state_like_tensors[example_name] = seq
        for example_name, seq in action_like_seqs.items():
            seq = tf.convert_to_tensor(seq)
            seq.set_shape([example_sequence_length - 1] + seq.shape.as_list()[1:])
            action_like_tensors[example_name] = seq

        return state_like_tensors, action_like_tensors

    def slice_sequences(self, state_like_seqs, action_like_seqs, example_sequence_length):
        """
        Slices sequences of length `example_sequence_length` into subsequences
        of length `sequence_length`. The dicts of sequences are updated
        in-place and the same dicts are returned.
        """
        sequence_length = self.hparams.sequence_length
        time_shift = self.hparams.time_shift
        frame_skip = self.hparams.frame_skip

        # FIXME: does not respect frame_skip or time_shift, assumes frame_skip=0 and time_shift=1
        def choose_random_valid_start_index(constraints_seq):
            valid_start_onehot = constraints_seq.squeeze() @ self.start_mask
            valid_start_indeces = np.argwhere(valid_start_onehot == 0)
            valid_start_indeces = np.atleast_1d(valid_start_indeces.squeeze())
            try:
                choice = np.random.choice(valid_start_indeces)
            except Exception:
                print('invalid!', valid_start_indeces)
            return choice

        if self.hparams.free_space_only:
            t_start = tf.py_func(choose_random_valid_start_index,
                                 [state_like_seqs['constraints']],
                                 tf.int64, name='choose_valid_start_t')
        else:
            if (time_shift and self.mode == 'train') or self.hparams.force_time_shift:
                assert time_shift > 0 and isinstance(time_shift, int)
                if isinstance(example_sequence_length, tf.Tensor):
                    example_sequence_length = tf.cast(example_sequence_length, tf.int32)
                num_shifts = ((example_sequence_length - 1) - (sequence_length - 1) * (frame_skip + 1)) // time_shift
                assert_message = ('example_sequence_length has to be at least %d when '
                                  'sequence_length=%d, frame_skip=%d.' %
                                  ((sequence_length - 1) * (frame_skip + 1) + 1,
                                   sequence_length, frame_skip))
                with tf.control_dependencies([tf.assert_greater_equal(num_shifts, 0,
                                                                      data=[example_sequence_length, num_shifts],
                                                                      message=assert_message)]):
                    t_start = tf.random_uniform([], 0, num_shifts + 1, dtype=tf.int64, seed=self.seed) * time_shift
            else:
                t_start = 0

        state_like_t_slice = slice(t_start, t_start + (sequence_length - 1) * (frame_skip + 1) + 1, frame_skip + 1)
        action_like_t_slice = slice(t_start, t_start + (sequence_length - 1) * (frame_skip + 1))

        state_like_sliced_seqs = OrderedDict()
        action_like_sliced_seqs = OrderedDict()
        for example_name, seq in state_like_seqs.items():
            sliced_seq = seq[state_like_t_slice]
            sliced_seq.set_shape([sequence_length] + seq.shape.as_list()[1:])
            state_like_sliced_seqs[example_name] = sliced_seq
        for example_name, seq in action_like_seqs.items():
            sliced_seq = seq[action_like_t_slice]
            sliced_seq.set_shape([(sequence_length - 1) * (frame_skip + 1)] + seq.shape.as_list()[1:])
            action_like_sliced_seqs[example_name] = sliced_seq

        return state_like_sliced_seqs, action_like_sliced_seqs

    def num_examples_per_epoch(self):
        raise NotImplementedError


class VideoDataset(BaseVideoDataset):
    """
    This class supports reading tfrecords where a sequence is stored as
    multiple tf.train.Example and each of them is stored under a different
    feature name (which is indexed by the time step).
    """

    def __init__(self, *args, **kwargs):
        super(VideoDataset, self).__init__(*args, **kwargs)
        self._dict_message = None

    def _infer_seq_length_and_setup(self):
        """
        Should be called after state_like_names_and_shapes and
        action_like_names_and_shapes have been finalized.
        """
        options = tf.python_io.TFRecordOptions(compression_type=self.hparams.compression_type)
        example = next(tf.python_io.tf_record_iterator(self.filenames[0], options=options))
        dict_message = MessageToDict(tf.train.Example.FromString(example))
        feature = dict_message['features']['feature']
        self._max_sequence_length = 0
        for feature_name in feature.keys():
            try:
                # plus 1 because time is 0 indexed here
                time_str = feature_name.split("/")[0]
                self._max_sequence_length = max(self._max_sequence_length, int(time_str) + 1)
            except ValueError:
                pass

        # set sequence_length to the longest possible if it is not specified
        if not self.hparams.sequence_length:
            self.hparams.sequence_length = (self._max_sequence_length - 1) // (self.hparams.frame_skip + 1) + 1

        self.start_mask = make_mask(self._max_sequence_length, self.hparams.sequence_length)

    def set_sequence_length(self, sequence_length):
        if not sequence_length:
            sequence_length = (self._max_sequence_length - 1) // (self.hparams.frame_skip + 1) + 1
        self.hparams.sequence_length = sequence_length

    def parser(self, serialized_example):
        """
        Parses a single tf.train.Example into images, states, actions, etc tensors.
        """
        features = dict()
        for example_name, (name, shape) in self.trajectory_constant_names_and_shapes.items():
            features[name] = tf.FixedLenFeature(shape, tf.float32)
        for example_name, (name, shape) in self.trajectory_constant_names_and_shapes.items():
            features[name] = tf.FixedLenFeature(shape, tf.float32)
        for i in range(self._max_sequence_length):
            for example_name, (name, shape) in self.state_like_names_and_shapes.items():
                # FIXME: support loading of int64 features
                if example_name == 'images':  # special handling for image
                    features[name % i] = tf.FixedLenFeature([1], tf.string)
                else:
                    features[name % i] = tf.FixedLenFeature(shape, tf.float32)
        for i in range(self._max_sequence_length - 1):
            for example_name, (name, shape) in self.action_like_names_and_shapes.items():
                features[name % i] = tf.FixedLenFeature(shape, tf.float32)

        # parse all the features of all time steps together
        features = tf.parse_single_example(serialized_example, features=features)

        state_like_seqs = OrderedDict([(example_name, []) for example_name in self.state_like_names_and_shapes])
        action_like_seqs = OrderedDict([(example_name, []) for example_name in self.action_like_names_and_shapes])
        for i in range(self._max_sequence_length):
            for example_name, (name, shape) in self.state_like_names_and_shapes.items():
                state_like_seqs[example_name].append(features[name % i])
            for example_name, (name, shape) in self.trajectory_constant_names_and_shapes.items():
                if example_name not in state_like_seqs:
                    state_like_seqs[example_name] = []
                state_like_seqs[example_name].append(features[name])
        for i in range(self._max_sequence_length - 1):
            for example_name, (name, shape) in self.action_like_names_and_shapes.items():
                action_like_seqs[example_name].append(features[name % i])

        # for this class, it's much faster to decode and preprocess the entire sequence before sampling a slice
        _, image_shape = self.state_like_names_and_shapes['images']
        state_like_seqs['images'] = self.decode_and_preprocess_images(state_like_seqs['images'], image_shape)

        state_like_seqs, action_like_seqs = self.convert_to_sequences(state_like_seqs, action_like_seqs,
                                                                      self._max_sequence_length)
        return state_like_seqs, action_like_seqs
