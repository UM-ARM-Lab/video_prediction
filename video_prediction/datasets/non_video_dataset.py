import glob
import os
import random
import re
from collections import OrderedDict

import tensorflow as tf
from google.protobuf.json_format import MessageToDict
from tensorflow.contrib.training import HParams


class NonVideoDataset(object):
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

    def _check_or_infer_shapes(self):
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

    def get_default_hparams_dict(self):
        """
        Returns:
            A dict with the following hyperparameters.

            compression_type: empty string, GZIP, or ZLIB
        """
        hparams = dict(
            compression_type=None,
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
        return parsed_hparams

    def make_dataset(self, batch_size, use_batches=True, shuffle=True, num_parallel_calls=None):
        filenames = self.filenames
        if shuffle:
            random.shuffle(filenames)

        dataset = tf.data.TFRecordDataset(filenames, buffer_size=8 * 1024 * 1024, compression_type=self.hparams.compression_type)
        if shuffle:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1024, count=self.num_epochs))
        else:
            dataset = dataset.repeat(self.num_epochs)

        def _parser(serialized_example):
            state_like_seqs, action_like_seqs = self.parser(serialized_example)
            return state_like_seqs, action_like_seqs

        def _merge(state_like_sliced_seqs, action_like_sliced_seqs):
            merge_input_dict = state_like_sliced_seqs
            merge_input_dict.update(action_like_sliced_seqs)
            return merge_input_dict

        def _sample_one_time_step(sequence):
            # -1 because we can't get the action at that last time step
            idx = tf.random.uniform([], 0, self._max_sequence_length - 1, dtype=tf.int64, seed=1)
            step = {}
            with tf.name_scope("sample_one_time_step"):
                for name, tensor in sequence.items():
                    step[name] = tensor[idx]
            return step

        parsed_dataset = dataset.map(_parser, num_parallel_calls=num_parallel_calls)
        merged_dataset = parsed_dataset.map(_merge, num_parallel_calls=num_parallel_calls)
        one_time_step_dataset = merged_dataset.map(_sample_one_time_step, num_parallel_calls=num_parallel_calls)
        if use_batches:
            batched_dataset = one_time_step_dataset.batch(batch_size, drop_remainder=True)
            dataset = batched_dataset.prefetch(batch_size)
        else:
            dataset = one_time_step_dataset.prefetch(batch_size)
        return dataset

    def make_batch(self, batch_size, shuffle=True):
        dataset = self.make_dataset(batch_size, shuffle)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def decode_and_preprocess_images(self, image_buffers, image_shape):
        def decode_and_preprocess_image(image_buffer):
            image_buffer = tf.reshape(image_buffer, [])
            image = tf.decode_raw(image_buffer, tf.uint8)
            image = tf.reshape(image, image_shape)
            return image

        if not isinstance(image_buffers, (list, tuple)):
            image_buffers = tf.unstack(image_buffers)
        images = [decode_and_preprocess_image(image_buffer) for image_buffer in image_buffers]
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        return images

    def convert_to_sequences(self, state_like_seqs, action_like_seqs, example_sequence_length):
        """
        Convert anything which is a list of tensors along time dimension to one tensor where the first dimension is time
        """
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

    def parser(self, serialized_example):
        """
        Parses a single tf.train.Example into images, states, actions, etc tensors.
        """
        features = dict()
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

    def num_examples_per_epoch(self):
        # extract information from filename to count the number of trajectories in the dataset
        count = 0
        for filename in self.filenames:
            match = re.search('traj_(\d+)_to_(\d+).tfrecords', os.path.basename(filename))
            start_traj_iter = int(match.group(1))
            end_traj_iter = int(match.group(2))
            count += end_traj_iter - start_traj_iter + 1

        return count
