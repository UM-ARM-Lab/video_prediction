import json

import tensorflow as tf

# noinspection PyUnresolvedReferences
from .base_dataset import BaseVideoDataset
# noinspection PyUnresolvedReferences
from .base_dataset import VideoDataset
# noinspection PyUnresolvedReferences
from .google_robot_dataset import GoogleRobotVideoDataset
# noinspection PyUnresolvedReferences
from .link_bot_dataset import LinkBotDataset
# noinspection PyUnresolvedReferences
from .link_bot_video_dataset import LinkBotVideoDataset
# noinspection PyUnresolvedReferences
from .moving_block_dataset import MovingBlockDataset
# noinspection PyUnresolvedReferences
from .softmotion_dataset import SoftmotionVideoDataset
# noinspection PyUnresolvedReferences
from .sv2p_dataset import SV2PVideoDataset
# noinspection PyUnresolvedReferences
from .unity_cloth_dataset import UnityClothDataset


def get_dataset_class(dataset):
    dataset_mappings = {
        'google_robot': 'GoogleRobotVideoDataset',
        'sv2p': 'SV2PVideoDataset',
        'softmotion': 'SoftmotionVideoDataset',
        'bair': 'SoftmotionVideoDataset',  # alias of softmotion
        'moving_block': 'MovingBlockDataset',
        'unity_cloth': 'UnityClothDataset',
        'link_bot_video': 'LinkBotVideoDataset',
        'link_bot': 'LinkBotDataset',
    }
    dataset_class = dataset_mappings.get(dataset, dataset)
    dataset_class = globals().get(dataset_class)
    return dataset_class


def flatten_concat_pairs(ex_pos, ex_neg):
    flat_pair = tf.data.Dataset.from_tensors(ex_pos).concatenate(tf.data.Dataset.from_tensors(ex_neg))
    return flat_pair


def balance_dataset(dataset):
    positive_examples = dataset.filter(lambda example: tf.squeeze(tf.equal(example['constraints'], 1)))
    negative_examples = dataset.filter(lambda example: tf.squeeze(tf.equal(example['constraints'], 0)))
    # zipping takes the shorter of the two, hence why this makes it balanced
    balanced_dataset = tf.data.Dataset.zip((positive_examples, negative_examples))
    balanced_dataset = balanced_dataset.flat_map(flatten_concat_pairs)
    return balanced_dataset


def get_inputs(dataset_directory: str,
               dataset_class_name: str,
               dataset_hparams_dict: str,
               dataset_hparams: str,
               mode: str,
               epochs: int,
               batch_size: int,
               seed: int,
               balance_constraints_label: bool = False,
               shuffle: bool = True):
    dataset_hparams_dict = json.load(open(dataset_hparams_dict, 'r'))
    VideoDataset = get_dataset_class(dataset_class_name)
    my_dataset = VideoDataset(dataset_directory,
                              mode=mode,
                              num_epochs=epochs,
                              seed=seed,
                              hparams_dict=dataset_hparams_dict,
                              hparams=dataset_hparams)

    if balance_constraints_label:
        tf_dataset = my_dataset.make_dataset(batch_size=batch_size, use_batches=False)
        tf_dataset = balance_dataset(tf_dataset)
        tf_dataset = tf_dataset.batch(batch_size)
    else:
        tf_dataset = my_dataset.make_dataset(batch_size, shuffle=shuffle)

    iterator = tf_dataset.make_one_shot_iterator()
    handle = iterator.string_handle()
    iterator = tf.data.Iterator.from_string_handle(handle, tf_dataset.output_types,
                                                   tf_dataset.output_shapes)
    inputs = iterator.get_next()
    steps_per_epoch = int(my_dataset.num_examples_per_epoch() / batch_size)
    return my_dataset, inputs, steps_per_epoch


def get_dataset(dataset_directory: str,
                dataset_class_name: str,
                dataset_hparams_dict: str,
                dataset_hparams: str,
                mode: str,
                epochs: int,
                batch_size: int,
                seed: int):
    dataset_hparams_dict = json.load(open(dataset_hparams_dict, 'r'))
    VideoDataset = get_dataset_class(dataset_class_name)
    my_dataset = VideoDataset(dataset_directory,
                              mode=mode,
                              num_epochs=epochs,
                              seed=seed,
                              hparams_dict=dataset_hparams_dict,
                              hparams=dataset_hparams)

    tf_dataset = my_dataset.make_dataset(batch_size)
    return tf_dataset
