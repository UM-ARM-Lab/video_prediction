import json
from typing import Tuple, Optional, List, Union

import numpy as np
from PIL import Image

from link_bot_data.link_bot_dataset_utils import balance_xy_dataset
from video_prediction.datasets import get_dataset_class


def get_dataset(dataset_directory: str,
                dataset_class_name: str,
                dataset_hparams_dict: Union[str, dict],
                dataset_hparams: str,
                mode: str,
                epochs: Optional[int],
                batch_size: int,
                seed: int,
                balance_key: Optional[str] = None,
                shuffle: bool = True):
    if isinstance(dataset_hparams_dict, str):
        dataset_hparams_dict = json.load(open(dataset_hparams_dict, 'r'))

    dataset_class = get_dataset_class(dataset_class_name)
    my_dataset = dataset_class(dataset_directory,
                               mode=mode,
                               num_epochs=epochs,
                               seed=seed,
                               hparams_dict=dataset_hparams_dict,
                               hparams=dataset_hparams)

    if balance_key is not None:
        tf_dataset = my_dataset.make_dataset(batch_size=batch_size, use_batches=False)
        tf_dataset = balance_xy_dataset(tf_dataset, balance_key)
        tf_dataset = tf_dataset.batch(batch_size)
    else:
        tf_dataset = my_dataset.make_dataset(batch_size, shuffle=shuffle)

    return my_dataset, tf_dataset


def get_iterators(dataset_directory: str,
                  dataset_class_name: str,
                  dataset_hparams_dict: str,
                  dataset_hparams: str,
                  mode: str,
                  epochs: Optional[int],
                  batch_size: int,
                  seed: int,
                  balance_key: Optional[str] = None,
                  shuffle: bool = True):
    my_dataset, tf_dataset = get_dataset(dataset_directory, dataset_class_name, dataset_hparams_dict, dataset_hparams, mode,
                                         epochs, batch_size, seed, balance_key, shuffle)

    iterator = tf_dataset.make_one_shot_iterator()
    # handle = iterator.string_handle()
    # iterator = tf.data.Iterator.from_string_handle(handle, tf_dataset.output_types,
    #                                                tf_dataset.output_shapes)
    steps_per_epoch = int(my_dataset.num_examples_per_epoch() / batch_size)
    return my_dataset, iterator, steps_per_epoch


def get_inputs(dataset_directory: str,
               dataset_class_name: str,
               dataset_hparams_dict: str,
               dataset_hparams: str,
               mode: str,
               epochs: Optional[int],
               batch_size: int,
               seed: int,
               balance_key: Optional[str] = None,
               shuffle: bool = True):
    my_dataset, iterator, steps_per_epoch = get_iterators(dataset_directory, dataset_class_name, dataset_hparams_dict,
                                                          dataset_hparams, mode, epochs, batch_size, seed,
                                                          balance_key, shuffle)
    inputs = iterator.get_next()
    return my_dataset, inputs, steps_per_epoch


def load_data(context_image_filenames: List[str],
              context_states_filename: str,
              context_actions_filename: str,
              actions_filename: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if actions_filename is not None:
        actions = np.atleast_2d(np.genfromtxt(actions_filename, delimiter=',', dtype=np.float32))
    else:
        actions = None
    context_actions = np.atleast_2d(np.genfromtxt(context_actions_filename, delimiter=',', dtype=np.float32))
    context_states = np.genfromtxt(context_states_filename, delimiter=',', dtype=np.float32)
    context_images = []
    for time_step_idx, context_image_filename in enumerate(context_image_filenames):
        rgba_image_uint8 = np.array(Image.open(context_image_filename), dtype=np.uint8)
        rgb_image_uint8 = rgba_image_uint8[:, :, :3]  # remove alpha channel if it exists
        rgb_image_float = rgb_image_uint8.astype(np.float32) / 255.0
        context_images.append(rgb_image_float)
    context_images = np.array(context_images)

    return context_states, context_images, context_actions, actions
