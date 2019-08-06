import numpy as np
from PIL import Image

from . import losses
from . import metrics
from . import ops


def load_data(context_image_filenames, context_states_filename, actions_filename):
    actions = np.genfromtxt(actions_filename, delimiter=',', dtype=np.float32)
    context_states = np.genfromtxt(context_states_filename, delimiter=',', dtype=np.float32)
    context_images = []
    for time_step_idx, context_image_filename in enumerate(context_image_filenames):
        rgba_image_uint8 = np.array(Image.open(context_image_filename), dtype=np.uint8)
        rgb_image_uint8 = rgba_image_uint8[:, :, :3]  # remove alpha channel if it exists
        rgb_image_float = rgb_image_uint8.astype(np.float32) / 255.0
        context_images.append(rgb_image_float)
    context_images = np.array(context_images)

    return context_states, context_images, actions

