#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.transform import resize

from video_prediction.datasets.dataset_utils import get_inputs
from video_prediction.models import sna_model


def main():
    np.set_printoptions(suppress=True, linewidth=250, precision=4, threshold=64 * 64 * 3)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("dataset", type=str, help="dataset class name")
    parser.add_argument("dataset_hparams_dict", type=str, help="json file")
    parser.add_argument("model_hparams_dict", type=str, help="json file")
    parser.add_argument("checkpoint", help="directory with checkpoint or checkpoint name")
    parser.add_argument("--dataset-hparams", type=str, help="a string of comma separated list of dataset hyperparameters")
    parser.add_argument("--model-hparams", type=str, help="a string of comma separated list of dataset hyperparameters")
    parser.add_argument("--results-dir", type=str, default='results', help="ignored if output_gif_dir is specified")
    parser.add_argument("--mode", type=str, choices=['val', 'test'], default='val', help='mode for dataset, val or test.')
    parser.add_argument("--batch-size", type=int, default=8, help="number of samples in batch")
    parser.add_argument("--num-samples", type=int, help="number of samples in total (all of them by default)")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.seed is not None:
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    dataset, inputs, steps_per_epoch = get_inputs(args.input_dir,
                                                  args.dataset,
                                                  args.dataset_hparams_dict,
                                                  args.dataset_hparams,
                                                  args.mode,
                                                  epochs=1,
                                                  batch_size=args.batch_size,
                                                  seed=args.seed)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.graph.as_default()

    hparams_dict = json.loads(open(args.model_hparams_dict, 'r').read())
    hparams_dict.update({
        'context_frames': dataset.hparams.context_frames,
        'sequence_length': dataset.hparams.sequence_length,
    })
    model = sna_model.SNAVideoPredictionModel(hparams_dict=hparams_dict)

    model.build_graph(inputs)
    model.restore(sess, args.checkpoint)

    min_red = np.array([0.395, 0, 0])
    max_red = np.array([1.05, 0.05, 0.05])
    min_gray = np.array([0.57, .57, .57])
    max_gray = np.array([1.05, 1.05, 1.05])

    sample_ind = 0
    while True:
        if args.num_samples and sample_ind >= args.num_samples:
            break
        print("evaluation samples from %d to %d" % (sample_ind, sample_ind + args.batch_size))
        fetches = [model.outputs['gen_images'], inputs['sdf']]
        gen_images_batch, sdf_batch = sess.run(fetches)
        for gen_images_traj, sdf_traj in zip(gen_images_batch, sdf_batch):
            for t, (gen_image, sdf) in enumerate(zip(gen_images_traj, sdf_traj)):
                obstacle_map = np.expand_dims(np.flipud(np.squeeze(sdf).T), axis=2)
                obstacle_map = resize(obstacle_map, (gen_image.shape[0], gen_image.shape[1])) < 0
                # check if any of the pixels in the obstacle map are green, or black, or yellow
                masked_image = obstacle_map * gen_image

                row, col = np.where(np.any(masked_image > 0, axis=2))
                obstacle_pixels = masked_image[row, col]
                coords = list(zip(row, col))
                bad = False
                bad_coords = []
                for i, pixel in enumerate(obstacle_pixels):
                    not_red = np.any(pixel < min_red) or np.any(pixel > max_red)
                    not_gray = np.any(pixel < min_gray) or np.any(pixel > max_gray)
                    if not_red and not_gray:
                        bad_coords.append(coords[i])
                        bad = True

                if bad:
                    bad_coords = np.array(bad_coords)
                    plt.figure()
                    plt.imshow(gen_image)
                    plt.scatter(bad_coords[:, 1], bad_coords[:, 0], c='white', marker='D', s=10, edgecolors='k')
                    plt.show()

        sample_ind += args.batch_size


if __name__ == '__main__':
    main()
