# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Model architecture for predictive model, including CDNA (DNA and STP retransformed for my use)"""

import itertools

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.contrib.slim import add_arg_scope
from tensorflow.contrib.slim import layers

from video_prediction.models import VideoPredictionModel

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12


@add_arg_scope
def basic_conv_lstm_cell(inputs, state, num_channels, filter_size=5, forget_bias=1.0, scope=None, reuse=None):
    """Basic LSTM recurrent network cell, with 2D convolution connctions.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
    Args:
        inputs: input Tensor, 4D, batch x height x width x channels.
        state: state Tensor, 4D, batch x height x width x channels.
        num_channels: the number of output channels in the layer.
        filter_size: the shape of the each convolution filter.
        forget_bias: the initial value of the forget biases.
        scope: Optional scope for variable_scope.
        reuse: whether or not the layer and the variables should be reused.
    Returns:
        a tuple of tensors representing output and the new state.
    """
    if state is None:
        state = tf.zeros(inputs.get_shape().as_list()[:3] + [2 * num_channels], name='init_state')

    with tf.variable_scope(scope,
                           'BasicConvLstmCell',
                           [inputs, state],
                           reuse=reuse):
        inputs.get_shape().assert_has_rank(4)
        state.get_shape().assert_has_rank(4)
        c, h = tf.split(axis=3, num_or_size_splits=2, value=state)
        inputs_h = tf.concat(values=[inputs, h], axis=3)
        # Parameters of gates are concatenated into one conv for efficiency.
        i_j_f_o = layers.conv2d(inputs_h,
                                4 * num_channels, [filter_size, filter_size],
                                stride=1,
                                activation_fn=None,
                                scope='Gates',
                                )

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(value=i_j_f_o, num_or_size_splits=4, axis=3)

        new_c = c * tf.sigmoid(f + forget_bias) + tf.sigmoid(i) * tf.tanh(j)
        new_h = tf.tanh(new_c) * tf.sigmoid(o)

        return new_h, tf.concat(values=[new_c, new_h], axis=3)


class Prediction_Model(object):

    def __init__(self,
                 images,
                 actions=None,
                 states=None,
                 iter_num=-1.0,
                 pix_distributions1=None,
                 pix_distributions2=None,
                 conf=None):

        self.pix_distributions1 = pix_distributions1
        self.pix_distributions2 = pix_distributions2
        self.actions = actions
        self.iter_num = iter_num
        self.conf = conf
        print("[SNA Model Configuration]")
        for k, v in self.conf.items():
            if v == '':
                print('\t{} (Yes)'.format(k))
            else:
                print('\t{}: {}'.format(k, v))
        self.images = images

        self.k = conf['schedsamp_k']
        self.use_state = conf['use_state']
        self.num_masks = conf['num_masks']
        self.context_length = conf['context_frames']

        self.batch_size, self.img_height, self.img_width, self.color_channels = [int(i) for i in images[0].get_shape()[0:4]]
        self.lstm_func = basic_conv_lstm_cell

        # Generated robot states and images.
        self.gen_states = []
        self.gen_images = []
        self.gen_masks = []
        self.gen_cdna_kernels = []
        self.gen_made_up_images = []
        self.gen_masked_images = []
        self.gen_background_images = []

        self.transformed_images = []

        self.transformed_pix_distrib1 = []
        self.transformed_pix_distrib2 = []

        self.states = states
        self.gen_pix_distrib1 = []
        self.gen_made_up_pix_distrib1 = []
        self.gen_background_pix_distrib1 = []
        self.gen_masked_pix_distrib1 = []
        self.gen_pix_distrib2 = []
        self.gen_made_up_pix_distrib2 = []
        self.gen_background_pix_distrib2 = []
        self.gen_masked_pix_distribs2 = []

        self.trafos = []

    def build(self):
        batch_size, img_height, img_width, color_channels = self.images[0].get_shape()[0:4]
        lstm_func = basic_conv_lstm_cell

        if self.actions is None:
            self.actions = [None for _ in self.images]

        if self.k == -1:
            no_scheduled_sampling = True
            num_ground_truth = None
        else:
            # Scheduled sampling:
            # Calculate number of ground-truth frames to pass in.
            # Inverse sigmoid decay
            num_ground_truth = tf.to_int32(
                tf.round(tf.to_float(batch_size) * (self.k / (self.k + tf.exp(self.iter_num / self.k)))))
            no_scheduled_sampling = False

        # LSTM state sizes and states.
        if 'lstm_size' in self.conf:
            lstm_size = self.conf['lstm_size']
            print('using lstm size', lstm_size)
        else:
            ngf = self.conf['ngf']
            lstm_size = np.int32(np.array([ngf, ngf * 2, ngf * 4, ngf * 2, ngf]))

        # TODO: better abstraction here, also 2, 4 are not used?!
        lstm_states = [None] * 7

        for t, action in enumerate(self.actions):
            # Reuse variables after the first timestep.
            reuse = bool(self.gen_images)

            done_warm_start = len(self.gen_images) > self.context_length - 1
            with slim.arg_scope([lstm_func, slim.layers.conv2d, slim.layers.fully_connected, tf_layers.layer_norm,
                                 slim.layers.conv2d_transpose], reuse=reuse):

                if done_warm_start:
                    if no_scheduled_sampling:
                        # Feed in generated image.
                        prev_image = self.gen_images[-1]  # 64x64x6
                        if self.pix_distributions1 is not None:
                            prev_pix_distrib1 = self.gen_pix_distrib1[-1]
                            if 'ndesig' in self.conf:
                                prev_pix_distrib2 = self.gen_pix_distrib2[-1]
                    else:
                        # Scheduled sampling, feeds the real images at first but
                        # transitions to feeding only the generate images later on
                        prev_image = scheduled_sample(self.images[t], self.gen_images[-1], batch_size, num_ground_truth)
                else:
                    # Feed in the context states, images, and pixel distributions
                    current_state = self.states[t]
                    prev_image = self.images[t]
                    if self.pix_distributions1 is not None:
                        prev_pix_distrib1 = self.pix_distributions1[t]
                        if 'ndesig' in self.conf:
                            prev_pix_distrib2 = self.pix_distributions2[t]
                        if len(prev_pix_distrib1.get_shape()) == 3:
                            prev_pix_distrib1 = tf.expand_dims(prev_pix_distrib1, -1)
                            if 'ndesig' in self.conf:
                                prev_pix_distrib2 = tf.expand_dims(prev_pix_distrib2, -1)

                if 'refeed_firstimage' in self.conf:
                    raise NotImplementedError()
                else:
                    input_image = prev_image

                # Predicted state is always fed back in
                if 'ignore_state_action' not in self.conf:
                    state_action = tf.concat(axis=1, values=[action, current_state])

                enc6, hidden5, lstm_states = self.encoder_decoder_fn(action, batch_size, input_image, lstm_func, lstm_size,
                                                                     lstm_states, state_action)

                if 'transform_from_firstimage' in self.conf:
                    raise NotImplementedError()

                if self.conf['model'] == 'DNA':
                    raise NotImplementedError()

                if self.conf['model'] == 'CDNA':
                    if 'gen_pix' in self.conf:
                        # Using largest hidden state for predicting a new image layer.
                        enc7 = slim.layers.conv2d_transpose(enc6, color_channels, 1, stride=1, scope='convt4', activation_fn=None)
                        # This allows the network to also generate one image from scratch,
                        # which is useful when regions of the image become unoccluded.
                        gen_made_up_image = tf.nn.sigmoid(enc7)
                        self.gen_made_up_images.append(gen_made_up_image)
                        transformed_list = [gen_made_up_image]
                        extra_masks = 2
                    else:
                        transformed_list = []
                        extra_masks = 1

                    cdna_input = tf.reshape(hidden5, [int(batch_size), -1])
                    new_transformed, cdna_kerns_for_viz = self.cdna_transformation(prev_image, cdna_input, reuse_sc=reuse)
                    self.gen_cdna_kernels.append(cdna_kerns_for_viz)
                    transformed_list += new_transformed
                    # NOTE: this does not include the background image, which is untransformed
                    self.transformed_images.append(tf.stack(transformed_list, axis=1))

                    if self.pix_distributions1 is not None:
                        transf_distrib_ndesig1, _ = self.cdna_transformation(prev_pix_distrib1, cdna_input, reuse_sc=True)
                        self.transformed_pix_distrib1.append(tf.stack(transf_distrib_ndesig1, axis=1))
                        if 'ndesig' in self.conf:
                            transf_distrib_ndesig2, _ = self.cdna_transformation(prev_pix_distrib2, cdna_input, reuse_sc=True)

                            self.transformed_pix_distrib2.append(transf_distrib_ndesig2)

                if self.conf['model'] == 'STP':
                    raise NotImplementedError()

                if 'first_image_background' in self.conf:
                    background = self.images[0]
                else:
                    background = prev_image

                self.gen_background_images.append(background)
                # NOTE: the first image in masked_images will be the masked background image
                fused_images, mask_list, masked_images = self.mask_and_fuse_transformed_images(enc6, background,
                                                                                               transformed_list,
                                                                                               scope='convt7_cam2',
                                                                                               extra_masks=extra_masks)
                self.gen_masked_images.append(tf.stack(masked_images, axis=1))
                self.gen_images.append(fused_images)
                self.gen_masks.append(tf.stack(mask_list, axis=1))

                if self.pix_distributions1 is not None:
                    mask_fuse_pix_distrib1_result = self.mask_and_fuse_transformed_pix_distribs(extra_masks,
                                                                                                mask_list,
                                                                                                self.pix_distributions1,
                                                                                                prev_pix_distrib1,
                                                                                                transf_distrib_ndesig1)
                    background_pix_distrib1 = mask_fuse_pix_distrib1_result[0]
                    # NOTE: is "previous_pix_distrib" the equivalent of the "made_up_image"
                    made_up_pix_distrib1 = mask_fuse_pix_distrib1_result[1]
                    # NOTE: the masked version of the made-up pix is in the 0th element of masked_pix_distribs1
                    masked_pix_distribs1 = mask_fuse_pix_distrib1_result[2]
                    fused_pix_distrib1 = mask_fuse_pix_distrib1_result[3]

                    self.gen_pix_distrib1.append(fused_pix_distrib1)
                    self.gen_made_up_pix_distrib1.append(made_up_pix_distrib1)
                    self.gen_background_pix_distrib1.append(background_pix_distrib1)
                    self.gen_masked_pix_distrib1.append(tf.stack(masked_pix_distribs1, axis=1))
                    # ndeign means there are two "designated" source/target pixel pairs
                    # this code base only supports one or two pixel pairs
                    if 'ndesig' in self.conf:
                        raise NotImplementedError()

                if int(current_state.get_shape()[1]) == 0:
                    raise NotImplementedError()
                    # pass in zeros for the state if there is no state vector for this dataset?
                    # not sure what's going on here
                else:
                    current_state = slim.layers.fully_connected(
                        state_action,
                        int(current_state.get_shape()[1]),
                        scope='state_pred',
                        activation_fn=None)

                self.gen_states.append(current_state)

    def mask_and_fuse_transformed_images(self, enc6, background_image, transformed, scope, extra_masks):
        masks = slim.layers.conv2d_transpose(enc6, (self.conf['num_masks'] + extra_masks), 1, stride=1, activation_fn=None,
                                             scope=scope)

        img_height = 64
        img_width = 64
        num_masks = self.conf['num_masks']

        if self.conf['model'] == 'DNA':
            raise NotImplementedError()

        # the total number of masks is num_masks +extra_masks because of background and generated pixels!
        masks = tf.reshape(
            tf.nn.softmax(tf.reshape(masks, [-1, num_masks + extra_masks])),
            [int(self.batch_size), int(img_height), int(img_width), num_masks + extra_masks])
        mask_list = tf.split(axis=3, num_or_size_splits=num_masks + extra_masks, value=masks)
        fused_image = mask_list[0] * background_image

        assert len(transformed) == len(mask_list[1:])
        masked_images = [fused_image]
        # the same transformations are applied to all channels of the image
        # this list includes the "made-up" image
        for layer, mask in zip(transformed, mask_list[1:]):
            masked_images.append(layer * mask)
            fused_image += layer * mask

        return fused_image, mask_list, masked_images

    def mask_and_fuse_transformed_pix_distribs(self, extra_masks, mask_list, pix_distributions, prev_pix_distrib,
                                               transformed_pix_distrib):
        if 'first_image_background' in self.conf:
            # Take the first image in from the pix_distribs tensor in feed_dict
            background_pix = pix_distributions[0]
            if len(background_pix.get_shape()) == 3:
                background_pix = tf.expand_dims(background_pix, -1)
        else:
            background_pix = prev_pix_distrib
        fused_pix_distrib = mask_list[0] * background_pix
        masked_pix_distribs = [fused_pix_distrib]
        if 'gen_pix' in self.conf:
            # if we are using the "made-up" image, then in the pixel distribution image we use the background image in place
            # This is essentially saying that the image from which you should
            # In theory the "made-up" image should be for regions of the image are no longer occluded,
            # so the mask should indicate areas which are (recently) no longer occluded. When applied to the background image,
            # we're essentially copying parts from the background image that are no longer occluded. This seems redundant with the
            # other use of the background image. One interesting question is -- which of these does the model actually use?
            made_up_pix_distrib = prev_pix_distrib
            masked_made_up_pix_distrib = mask_list[1] * made_up_pix_distrib
            masked_pix_distribs.append(masked_made_up_pix_distrib)
            fused_pix_distrib += masked_made_up_pix_distrib
        else:
            made_up_pix_distrib = None
        for i in range(self.num_masks):
            masked_transformed_pix_distrib = transformed_pix_distrib[i] * mask_list[i + extra_masks]
            masked_pix_distribs.append(masked_transformed_pix_distrib)
            fused_pix_distrib += masked_transformed_pix_distrib
        pix_distrib_sum = tf.reduce_sum(fused_pix_distrib, axis=(1, 2), keepdims=True)
        fused_pix_distrib /= pix_distrib_sum
        # normalize the individual outputs the same way the combined output is normalized
        normalized_masked_pix_distribs = []
        for masked_pix_distrib in masked_pix_distribs:
            normalized_pix_distrib_output = masked_pix_distrib / pix_distrib_sum
            normalized_masked_pix_distribs.append(normalized_pix_distrib_output)
        # prev_pix_distrib is what's used in place of the "made-up" image, and will either be the previous predicted pix_distrib
        # or (in our usage) the first context pix_distrib
        return background_pix, made_up_pix_distrib, normalized_masked_pix_distribs, fused_pix_distrib

    def encoder_decoder_fn(self, action, batch_size, input_image, lstm_func, lstm_size, lstm_states, state_action):
        """
        :return:
            enc6: the representation use to construct the masks
            hidden5: the representation use to construct the CDNA kernels
            lstm_states: hidden lstm states
        """
        lstm_state1, lstm_state2, lstm_state3, lstm_state4, lstm_state5, lstm_state6, lstm_state7 = lstm_states
        enc0 = slim.layers.conv2d(  # 32x32x32
            input_image,
            32, [5, 5],
            stride=2,
            scope='scale1_conv1',
            normalizer_fn=tf_layers.layer_norm,
            normalizer_params={'scope': 'layer_norm1'})
        hidden1, lstm_state1 = lstm_func(  # 32x32x16
            enc0, lstm_state1, lstm_size[0], scope='state1')
        hidden1 = tf_layers.layer_norm(hidden1, scope='layer_norm2')
        enc1 = slim.layers.conv2d(  # 16x16x16
            hidden1, hidden1.get_shape()[3], [3, 3], stride=2, scope='conv2')
        hidden3, lstm_state3 = lstm_func(  # 16x16x32
            enc1, lstm_state3, lstm_size[1], scope='state3')
        hidden3 = tf_layers.layer_norm(hidden3, scope='layer_norm4')
        enc2 = slim.layers.conv2d(  # 8x8x32
            hidden3, hidden3.get_shape()[3], [3, 3], stride=2, scope='conv3')
        if 'ignore_state_action' not in self.conf:
            # Pass in state and action.
            if 'ignore_state' in self.conf:
                lowdim = action
                print('ignoring state')
            else:
                lowdim = state_action

            smear = tf.reshape(lowdim, [int(batch_size), 1, 1, int(lowdim.get_shape()[1])])
            smear = tf.tile(smear, [1, int(enc2.get_shape()[1]), int(enc2.get_shape()[2]), 1])
            enc2 = tf.concat(axis=3, values=[enc2, smear])
        else:
            print('ignoring states and actions')
        enc3 = slim.layers.conv2d(  # 8x8x32
            enc2, hidden3.get_shape()[3], [1, 1], stride=1, scope='conv4')
        hidden5, lstm_state5 = lstm_func(  # 8x8x64
            enc3, lstm_state5, lstm_size[2], scope='state5')
        hidden5 = tf_layers.layer_norm(hidden5, scope='layer_norm6')
        enc4 = slim.layers.conv2d_transpose(  # 16x16x64
            hidden5, hidden5.get_shape()[3], 3, stride=2, scope='convt1')
        hidden6, lstm_state6 = lstm_func(  # 16x16x32
            enc4, lstm_state6, lstm_size[3], scope='state6')
        hidden6 = tf_layers.layer_norm(hidden6, scope='layer_norm7')
        if 'noskip' not in self.conf:
            # Skip connection.
            hidden6 = tf.concat(axis=3, values=[hidden6, enc1])  # both 16x16
        enc5 = slim.layers.conv2d_transpose(  # 32x32x32
            hidden6, hidden6.get_shape()[3], 3, stride=2, scope='convt2')
        hidden7, lstm_state7 = lstm_func(  # 32x32x16
            enc5, lstm_state7, lstm_size[4], scope='state7')
        hidden7 = tf_layers.layer_norm(hidden7, scope='layer_norm8')
        if 'noskip' not in self.conf:
            # Skip connection.
            hidden7 = tf.concat(axis=3, values=[hidden7, enc0])  # both 32x32
        enc6 = slim.layers.conv2d_transpose(  # 64x64x16
            hidden7,
            hidden7.get_shape()[3], 3, stride=2, scope='convt3',
            normalizer_fn=tf_layers.layer_norm,
            normalizer_params={'scope': 'layer_norm9'})
        lstm_states = lstm_state1, lstm_state2, lstm_state3, lstm_state4, lstm_state5, lstm_state6, lstm_state7
        return enc6, hidden5, lstm_states

    def cdna_transformation(self, prev_image, cdna_input, reuse_sc=None):
        """Apply convolutional dynamic neural advection to previous image.

        Args:
          prev_image: previous image to be transformed.
          cdna_input: hidden layer to be used for computing CDNA kernels.
          num_masks: the number of masks and hence the number of CDNA transformations.
          color_channels: the number of color channels in the images.
        Returns:
          List of images transformed by the predicted CDNA kernels.
        """
        batch_size = int(cdna_input.get_shape()[0])
        height = int(prev_image.get_shape()[1])
        width = int(prev_image.get_shape()[2])

        DNA_KERN_SIZE = self.conf['kern_size']
        num_masks = self.conf['num_masks']
        color_channels = int(prev_image.get_shape()[3])

        # Predict kernels using linear function of last hidden layer.
        cdna_kerns = slim.layers.fully_connected(
            cdna_input,
            DNA_KERN_SIZE * DNA_KERN_SIZE * num_masks,
            scope='cdna_params',
            activation_fn=None,
            reuse=reuse_sc)

        # Reshape and normalize.
        cdna_kerns = tf.reshape(
            cdna_kerns, [batch_size, DNA_KERN_SIZE, DNA_KERN_SIZE, 1, num_masks])
        cdna_kerns = tf.nn.relu(cdna_kerns - RELU_SHIFT) + RELU_SHIFT
        norm_factor = tf.reduce_sum(cdna_kerns, [1, 2, 3], keepdims=True)
        cdna_kerns /= norm_factor
        cdna_kerns_for_viz = tf.transpose(cdna_kerns, [0, 4, 1, 2, 3])

        # Transpose and reshape.
        cdna_kerns = tf.transpose(cdna_kerns, [1, 2, 0, 4, 3])
        cdna_kerns = tf.reshape(cdna_kerns, [DNA_KERN_SIZE, DNA_KERN_SIZE, batch_size, num_masks])
        prev_image = tf.transpose(prev_image, [3, 1, 2, 0])

        transformed = tf.nn.depthwise_conv2d(prev_image, cdna_kerns, [1, 1, 1, 1], 'SAME')

        # Transpose and reshape.
        transformed = tf.reshape(transformed, [color_channels, height, width, batch_size, num_masks])
        transformed = tf.transpose(transformed, [3, 1, 2, 0, 4])
        transformed = tf.unstack(value=transformed, axis=-1)

        return transformed, cdna_kerns_for_viz


def scheduled_sample(ground_truth_x, generated_x, batch_size, num_ground_truth):
    """Sample batch with specified mix of ground truth and generated data_files points.

    Args:
      ground_truth_x: tensor of ground-truth data_files points.
      generated_x: tensor of generated data_files points.
      batch_size: batch size
      num_ground_truth: number of ground-truth examples to include in batch.
    Returns:
      New batch with num_ground_truth sampled from ground_truth_x and the rest
      from generated_x.
    """
    idx = tf.random_shuffle(tf.range(int(batch_size)))
    ground_truth_idx = tf.gather(idx, tf.range(num_ground_truth))
    generated_idx = tf.gather(idx, tf.range(num_ground_truth, int(batch_size)))

    ground_truth_examps = tf.gather(ground_truth_x, ground_truth_idx)
    generated_examps = tf.gather(generated_x, generated_idx)
    return tf.dynamic_stitch([ground_truth_idx, generated_idx],
                             [ground_truth_examps, generated_examps])


def generator_fn(inputs, mode, hparams):
    images = tf.unstack(inputs['images'], axis=0)
    actions = tf.unstack(inputs['actions'], axis=0)
    states = tf.unstack(inputs['states'], axis=0)
    pix_distributions1 = tf.unstack(inputs['pix_distribs'], axis=0) if 'pix_distribs' in inputs else None
    iter_num = tf.to_float(tf.train.get_or_create_global_step())

    if isinstance(hparams.kernel_size, (tuple, list)):
        kernel_height, kernel_width = hparams.kernel_size
        assert kernel_height == kernel_width
        kern_size = kernel_height
    else:
        kern_size = hparams.kernel_size

    schedule_sampling_k = hparams.schedule_sampling_k if mode == 'train' else -1
    conf = {
        'context_frames': hparams.context_frames,  # of frames before predictions.' ,
        'use_state': 1,  # 'Whether or not to give the state+action to the model' ,
        'ngf': hparams.ngf,
        'model': hparams.transformation.upper(),  # 'model architecture to use - CDNA' ,
        'num_masks': hparams.num_masks,  # 'number of masks, 10 for CDNA' ,
        'schedsamp_k': schedule_sampling_k,  # 'The k hyperparameter for scheduled sampling -1 for no scheduled sampling.' ,
        'kern_size': kern_size,  # size of DNA kerns
    }
    if hparams.first_image_background:
        conf['first_image_background'] = ''
    if hparams.generate_scratch_image:
        conf['gen_pix'] = ''

    m = Prediction_Model(images, actions, states,
                         pix_distributions1=pix_distributions1,
                         iter_num=iter_num, conf=conf)
    m.build()
    outputs = {
        'gen_cdna_kernels': tf.stack(m.gen_cdna_kernels, axis=0),
        'gen_masks': tf.stack(m.gen_masks, axis=0),
        'gen_transformed_images': tf.stack(m.transformed_images, axis=0),
        'gen_masked_images': tf.stack(m.gen_masked_images, axis=0),
        'gen_background_images': tf.stack(m.gen_background_images, axis=0),
        'gen_images': tf.stack(m.gen_images, axis=0),
        'gen_states': tf.stack(m.gen_states, axis=0),
    }
    if hparams.generate_scratch_image:
        outputs['gen_made_up_images'] = tf.stack(m.gen_made_up_images, axis=0)
        outputs['gen_made_up_pix_distribs'] = tf.stack(m.gen_made_up_pix_distrib1, axis=0)
    else:
        outputs['gen_made_up_images'] = None
        outputs['gen_made_up_pix_distribs'] = None

    if 'pix_distribs' in inputs:
        # TODO: support more than one pix pair
        outputs['gen_pix_distribs'] = tf.stack(m.gen_pix_distrib1, axis=0)
        outputs['gen_background_pix_distribs'] = tf.stack(m.gen_background_pix_distrib1, axis=0)
        outputs['gen_transformed_pix_distribs'] = tf.stack(m.transformed_pix_distrib1, axis=0)
        outputs['gen_masked_pix_distribs'] = tf.stack(m.gen_masked_pix_distrib1, axis=0)
    return outputs


class SNAVideoPredictionModel(VideoPredictionModel):
    def __init__(self, *args, **kwargs):
        super(SNAVideoPredictionModel, self).__init__(
            generator_fn, *args, **kwargs)

    def get_default_hparams_dict(self):
        default_hparams = super(SNAVideoPredictionModel, self).get_default_hparams_dict()
        hparams = dict(
            batch_size=32,
            l1_weight=0.0,
            l2_weight=1.0,
            ngf=16,  # number of generator filters
            transformation='cdna',
            kernel_size=(5, 5),
            num_masks=10,
            first_image_background=True,
            generate_scratch_image=True,
            schedule_sampling_k=900.0,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))
