import re

import tensorflow as tf
from google.protobuf.json_format import MessageToDict

from video_prediction.datasets.non_video_dataset import NonVideoDataset


class LinkBotDataset(NonVideoDataset):
    def __init__(self, *args, **kwargs):
        super(LinkBotDataset, self).__init__(*args, **kwargs)

        # infer name of image feature
        options = tf.python_io.TFRecordOptions(compression_type=self.hparams.compression_type)
        example = next(tf.python_io.tf_record_iterator(self.filenames[0], options=options))
        dict_message = MessageToDict(tf.train.Example.FromString(example))
        feature = dict_message['features']['feature']
        image_names = set()
        for name in feature.keys():
            m = re.search('\d+/(\w+)/encoded', name)
            if m:
                image_names.add(m.group(1))
        # look for image_aux1 and image_view0 in that order of priority
        image_name = None
        for name in ['image_aux1', 'image_view0']:
            if name in image_names:
                image_name = name
                break
        if not image_name:
            if len(image_names) == 1:
                image_name = image_names.pop()
            else:
                raise ValueError('The examples have images under more than one name.')
        self.state_like_names_and_shapes['images'] = '%%d/%s/encoded' % image_name, self.hparams.image_shape
        self.state_like_names_and_shapes['states'] = '%d/endeffector_pos', (2,)
        self.state_like_names_and_shapes['rope_configurations'] = '%d/rope_configuration', (self.hparams.rope_config_dim,)
        self.state_like_names_and_shapes['constraints'] = '%d/constraint', (1,)
        self.action_like_names_and_shapes['actions'] = '%d/action', (2,)
        self.trajectory_constant_names_and_shapes['sdf'] = 'sdf/sdf', [self.hparams.sdf_shape[0], self.hparams.sdf_shape[1], 1]
        self.trajectory_constant_names_and_shapes['sdf_gradient'] = 'sdf/gradient', [self.hparams.sdf_shape[0],
                                                                                     self.hparams.sdf_shape[1], 2]
        self.trajectory_constant_names_and_shapes['sdf_resolution'] = 'sdf/resolution', (2,)
        self.trajectory_constant_names_and_shapes['sdf_origin'] = 'sdf/origin', (2,)
        self._check_or_infer_shapes()

    def get_default_hparams_dict(self):
        default_hparams = super(LinkBotDataset, self).get_default_hparams_dict()
        hparams = dict(
            image_shape=[64, 64, 3],
            sdf_shape=[101, 101],
            rope_config_dim=6,
        )
        hparams.update(default_hparams)
        return hparams

    def parser(self, serialized_example):
        state_like_seqs, action_like_seqs = super(LinkBotDataset, self).parser(serialized_example)
        return state_like_seqs, action_like_seqs
