import itertools

from .video_dataset import VideoDataset
from .softmotion_dataset import SoftmotionVideoDataset


class CartgripperVideoDataset(SoftmotionVideoDataset):
    def __init__(self, *args, **kwargs):
        VideoDataset.__init__(self, *args, **kwargs)
        self.state_like_names_and_shapes['images'] = '%d/image_view0/encoded', (48, 64, 3)
        if self.hparams.use_state:
            self.state_like_names_and_shapes['states'] = '%d/endeffector_pos', (6,)
            self.action_like_names_and_shapes['actions'] = '%d/action', (3,)
        self._infer_seq_length_and_setup()

    def get_default_hparams_dict(self):
        default_hparams = super(CartgripperVideoDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=2,
            sequence_length=15,
            time_shift=3,
            use_state=True,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))
