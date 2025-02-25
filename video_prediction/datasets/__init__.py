from .video_dataset import BaseVideoDataset
from .video_dataset import VideoDataset
from .google_robot_dataset import GoogleRobotVideoDataset
from .link_bot_dataset import LinkBotDataset
from .moving_block_dataset import MovingBlockDataset
from .softmotion_dataset import SoftmotionVideoDataset
from .sv2p_dataset import SV2PVideoDataset
from .unity_cloth_dataset import UnityClothDataset


def get_dataset_class(dataset):
    dataset_mappings = {
        'google_robot': 'GoogleRobotVideoDataset',
        'sv2p': 'SV2PVideoDataset',
        'softmotion': 'SoftmotionVideoDataset',
        'bair': 'SoftmotionVideoDataset',  # alias of softmotion
        'moving_block': 'MovingBlockDataset',
        'unity_cloth': 'UnityClothDataset',
    }
    dataset_class = dataset_mappings.get(dataset, dataset)
    dataset_class = globals().get(dataset_class)
    return dataset_class
