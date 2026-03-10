# from .kitti_dataset_1215 import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .scared_dataset import ScaredDatset # 导入新数据集

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    # "kitti": KITTIDataset,
    "scared": ScaredDatset #注册SCARED
}
