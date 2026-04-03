import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SUIMDataset(CustomDataset):
    """SUIM dataset.
    Args:
        split (str): Split txt file for SUIM.
    """
    CLASSES = ('BW', 'HD', 'PF', 'WR', 'RO', 'RI', 'FV', 'SR')
    PALETTE = [[0, 0, 0], [0, 0, 255], [0, 255, 0], [0, 255, 255], [255, 0, 0],
               [255, 0, 255], [255, 255, 0], [255, 255, 255]]

    def __init__(self, split, **kwargs):
        super(SUIMDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
