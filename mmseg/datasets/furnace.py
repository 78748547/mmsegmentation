# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class BaoGangFurnaceDataset(BaseSegDataset):
    """BaoGang Furnace dataset.

    Args:
        split (str): Split txt file for BaoGang Furnace.
    """
    METAINFO = dict(
        classes=('background', 'beat', 'cap', 'generator', 'joint'),
        palette=[[0, 0, 0], [0, 255, 206], [199, 252, 0], [255, 33, 33], [0, 255, 61]])

    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args) and osp.isfile(self.ann_file)
