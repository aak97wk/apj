
import jittor as jt
from jittor import init
from jittor import nn
from jittor.dataset.dataset import Dataset
import bisect
from alphapose.models.builder import DATASET, build_dataset

@DATASET.register_module
class ConcatDataset(Dataset):
    'Custom Concat dataset.\n    Annotation file must be in `coco` format.\n\n    Parameters\n    ----------\n    train: bool, default is True\n        If true, will set as training mode.\n    dpg: bool, default is False\n        If true, will activate `dpg` for data augmentation.\n    skip_empty: bool, default is False\n        Whether skip entire image if no valid label is found.\n    cfg: dict, dataset configuration.\n    '

    def __init__(self, train=True, dpg=False, skip_empty=True, **cfg):
        super().__init__()
        self._cfg = cfg
        self._subset_cfg_list = cfg['SET_LIST']
        self._preset_cfg = cfg['PRESET']
        self._mask_id = [item['MASK_ID'] for item in self._subset_cfg_list]
        self.num_joints = self._preset_cfg['NUM_JOINTS']
        self._subsets = []
        self._subset_size = [0]
        for _subset_cfg in self._subset_cfg_list:
            subset = build_dataset(_subset_cfg, preset_cfg=self._preset_cfg, train=train)
            self._subsets.append(subset)
            self._subset_size.append(len(subset))
        self.cumulative_sizes = self.cumsum(self._subset_size)
        self.total_len = self.cumulative_sizes[(- 1)]
    def __getitem__(self, idx):
        assert (idx >= 0)
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        dataset_idx -= 1
        sample_idx = (idx - self.cumulative_sizes[dataset_idx])
        sample = self._subsets[dataset_idx][sample_idx]
        (img, label, label_mask, img_id, bbox) = sample
        K = label.shape[0]
        expend_label = jt.zeros((self.num_joints, *label.shape[1:]), dtype=label.dtype)
        expend_label_mask = jt.zeros((self.num_joints, *label_mask.shape[1:]), dtype=label_mask.dtype)
        expend_label[self._mask_id[dataset_idx]:(self._mask_id[dataset_idx] + K)] = label
        expend_label_mask[self._mask_id[dataset_idx]:(self._mask_id[dataset_idx] + K)] = label_mask
        return (img, expend_label, expend_label_mask, img_id, bbox)

    def __len__(self):
        return super().__len__()

    @staticmethod
    def cumsum(sequence):
        (r, s) = ([], 0)
        for e in sequence:
            r.append((e + s))
            s += e
        return r
