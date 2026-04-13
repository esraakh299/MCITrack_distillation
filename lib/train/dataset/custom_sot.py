import os
import numpy as np
import torch
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader


class CustomSOT(BaseVideoDataset):
    """Custom SOT dataset with unified folder structure.

    Expected layout:
        root/
        ├── sequences/
        │   ├── video_001/
        │   │   ├── 00001.jpg  (or 0001.jpg, 1.jpg, etc.)
        │   │   ├── 00002.jpg
        │   │   └── ...
        │   └── video_002/
        └── annotations/
            ├── video_001.txt  (x,y,w,h per line)
            └── video_002.txt

    Supports multiple root dirs (pass as list) to combine multiple datasets.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader,
                 multi_modal_vision=False, multi_modal_language=False):
        """
        args:
            root - path (str) or list of paths to dataset root(s)
            image_loader - function to load images
        """
        if isinstance(root, list):
            root_list = root
            super().__init__('CustomSOT', root_list[0], image_loader)
            self.root_list = root_list
        else:
            super().__init__('CustomSOT', root, image_loader)
            self.root_list = [root]

        self.sequence_list = []  # list of (root_path, seq_name) tuples
        self._frame_cache = {}   # cache: seq_id -> sorted list of frame filenames

        for r in self.root_list:
            seq_dir = os.path.join(r, 'sequences')
            anno_dir = os.path.join(r, 'annotations')
            if not os.path.isdir(seq_dir):
                print(f"[CustomSOT] Warning: sequences dir not found: {seq_dir}")
                continue
            for seq_name in sorted(os.listdir(seq_dir)):
                seq_path = os.path.join(seq_dir, seq_name)
                anno_path = os.path.join(anno_dir, seq_name + '.txt')
                if os.path.isdir(seq_path) and os.path.isfile(anno_path):
                    self.sequence_list.append((r, seq_name))

        print(f"[CustomSOT] Loaded {len(self.sequence_list)} sequences from {len(self.root_list)} root(s)")

        self.multi_modal_vision = multi_modal_vision
        self.multi_modal_language = multi_modal_language

    def get_name(self):
        return 'custom_sot'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return False

    def _get_sequence_path(self, seq_id):
        root, seq_name = self.sequence_list[seq_id]
        return os.path.join(root, 'sequences', seq_name)

    def _get_anno_path(self, seq_id):
        root, seq_name = self.sequence_list[seq_id]
        return os.path.join(root, 'annotations', seq_name + '.txt')

    def _get_sorted_frames(self, seq_id):
        """Get sorted list of image filenames for a sequence."""
        if seq_id not in self._frame_cache:
            seq_path = self._get_sequence_path(seq_id)
            img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
            frames = [f for f in os.listdir(seq_path)
                      if os.path.splitext(f)[1].lower() in img_exts]
            frames.sort()
            self._frame_cache[seq_id] = frames
        return self._frame_cache[seq_id]

    def _read_bb_anno(self, seq_id):
        anno_path = self._get_anno_path(seq_id)
        gt = []
        with open(anno_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Support both comma and space/tab delimiters
                line = line.replace(',', ' ')
                vals = line.split()
                if len(vals) >= 4:
                    gt.append([float(v) for v in vals[:4]])
        return torch.tensor(gt, dtype=torch.float32)

    def get_sequence_info(self, seq_id):
        bbox = self._read_bb_anno(seq_id)
        # Valid if w > 0 and h > 0
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        # Assume all frames are visible (no occlusion info)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_id, frame_id):
        seq_path = self._get_sequence_path(seq_id)
        frames = self._get_sorted_frames(seq_id)
        return os.path.join(seq_path, frames[frame_id])

    def _get_frame(self, seq_id, frame_id):
        frame_path = self._get_frame_path(seq_id, frame_id)
        frame = self.image_loader(frame_path)
        if self.multi_modal_vision:
            frame = np.concatenate((frame, frame), axis=-1)
        return frame

    def get_frames(self, seq_id, frame_ids, anno=None):
        frame_list = [self._get_frame(seq_id, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({
            'object_class_name': 'unknown',
            'motion_class': None,
            'major_class': None,
            'root_class': None,
            'motion_adverb': None
        })

        return frame_list, anno_frames, object_meta

    def get_annos(self, seq_id, frame_ids, anno=None):
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return anno_frames
