"""
Ref-YoutubeVOS data loader
"""
from pathlib import Path

import torch
from torch.autograd.grad_mode import F
from torch.utils.data import Dataset
import datasets.transform_video as T

import os
from glob import glob
from PIL import Image
import json
import numpy as np
import random

from datasets.categories import ytvos_category_dict as category_dict


class ReferYouTubeVOSDataset(Dataset):
    """
    A dataset class for the Refer-Youtube-VOS dataset which was first introduced in the paper:
    "URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark"
    (see https://link.springer.com/content/pdf/10.1007/978-3-030-58555-6_13.pdf).
    The original release of the dataset contained both 'first-frame' and 'full-video' expressions. However, the first
    dataset is not publicly available anymore as now only the harder 'full-video' subset is available to download
    through the Youtube-VOS referring video object segmentation competition page at:
    https://competitions.codalab.org/competitions/29139
    Furthermore, for the competition the subset's original validation set, which consists of 507 videos, was split into
    two competition 'validation' & 'test' subsets, consisting of 202 and 305 videos respectively. Evaluation can
    currently only be done on the competition 'validation' subset using the competition's server, as
    annotations were publicly released only for the 'train' subset of the competition.

    """
    # (self, subset_type: str = 'train', dataset_path: str = 'data/ref_youtube_vos', window_size=12, distributed=False, device=None, ** kwargs):
    # subset_type = 'train', dataset_path = 'data/mevis', num_frames = 16, sampling_frame_range = 8,** kwargs):


    def __init__(self, subset_type, dataset_path, num_frames, ** kwargs):
        if subset_type == 'test':
            subset_type = 'valid'
        self.subset_type = subset_type

        self.dataset_path = dataset_path
        self.img_folder = os.path.join(dataset_path, subset_type)
        self.ann_file = os.path.join(dataset_path, 'meta_expressions', subset_type, 'meta_expressions.json')
        self._transforms = make_coco_transforms(subset_type)
        self.num_frames = num_frames
        # create video meta data
        self.metas, self.videos = self.prepare_metas()
        # self.is_noun = lambda pos: pos[:2] == "NN"
        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))
        print('\n')

    def prepare_metas(self):
        if self.subset_type == 'train':
            # read object information
            with open(os.path.join(str(self.img_folder), 'meta.json'), 'r') as f:
                subset_metas_by_video = json.load(f)['videos']

        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        if self.subset_type == 'valid':
            # for some reason the competition's validation expressions dict contains both the validation & test
            # videos. so we simply load the test expressions dict and use it to filter out the test videos from
            # the validation expressions dict:
            test_expressions_file_path = os.path.join(self.dataset_path, 'meta_expressions', 'test', 'meta_expressions.json')
            with open(test_expressions_file_path, 'r') as f:
                test_expressions_by_video = json.load(f)['videos']
            test_videos = set(test_expressions_by_video.keys())
            valid_plus_test_videos = set(subset_expressions_by_video.keys())
            valid_videos = valid_plus_test_videos - test_videos
            subset_expressions_by_video = {k: subset_expressions_by_video[k] for k in valid_videos}
            assert len(subset_expressions_by_video) == 202, 'error: incorrect number of validation expressions'

        videos = list(subset_expressions_by_video.keys())

        metas = []
        for vid in videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            # for each expression
            for exp_id, exp_dict in vid_data['expressions'].items():
                exp = exp_dict['exp']
                if self.subset_type == 'train':
                    oid = int(exp_dict['obj_id'])
                    category = subset_metas_by_video[vid]['objects'][exp_dict['obj_id']]['category']
                    # for each frame
                    for frame_id in range(0, vid_len, self.num_frames):
                        meta = {
                            'video': vid,
                            'exp': exp,
                            'frames': vid_frames,
                            'obj_id': oid,
                            'frame_id': frame_id,
                            'category': category,
                        }
                        metas.append(meta)
                else:
                    meta = {
                        'video': vid,
                        'exp_id': exp_id,
                        'exp': exp,
                        'frames': vid_frames
                    }
                    metas.append(meta)
        return metas, videos

    @staticmethod
    def bounding_box(img: torch.tensor):
        rows = torch.any(img, dim=1)
        cols = torch.any(img, dim=0)
        y0, y1 = torch.where(rows)[0][[0, -1]]
        x0, x1 = torch.where(cols)[0][[0, -1]]
        return x0, y0, x1, y1

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        if self.subset_type == 'train':
            instance_check = False
            while not instance_check:
                meta = self.metas[idx]  # dict
                video, exp, obj_id, category, frames, frame_id = \
                    meta['video'], meta['exp'], meta['obj_id'], meta['category'], meta['frames'], meta['frame_id']
                exp = " ".join(exp.lower().split())
                category_id = category_dict[category]
                vid_len = len(frames)

                sample_indx = [frame_id]
                if self.num_frames != 1:
                    # local sample [before and after].
                    sample_id_before = random.randint(1, 3)
                    sample_id_after = random.randint(1, 3)
                    local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                    sample_indx.extend(local_indx)

                    # global sampling [in rest frames]
                    if self.num_frames > 3:
                        all_inds = list(range(vid_len))
                        global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                        global_n = self.num_frames - len(sample_indx)
                        if len(global_inds) > global_n:
                            select_id = random.sample(range(len(global_inds)), global_n)
                            for s_id in select_id:
                                sample_indx.append(global_inds[s_id])
                        elif vid_len >= global_n:
                            select_id = random.sample(range(vid_len), global_n)
                            for s_id in select_id:
                                sample_indx.append(all_inds[s_id])
                        else:
                            select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))
                            for s_id in select_id:
                                sample_indx.append(all_inds[s_id])
                sample_indx.sort()
                # random reverse
                if self.subset_type == "train" and np.random.rand() < 0.3:
                    sample_indx = sample_indx[::-1]

                # read frames and masks
                imgs, labels, boxes, masks, valid = [], [], [], [], []
                for j in range(self.num_frames):
                    frame_indx = sample_indx[j]
                    frame_name = frames[frame_indx]
                    img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
                    mask_path = os.path.join(str(self.img_folder), 'Annotations', video, frame_name + '.png')
                    img = Image.open(img_path).convert('RGB')
                    mask = Image.open(mask_path).convert('P')

                    # create the target
                    label = torch.tensor(category_id)
                    mask = np.array(mask)
                    mask = (mask == obj_id).astype(np.float32)
                    mask = torch.from_numpy(mask)
                    if mask.any():
                        box = torch.tensor(self.bounding_box(mask), dtype=torch.float)
                        valid.append(1)
                    else:  # some frame didn't contain the instance
                        box = torch.zeros(4, dtype=torch.float)
                        valid.append(0)

                    # append
                    imgs.append(img)
                    labels.append(label)
                    masks.append(mask)
                    boxes.append(box)

                # transform
                w, h = img.size
                labels = torch.stack(labels, dim=0)
                boxes = torch.stack(boxes, dim=0)
                boxes[:, 0::2].clamp_(min=0, max=w)
                boxes[:, 1::2].clamp_(min=0, max=h)
                masks = torch.stack(masks, dim=0)

                target = {
                    'frames_idx': torch.tensor(sample_indx),  # [T,]
                    'labels': labels,  # [T,] class id of categories
                    'boxes': boxes,  # [T, 4], xyxy
                    'masks': masks,  # [T, H, W]
                    'valid': torch.tensor(valid),  # [T,]  whether include object
                    'caption': exp,
                    'orig_size': torch.as_tensor([int(h), int(w)]),
                    'size': torch.as_tensor([int(h), int(w)])
                }

                imgs, target = self._transforms(imgs, target)
                imgs = torch.stack(imgs, dim=0)

                if target['valid'].any():  # at least one instance
                    instance_check = True
                else:
                    idx = random.randint(0, self.__len__() - 1)
        else:
            meta = self.metas[idx]  # dict
            video, exp_id, exp, frames = meta['video'], meta['exp_id'], meta['exp'], meta['frames']
            exp = " ".join(exp.lower().split())
            img_path = os.path.join(str(self.img_folder), 'JPEGImages', video)
            imgs = [Image.open(os.path.join(img_path, i + '.jpg')).convert('RGB') for i in frames]
            w, h = imgs[0].size
            target = {
                'video_id': video,
                'exp_id': exp_id,
                'frame_indices': frames,
                'caption': exp,
                'orig_size': (int(h), int(w)),
                'size': torch.as_tensor([int(h), int(w)])
            }
            imgs, target = self._transforms(imgs, target)
            imgs = torch.stack(imgs, dim=0)

        return imgs, target


def make_coco_transforms(image_set, max_size=640):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    # CLIP at first to save time
    if image_set == 'train':
        return T.Compose([
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ]),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ])
            ),
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            normalize,
        ])

    # we do not use the 'valid' set since the annotations are inaccessible
    if image_set == 'valid':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')