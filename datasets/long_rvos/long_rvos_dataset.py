"""
Ref-YoutubeVOS data loader
"""
from pathlib import Path

import cv2
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
import h5py


class LongRVOSDataset(Dataset):
    """
    mode:
        "vanilla": training on all frames (traditional way);
        "key": training on only key frames;
        "motion": training on a key frame and a motion clip;
    """
    def __init__(self, subset_type, dataset_path, num_frames, sampling_step=0, motion_len=11, ** kwargs):
        if subset_type == 'test':
            subset_type = 'valid'
        self.subset_type = subset_type

        self.dataset_path = dataset_path
        self.img_folder = os.path.join(dataset_path, subset_type)
        self.ann_file = os.path.join(dataset_path, subset_type, 'meta_expressions.json')
        self._transforms = make_coco_transforms(subset_type)
        self.num_frames = num_frames
        self.sampling_step = sampling_step if sampling_step > 0 else num_frames
        self.frame_types = json.load(open(os.path.join(dataset_path, subset_type, 'frame_types.json')))
        self.motion_len = motion_len

        # create video meta data
        self.metas, self.videos = self.prepare_metas()
        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))
        print('\n')

    def prepare_metas(self):
        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']

        videos = list(subset_expressions_by_video.keys())

        metas = []
        for vid in videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)

            # FIlter one I-Frame + 11 P-Frame (for valid motion clip)
            start_indexs = np.where(np.array(self.frame_types[vid]) == 'I')[0]
            end_indexs = start_indexs + 12
            key_frame_ids = start_indexs[:-1][end_indexs[:-1] == start_indexs[1:]]
            key_frame_ids = key_frame_ids.tolist()

            times = len(key_frame_ids) // self.sampling_step

            # for each expression
            for exp_id, exp_dict in vid_data['expressions'].items():
                exp = exp_dict['exp']
                oid = int(exp_dict['obj_id'])
                for k in range(times):
                    meta = {
                        'video': vid,
                        'exp': exp,
                        'frames': vid_frames,
                        'obj_id': oid,
                        'frame_id': key_frame_ids,
                        'category': 0,
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
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]  # dict
            video, exp, obj_id, category, frames, frame_id = \
                meta['video'], meta['exp'], meta['obj_id'], meta['category'], meta['frames'], meta['frame_id']
            exp = " ".join(exp.lower().split())

            vid_len = len(frames)

            # max motion clip length is 12, the first is the key frame
            # the motion shape is T, C, H/16, W/16
            motions = np.load(os.path.join(str(self.img_folder), "motions", f'{video}.npy'))
            motions = torch.from_numpy(motions)

            sample_indx = random.sample(frame_id, self.num_frames)
            sample_indx.sort()
            motion_list = []
            for s in sample_indx:
                motion_clip = motions[s + 1: s + 1 + self.motion_len]  # a clip has up to 11 motion frames
                motion_list.append(motion_clip)
            motions = torch.stack(motion_list, dim=0)
            motions = motions.flatten(0, 1) # T*11, C, H, W

            # read frames and masks
            imgs, labels, boxes, masks, valid = [], [], [], [], []
            for j in range(len(sample_indx)):
                frame_indx = sample_indx[j]
                frame_name = frames[frame_indx]
                img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
                mask_path = os.path.join(str(self.img_folder), 'Annotations', video, str(obj_id), frame_name + '.png')
                img = Image.open(img_path).convert('RGB')
                w, h = img.size

                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert('P')
                    mask = np.array(mask)
                    mask = (mask == 255).astype(np.float32)
                    mask = torch.from_numpy(mask)
                else:
                    mask = torch.zeros((h, w), dtype=torch.float)

                # create the target
                label = torch.tensor(category)
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

            target['motions'] = motions # [T*11, C, H, W]

            imgs, target = self._transforms(imgs, target)
            imgs = torch.stack(imgs, dim=0)

            if target['valid'].any():  # at least one instance
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)
        return imgs, target


def make_coco_transforms(image_set, max_size=640):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # scales = [288, 320, 352, 392, 416, 448, 480, 512]
    scales = [360]  # save memory

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
