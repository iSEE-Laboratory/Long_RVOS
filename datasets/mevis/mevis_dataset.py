import json
import torch
import os
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import datasets.transform_video as T
from pycocotools import mask as coco_mask
import random
import os
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


class MeViSDataset(Dataset):
    """
    A dataset class for the MeViS dataset which was first introduced in the paper:
    "MeViS: A Large-scale Benchmark for Video Segmentation with Motion Expressions"
    """

    def __init__(self, subset_type='train', dataset_path='data/mevis', num_frames=16, sampling_frame_range=8,
                 **kwargs):
        if subset_type == 'test':
            subset_type = 'valid_u'
        self.subset_type = subset_type

        self.img_folder = os.path.join(dataset_path, subset_type)
        self.ann_file = os.path.join(dataset_path, subset_type, 'meta_expressions.json')

        assert num_frames < sampling_frame_range * 2 + 1
        self.num_frames = num_frames
        self.sampling_frame_range = sampling_frame_range
        self.metas, self.videos = self.prepare_metas()
        self._transforms = make_coco_transforms(subset_type)

        if subset_type == 'train':
            self.mask_dict = json.load(open(os.path.join(dataset_path, subset_type, 'mask_dict.json')))
        print('video num: ', len(self.videos), ' clip num: ', len(self.metas))
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
            window_num = max(1, len(vid_frames) // self.sampling_frame_range)
            for exp_id, exp_dict in vid_data['expressions'].items():
                if self.subset_type == 'train':
                    segments = np.array_split(np.array(vid_frames), window_num)
                    segments = [s.tolist() for s in segments]
                    for segment in segments:
                        metas.append({
                            'video_id': vid,
                            'exp_id': exp_id,
                            'exp_dict': exp_dict,
                            'frames': segment,
                        })
                else:
                    metas.append({
                        'video_id': vid,
                        'exp_id': exp_id,
                        'exp_dict': exp_dict,
                        'frames': vid_frames,
                    })
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

                (video_id, exp_id, exp_dict, frame_indices) = (
                    meta['video_id'], meta['exp_id'], meta['exp_dict'], meta['frames'])

                exp = " ".join(exp_dict['exp'].lower().split())

                vid_len = len(frame_indices)
                category_id = 0

                ref_frame = random.randrange(vid_len)
                sample_indx = [ref_frame]
                if self.num_frames != 1:
                    # local sample [before and after].
                    sample_id_before = random.randint(1, 5)
                    sample_id_after = random.randint(1, 5)
                    local_indx = [max(0, ref_frame - sample_id_before), min(vid_len - 1, ref_frame + sample_id_after)]
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
                imgs, labels, boxes, masks, valid, instances = [], [], [], [], [], []
                for j in range(self.num_frames):
                    frame_indx = sample_indx[j]
                    frame_name = frame_indices[frame_indx]
                    img_path = os.path.join(str(self.img_folder), 'JPEGImages', video_id, frame_name + '.jpg')
                    img = Image.open(img_path).convert('RGB')
                    w, h = img.size
                    frm_instances = []

                    # create the target
                    label = torch.tensor(category_id)
                    mask = torch.zeros((h, w), dtype=torch.uint8)
                    for aid, oid in zip(exp_dict['anno_id'], exp_dict['obj_id']):
                        frm_anno = self.mask_dict[str(aid)][int(frame_name)]
                        if frm_anno is not None:
                            instance_mask = torch.from_numpy(coco_mask.decode(frm_anno))
                            instance_box = torch.tensor(self.bounding_box(instance_mask), dtype=torch.float)
                            frm_instances.append({
                                "object_id": oid,
                                "bbox": instance_box,
                                "mask": instance_mask
                            })
                            mask += instance_mask
                    if mask.any():
                        box = torch.tensor(self.bounding_box(mask), dtype=torch.float)
                        valid.append(1)
                    else:
                        box = torch.zeros(4, dtype=torch.float)
                        valid.append(0)

                    imgs.append(img)
                    labels.append(label)
                    masks.append(mask)
                    boxes.append(box)
                    instances.append(frm_instances)

                # transform
                labels = torch.stack(labels, dim=0)
                masks = torch.stack(masks)
                boxes = torch.stack(boxes)
                masks.clamp_(max=1)
                boxes[:, 0::2].clamp_(min=0, max=w)
                boxes[:, 1::2].clamp_(min=0, max=h)

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

        else:   # only support valid_u
            meta = self.metas[idx]  # dict
            (video_id, exp_id, exp_dict, frame_indices) = (
                meta['video_id'], meta['exp_id'], meta['exp_dict'], meta['frames'])
            video_id = video_id.split('_')[0]
            exp = " ".join(exp_dict['exp'].lower().split())
            img_path = os.path.join(str(self.img_folder), 'JPEGImages', video_id)
            imgs = [Image.open(os.path.join(img_path, i + '.jpg')).convert('RGB') for i in frame_indices]
            w, h = imgs[0].size
            target = {
                'video_id': video_id,
                'exp_id': exp_id,
                'frame_indices': frame_indices,
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
    if image_set == 'valid_u':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')