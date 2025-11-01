# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import os
import torch
import torch.utils.data
import copy
from torch.utils.data import ConcatDataset
from datasets.coco.ref2seq import ModulatedDetection
from datasets.ref_youtube_vos.ref_youtube_vos_dataset import ReferYouTubeVOSDataset


def build_joint(image_set, **kwargs):
    concat_data = []

    print('preparing coco2seq dataset ....')
    coco_names = ["refcoco", "refcoco+", "refcocog"]
    configs = copy.deepcopy(kwargs)
    for dataset_file in coco_names:
        kwargs["ann_file"] = os.path.join(configs['ann_file'], dataset_file, 'instances_{}_{}.json'.format(dataset_file, image_set))
        coco_seq = ModulatedDetection(image_set, **kwargs)
        concat_data.append(coco_seq)

    print('preparing ytvos dataset  .... ')
    ytvos_dataset = ReferYouTubeVOSDataset(image_set, **kwargs)
    collator = ytvos_dataset.collator
    concat_data.append(ytvos_dataset)

    concat_data = ConcatDataset(concat_data)

    return concat_data, collator