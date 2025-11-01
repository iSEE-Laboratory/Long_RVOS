from datasets.a2d_sentences.a2d_sentences_dataset import A2DSentencesDataset
from datasets.jhmdb_sentences.jhmdb_sentences_dataset import JHMDBSentencesDataset
from datasets.ref_youtube_vos.ref_youtube_vos_dataset import ReferYouTubeVOSDataset
from datasets.davis.davis_dataset import ReferDavisDataset
# from datasets.mevis.mevis_dataset import MeViSDataset
from datasets.mevis.ReferFormer_dataset import MeViSDataset
from datasets.long_rvos.long_rvos_dataset import LongRVOSDataset
from datasets.coco.refercoco import ModulatedDetection
from datasets.concat_dataset import build_joint
import os
import torch
import torchvision
from misc import nested_tensor_from_videos_list


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


class Collator:
    def __call__(self, batch):
        samples, targets = list(zip(*batch))
        samples = nested_tensor_from_videos_list(samples, size_divisibility=32)   # [B, T, C, H, W]
        caption = [t.pop('caption') for t in targets]
        batch_dict = {
            'samples': samples,
            'targets': targets,
            'text_queries': caption,
        }
        if "motions" in targets[0]:
            motions = nested_tensor_from_videos_list([t.pop('motions') for t in targets], size_divisibility=32)
            batch_dict['motions'] = motions
        return batch_dict


def build_dataset(image_set, dataset_file, use_random_sample=None, **kwargs):
    if dataset_file == 'a2d_sentences':
        return A2DSentencesDataset(image_set, **kwargs)
    elif dataset_file == 'jhmdb_sentences':
        return JHMDBSentencesDataset(image_set, **kwargs)
    elif dataset_file == 'ref_youtube_vos':
        return ReferYouTubeVOSDataset(image_set, **kwargs)
    elif dataset_file == 'davis':
        return ReferDavisDataset(image_set, **kwargs)
    elif dataset_file == 'mevis':
        return MeViSDataset(image_set, **kwargs)
    elif dataset_file == 'long_rvos':
        return LongRVOSDataset(image_set, **kwargs)
    #### for pretraining   instances_refcoco_train.json'
    elif dataset_file == 'refcoco' or dataset_file == 'refcoco+' or dataset_file == 'refcocog':
        kwargs['ann_file'] = os.path.join(kwargs['ann_file'], dataset_file, 'instances_{}_{}.json'.format(dataset_file, image_set))
        return ModulatedDetection(image_set, **kwargs)
    elif dataset_file == "joint":
        joint_dataset, collator = build_joint(image_set, **kwargs)
        return joint_dataset, collator
    raise ValueError(f'dataset {dataset_file} not supported')
