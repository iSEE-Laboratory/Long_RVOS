import json
import torch
import numpy as np
from torch.utils.data import Dataset
from os import path
import datasets.transform_video as T
from PIL import Image
import scipy.io

def get_image_id(video_id, frame_idx):
    image_id = f'v_{video_id}_f_{frame_idx}'
    return image_id

class JHMDBSentencesDataset(Dataset):
    """
    A Torch dataset for JHMDB-Sentences.
    For more information check out: https://kgavrilyuk.github.io/publication/actor_action/ or the original paper at:
    https://arxiv.org/abs/1803.07485data/
    """
    def __init__(self, subset_type: str = 'train', dataset_path: str = 'data/jhmdb_sentences', **kwargs):
        super(JHMDBSentencesDataset, self).__init__()
        assert subset_type in ['train', 'test'], 'error, unsupported dataset subset type. supported: train, test'
        self.subset_type = subset_type
        self.dataset_path = dataset_path
        self.ann_file = path.join(dataset_path, f'jhmdb_sentences_samples_metadata.json')
        self.samples_metadata = self.get_samples_metadata()

        self._transforms = make_coco_transforms(subset_type)
        self.num_frames = 1

        print(f'\n {subset_type} sample num: ', len(self.samples_metadata))
        print('\n')

    def get_samples_metadata(self):
        with open(str(self.ann_file), 'r') as f:
            samples_metadata = [tuple(a) for a in json.load(f)]
            return samples_metadata
    
    @staticmethod
    def bounding_box(img: torch.tensor):
        rows = torch.any(img, dim=1)
        cols = torch.any(img, dim=0)
        y0, y1 = torch.where(rows)[0][[0, -1]]
        x0, x1 = torch.where(cols)[0][[0, -1]]
        return x0, y0, x1, y1

    def __getitem__(self, idx):
        # only support for evaluation
        video_id, chosen_frame_path, video_masks_path, video_total_frames, text_query = self.samples_metadata[idx]
        text_query = " ".join(text_query.lower().split())  # clean up the text query

        # read the source window frames:
        chosen_frame_idx = int(chosen_frame_path.split('/')[-1].split('.')[0])
        # get a window of window_size frames with frame chosen_frame_idx in the middle.
        start_idx, end_idx = chosen_frame_idx - self.num_frames // 2, chosen_frame_idx + (self.num_frames + 1) // 2
        frame_indices = list(range(start_idx, end_idx))  # note that jhmdb.yaml-sentences frames are 1-indexed
        # extract the window source frames:
        sample_indx = []
        for i in frame_indices:
            i = min(max(i, 1), video_total_frames)  # pad out of range indices with edge frames
            sample_indx.append(i)
        sample_indx.sort()
        # find the valid frame index in sampled frame list, there is only one valid frame
        valid_indices = sample_indx.index(chosen_frame_idx)

        # read frames
        imgs, boxes, masks, valid = [], [], [], []
        for i in sample_indx:
            p = '/'.join(chosen_frame_path.split('/')[2:-1]) + f'/{i:05d}.png'
            frame_path = path.join(self.dataset_path, p)
            imgs.append(Image.open(frame_path).convert('RGB'))

        # read the instance masks:
        video_masks_path = '/'.join(video_masks_path.split('/')[2:])
        video_masks_path = path.join(self.dataset_path, video_masks_path)
        all_video_masks = scipy.io.loadmat(video_masks_path)['part_mask'].transpose(2, 0, 1)  # [T, H, W]
        # note that to take the center-frame corresponding mask we switch to 0-indexing:
        mask = torch.tensor(all_video_masks[chosen_frame_idx - 1])  # [H, W]
        if mask.any():
            box = torch.tensor(self.bounding_box(mask), dtype=torch.float)
            valid.append(1)
        else:  # some frame didn't contain the instance
            box = torch.zeros(4, dtype=torch.float)
            valid.append(0)

        boxes.append(box)
        masks.append(mask)

        # transform
        h, w = mask.shape[-2:]
        boxes = torch.stack(boxes, dim=0)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        masks = torch.stack(masks, dim=0)
        # there is only one valid frame
        target = {
            'frames_idx': torch.tensor(sample_indx),  # [T,]
            'valid_indices': torch.tensor([valid_indices]),
            'boxes': boxes,  # [1, 4], xyxy
            'masks': masks,  # [1, H, W]
            'valid': torch.tensor(valid),  # [1,]
            'caption': text_query,
            'orig_size': (int(h), int(w)),
            'size': torch.as_tensor([int(h), int(w)]),
            'image_id': get_image_id(video_id, chosen_frame_idx)
        }

        # "boxes" normalize to [0, 1] and transform from xyxy to cxcywh in self._transform
        imgs, target = self._transforms(imgs, target)
        imgs = torch.stack(imgs, dim=0)  # [T, 3, H, W]

        # in 'val', valid always satisfies
        return imgs, target

    def __len__(self):
        return len(self.samples_metadata)


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

    if image_set == 'test':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    dataset = JHMDBSentencesDataset()
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4,
                            pin_memory=True, shuffle=True)
    for d in tqdm(dataloader):
        pass