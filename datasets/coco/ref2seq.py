
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import torchvision
import misc as utils
from pycocotools import mask as coco_mask
import datasets.transforms as T
from datasets.coco.image_to_seq_augmenter import ImageToSeqAugmenter
import random
import numpy as np
from PIL import Image


class ModulatedDetection(torchvision.datasets.CocoDetection):
    def __init__(self, sub_type, img_folder, ann_file, return_masks=True, return_box=False, window_size=8, **kwargs):
        super(ModulatedDetection, self).__init__(img_folder, ann_file)
        if sub_type == 'val':
            sub_type = 'valid'
        self._transforms = A2dSentencesTransforms(sub_type, **kwargs)
        self.num_frames = window_size
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.collator = Collator(sub_type)
        self.augmenter = ImageToSeqAugmenter(perspective=True, affine=True, motion_blur=True,
                                             rotation_range=(-20, 20), perspective_magnitude=0.08,
                                             hue_saturation_range=(-5, 5), brightness_range=(-40, 40),
                                             motion_blur_prob=0.25, motion_blur_kernel_sizes=(9, 11),
                                             translate_range=(-0.1, 0.1))
    
    def apply_random_sequence_shuffle(self, images, instance_masks):
        perm = list(range(self.num_frames))
        random.shuffle(perm)
        images = [images[i] for i in perm]
        instance_masks = [instance_masks[i] for i in perm]
        return images, instance_masks

    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            img, target = super(ModulatedDetection, self).__getitem__(idx)
            image_id = self.ids[idx]
            coco_img = self.coco.loadImgs(image_id)[0]
            caption = coco_img["caption"]
            dataset_name = coco_img["dataset_name"] if "dataset_name" in coco_img else None
            target = {"image_id": image_id, "annotations": target, "caption": caption}
            img, target = self.prepare(img, target)
            h, w = target['orig_size']
            # image2clip
            seq_images, seq_instance_masks = [img], [target['masks'].numpy()]
            numpy_masks = target['masks'].numpy() # [1, H, W]

            numinst = len(numpy_masks)
            assert numinst == 1
            for t in range(self.num_frames - 1):
                im_trafo, instance_masks_trafo = self.augmenter(np.asarray(img), numpy_masks)
                im_trafo = Image.fromarray(np.uint8(im_trafo))
                seq_images.append(im_trafo) 
                seq_instance_masks.append(np.stack(instance_masks_trafo, axis=0)) #[t 1 h w]
            seq_images, seq_instance_masks = self.apply_random_sequence_shuffle(seq_images, seq_instance_masks)
            targets = []
            for f_i in range(self.num_frames):
                mask = seq_instance_masks[f_i][0] #[H W]
                masks = torch.from_numpy(mask).unsqueeze(0)
                if mask.any() > 0:
                    boxes = utils.masks_to_boxes(masks)
                else:
                    box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                    boxes = box.unsqueeze(0)
                boxes = boxes.clamp(1e-6)
                target_temp = {
                    "masks": masks,
                    "boxes": boxes,
                    "labels": target['labels'],
                    'valid': masks.any(),
                    "referred_instance_idx": torch.tensor(0),
                    "orig_size": target["orig_size"],
                    "size": target["size"],
                    "iscrowd": target["iscrowd"]
                }
                targets.append(target_temp)
            source_frames, targets, text_query = self._transforms(seq_images, targets, target['caption'])
            if torch.any(target['valid'] == 1):  # at leatst one instance
                instance_check = True
            else:
                import random
                idx = random.randint(0, self.__len__() - 1)

        return source_frames, targets, text_query
        # using a2d_setences transformers, and treat coco as video clip


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        caption = target["caption"] if "caption" in target else None

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2] # xminyminwh -> xyxy
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            if len(segmentations) > 1:
                print(image_id)
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        # keep the valid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if caption is not None:
            target["caption"] = caption
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target['referred_instance_idx'] = torch.tensor(0) #only one object each image
        # target["valid"] = torch.tensor([1])
        target['valid'] = torch.tensor(1)
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target

class A2dSentencesTransforms:
    def __init__(self, subset_type, horizontal_flip_augmentations, resize_and_crop_augmentations,
                 random_color, train_short_size, train_max_size, eval_short_size, eval_max_size, **kwargs):
        self.h_flip_augmentation = subset_type == 'train' and horizontal_flip_augmentations
        self.random_color = subset_type == 'train' and random_color
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        scales = [train_short_size]  # no more scales for now due to GPU memory constraints. might be changed later
        # scales = [320]
        self.photometricDistort = T.PhotometricDistort()
        transforms = []
        if resize_and_crop_augmentations:
            if subset_type == 'train':
                transforms.append(T.RandomResize(scales, max_size=train_max_size))
            # elif subset_type == 'test':
            else:
                transforms.append(T.RandomResize([eval_short_size], max_size=eval_max_size)),
        transforms.extend([T.ToTensor(), normalize])
        self.size_transforms = T.Compose(transforms)

    def __call__(self, source_frames, targets, text_query):
        if self.h_flip_augmentation and torch.randn(1) > 0.5:
            source_frames = [F.hflip(f) for f in source_frames]
            for t in targets:
                h, w = t['size']
                t['masks'] = F.hflip(t['masks'])
                boxes = t['boxes'] 
                boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
                t["boxes"] = boxes
            # Note - is it possible for both 'right' and 'left' to appear together in the same query. hence this fix:
            text_query = text_query.replace('left', '@').replace('right', 'left').replace('@', 'right')
        if self.random_color and torch.randn(1) > 0.5:
            source_frames, targets = self.photometricDistort(source_frames, targets)
        source_frames, targets = list(zip(*[self.size_transforms(f, t) for f, t in zip(source_frames, targets)]))
        source_frames = torch.stack(source_frames)  # [T, 3, H, W]
        return source_frames, targets, text_query

class Collator:
    def __init__(self, subset_type):
        self.subset_type = subset_type
    
    def __call__(self, batch):
        samples, targets, text_queries = list(zip(*batch))
        samples = utils.nested_tensor_from_videos_list(samples)  # [T, B, C, H, W] and mask mask for useful area
        # convert targets to a list of tuples. outer list - time steps, inner tuples - time step batch
        targets = list(zip(*targets))
        batch_dict = {
            'samples': samples,
            'targets': targets,
            'text_queries': text_queries
        }
        return batch_dict