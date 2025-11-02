import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pycocotools.mask as mask_util
from einops import rearrange
from misc import box_cxcywh_to_xyxy


class A2DSentencesPostProcess(nn.Module):
    """
    This module converts the model's output into the format expected by the coco api for the given task
    """
    def __init__(self):
        super(A2DSentencesPostProcess, self).__init__()

    @torch.inference_mode()
    def forward(self, outputs, resized_padded_sample_size, resized_sample_sizes, orig_sample_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            resized_padded_sample_size: size of samples (input to model) after size augmentation + padding.
            resized_sample_sizes: size of samples after size augmentation but without padding.
            orig_sample_sizes: original size of the samples (no augmentations or padding)
        """
        pred_logits = outputs['pred_logits'][:, 0, :, 0]  # [B, T, N, 1] -> [B, N]
        pred_masks = outputs['pred_masks'][:, 0, :, :, :] # [B, T, N, out_h, out_w] -> [B, N, out_h, out_w]

        scores = pred_logits.sigmoid()

        pred_masks = F.interpolate(pred_masks, size=resized_padded_sample_size, mode="bilinear", align_corners=False)
        pred_masks = (pred_masks.sigmoid() > 0.5)
        processed_pred_masks, rle_masks = [], []
        for f_pred_masks, resized_size, orig_size in zip(pred_masks, resized_sample_sizes, orig_sample_sizes):
            f_mask_h, f_mask_w = resized_size  # resized shape without padding
            f_pred_masks_no_pad = f_pred_masks[:, :f_mask_h, :f_mask_w].unsqueeze(1)  # remove the samples' padding
            # resize the samples back to their original dataset (target) size for evaluation
            f_pred_masks_processed = F.interpolate(f_pred_masks_no_pad.float(), size=orig_size, mode="nearest")
            f_pred_rle_masks = [mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                                for mask in f_pred_masks_processed.cpu()]
            processed_pred_masks.append(f_pred_masks_processed)
            # processed_pred_masks.append(f_pred_masks_no_pad)
            rle_masks.append(f_pred_rle_masks)
        predictions = [{'scores': s, 'masks': m, 'rle_masks': rle}
                       for s, m, rle in zip(scores, processed_pred_masks, rle_masks)]
        return predictions


# PostProcess for pretraining
class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        Returns:

        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # coco, num_frames=1
        out_logits = outputs["pred_logits"].flatten(1, 2)
        out_boxes = outputs["pred_boxes"].flatten(1, 2)
        bs, num_queries = out_logits.shape[:2]

        prob = out_logits.sigmoid()  # [bs, num_queries, num_classes]
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), k=num_queries, dim=1, sorted=True)
        scores = topk_values  # [bs, num_queries]
        topk_boxes = topk_indexes // out_logits.shape[2]  # [bs, num_queries]
        labels = topk_indexes % out_logits.shape[2]  # [bs, num_queries]

        boxes = box_cxcywh_to_xyxy(out_boxes)  # [bs, num_queries, 4]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]  # [bs, num_queries, 4]

        assert len(scores) == len(labels) == len(boxes)
        # binary for the pretraining
        results = [{"scores": s, "labels": torch.ones_like(l), "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results


class PostProcessSegm(nn.Module):
    """Similar to PostProcess but for segmentation masks.
    This processor is to be called sequentially after PostProcess.
    Args:
        threshold: threshold that will be applied to binarize the segmentation masks.
    """

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        """Perform the computation
        Parameters:
            results: already pre-processed boxes (output of PostProcess) NOTE here
            outputs: raw outputs of the model
            orig_target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
            max_target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                              after data augmentation.
        """
        assert len(orig_target_sizes) == len(max_target_sizes)

        out_logits = outputs["pred_logits"].flatten(1, 2)
        out_masks = outputs["pred_masks"].flatten(1, 2)
        bs, num_queries = out_logits.shape[:2]

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), k=num_queries, dim=1, sorted=True)
        scores = topk_values # [bs, num_queries]
        topk_boxes = topk_indexes // out_logits.shape[2] # [bs, num_queries]
        labels = topk_indexes % out_logits.shape[2] # [bs, num_queries]

        outputs_masks = [out_m[topk_boxes[i]].unsqueeze(0) for i, out_m, in enumerate(out_masks)] # list[Tensor]
        outputs_masks = torch.cat(outputs_masks, dim=0) # [bs, num_queries, H, W]
        out_h, out_w = outputs_masks.shape[-2:]

        # max_h, max_w = max_target_sizes.max(0)[0].tolist()
        # outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)
        outputs_masks = F.interpolate(outputs_masks, size=(out_h*4, out_w*4), mode="bilinear", align_corners=False)
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1) # [:, 1, h, w]
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()
            results[i]["rle_masks"] = [mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                    for mask in results[i]["masks"].cpu()]

        return results


class COCOPostProcess(nn.Module):
    """
    This module coverts the model's output into coco format but with VOC output
    """
    def __init__(self) -> None:
        super(COCOPostProcess, self).__init__()
    
    @torch.inference_mode()
    def forward(self, outputs, resized_padded_sample_size, resized_sample_sizes, orig_sample_sizes):
        
        pred_cls = outputs["pred_logits"] #[t b nq 1]
        pred_cls = pred_cls.flatten(0, 1) #t*b nq, 1
        
        # prob = F.softmax(pred_cls, dim=-1) #[t*B，Query, 2]
        scores = pred_cls[..., 0].sigmoid()
        # pred_cls = outputs["pred_logits"]
        # prob = F.softmax(pred_cls, dim=-1) #[B*Query, 2]
        # scores = prob[..., 0]
        pred_masks = outputs['pred_masks'] #[t*b q h w]
        pred_masks = F.interpolate(pred_masks, size=resized_padded_sample_size, mode="bilinear", align_corners=False)
        pred_masks = (pred_masks.sigmoid() > 0.5)
        processed_pred_masks, rle_masks = [], []
        for f_pred_masks, resized_size, orig_size in zip(pred_masks, resized_sample_sizes, orig_sample_sizes):
            f_mask_h, f_mask_w = resized_size  # resized shape without padding
            f_pred_masks_no_pad = f_pred_masks[:, :f_mask_h, :f_mask_w].unsqueeze(1)  # remove the samples' padding
            # resize the samples back to their original dataset (target) size for evaluation
            f_pred_masks_processed = F.interpolate(f_pred_masks_no_pad.float(), size=orig_size, mode="nearest")
            f_pred_rle_masks = [mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                                for mask in f_pred_masks_processed.cpu()]
            processed_pred_masks.append(f_pred_masks_processed)
            rle_masks.append(f_pred_rle_masks)
        predictions = [{'scores': s, 'masks': m, 'rle_masks': rle}
                       for s, m, rle in zip(scores, processed_pred_masks, rle_masks)]
        return predictions


class ReferYoutubeVOSPostProcess(nn.Module):
    """
    This module converts the model's output into the format expected by the coco api for the given task
    """
    def __init__(self):
        super(ReferYoutubeVOSPostProcess, self).__init__()

    @torch.inference_mode()
    def forward(self, outputs, videos_metadata, samples_shape_with_padding):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            videos_metadata: a dictionary with each video's metadata.
            samples_shape_with_padding: size of the batch frames with padding.
        """
        pred_logits = outputs['pred_logits'].sigmoid()  # [B, T, N, k]
        prob_is_referred = pred_logits.mean(1)  # [B, N, k]
        max_scores, _ = prob_is_referred.max(-1)  # [B, N]
        pred_trajectory_indices = torch.argmax(max_scores, dim=-1)  # [B]
        pred_masks = outputs['pred_masks']

        # keep only the masks of the chosen trajectories:
        b = pred_masks.shape[0]
        pred_masks = pred_masks[torch.arange(b), :, pred_trajectory_indices]
        # resize the predicted masks to the size of the model input (which might include padding)
        pred_masks = F.interpolate(pred_masks, size=samples_shape_with_padding, mode="bilinear", align_corners=False)
        # apply a threshold to create binary masks:
        pred_masks = (pred_masks.sigmoid() > 0.5)
        # remove the padding per video (as videos might have different resolutions and thus different padding):
        preds_by_video = []
        for video_pred_masks, video_metadata in zip(pred_masks, videos_metadata):
            # size of the model input batch frames without padding:
            resized_h, resized_w = video_metadata['size']
            video_pred_masks = video_pred_masks[:, :resized_h, :resized_w].unsqueeze(1)  # remove the padding
            # resize the masks back to their original frames dataset size for evaluation:
            original_frames_size = video_metadata['orig_size']
            video_pred_masks = F.interpolate(video_pred_masks.float(), size=original_frames_size, mode="nearest")
            video_pred_masks = video_pred_masks.to(torch.uint8).cpu()
            # combine the predicted masks and the video metadata to create a final predictions dict:
            video_pred = {**video_metadata, **{'pred_masks': video_pred_masks}}
            preds_by_video.append(video_pred)
        return preds_by_video


def build_postprocessors(dataset_name):
    if dataset_name == 'a2d_sentences' or dataset_name == 'jhmdb_sentences':
        postprocessors = A2DSentencesPostProcess()
    elif "coco" in dataset_name:
        postprocessors = {"bbox": PostProcess(),
                          "segm": PostProcessSegm(threshold=0.5)
                          }
    elif dataset_name == 'ref_youtube_vos' or dataset_name == 'joint':
        postprocessors = ReferYoutubeVOSPostProcess()
        # for coco pretrain postprocessor
    else:
        postprocessors = None
    return postprocessors
