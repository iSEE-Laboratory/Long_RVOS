"""
Instance Sequence Matching
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from misc import nested_tensor_from_tensor_list, box_cxcywh_to_xyxy, generalized_box_iou

INF = 100000000


def dice_coef(inputs, targets):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1).unsqueeze(1) # [N, 1, THW]
    targets = targets.flatten(1).unsqueeze(0) # [1, M, THW]
    numerator = 2 * (inputs * targets).sum(2)
    denominator = inputs.sum(-1) + targets.sum(-1)

    # NOTE coef doesn't be subtracted to 1 as it is not necessary for computing costs
    coef = (numerator + 1) / (denominator + 1)
    return coef


def sigmoid_focal_coef(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    N, M = len(inputs), len(targets)
    inputs = inputs.flatten(1).unsqueeze(1).expand(-1, M, -1) # [N, M, THW]
    targets = targets.flatten(1).unsqueeze(0).expand(N, -1, -1) # [N, M, THW]

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    coef = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        coef = alpha_t * coef

    return coef.mean(2) # [N, M]


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1,
                 cost_mask: float = 1, cost_dice: float = 1, cost_proj: float = 1, num_classes: int = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
            cost_mask: This is the relative weight of the sigmoid focal loss of the mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_proj = cost_proj
        self.num_classes = num_classes
        self.mask_out_stride = 4
        self.mask = self.cost_mask > 0 or self.cost_dice > 0

    @torch.no_grad()
    def forward(self, outputs, targets):
        src_logits = outputs["pred_logits"]
        src_boxes = outputs["pred_boxes"]
        bs, nf, nq = src_logits.shape[:3]

        if self.mask:
            src_masks = outputs["pred_masks"]
            target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets],
                                                                 size_divisibility=32,
                                                                 split=False).decompose()
            target_masks = target_masks.to(src_masks)  # [B, T, H, W]

            # downsample ground truth masks with ratio mask_out_stride
            start = int(self.mask_out_stride // 2)
            im_h, im_w = target_masks.shape[-2:]

            target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride]
            valid = valid[:, start::self.mask_out_stride, start::self.mask_out_stride]
            assert target_masks.size(2) * self.mask_out_stride == im_h
            assert target_masks.size(3) * self.mask_out_stride == im_w

            src_masks = src_masks.masked_fill(valid[:, None, None], 0)

        indices = []
        for i in range(bs):
            out_prob = src_logits[i].sigmoid()
            out_bbox = src_boxes[i]

            tgt_ids = targets[i]["labels"]
            tgt_bbox = targets[i]["boxes"]
            tgt_valid = targets[i]["valid"]

            # class cost
            # we average the cost on valid frames
            cost_class = []
            for t in range(nf):
                if tgt_valid[t] == 0:
                    continue

                out_prob_split = out_prob[t]
                tgt_ids_split = tgt_ids[t].unsqueeze(0)

                # Compute the classification cost.
                alpha = 0.25
                gamma = 2.0
                neg_cost_class = (1 - alpha) * (out_prob_split ** gamma) * (-(1 - out_prob_split + 1e-8).log())
                pos_cost_class = alpha * ((1 - out_prob_split) ** gamma) * (-(out_prob_split + 1e-8).log())
                if self.num_classes == 1:  # binary referred
                    cost_class_split = pos_cost_class[:, [0]] - neg_cost_class[:, [0]]
                else:
                    cost_class_split = pos_cost_class[:, tgt_ids_split] - neg_cost_class[:, tgt_ids_split]

                cost_class.append(cost_class_split)
            cost_class = torch.stack(cost_class, dim=0).mean(0)  # [q, 1]

            # box cost
            # we average the cost on every frame
            cost_bbox, cost_giou = [], []
            for t in range(nf):
                out_bbox_split = out_bbox[t]
                tgt_bbox_split = tgt_bbox[t].unsqueeze(0)

                # Compute the L1 cost between boxes
                cost_bbox_split = torch.cdist(out_bbox_split, tgt_bbox_split, p=1)

                # Compute the giou cost betwen boxes
                cost_giou_split = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox_split),
                                                       box_cxcywh_to_xyxy(tgt_bbox_split))

                cost_bbox.append(cost_bbox_split)
                cost_giou.append(cost_giou_split)
            cost_bbox = torch.stack(cost_bbox, dim=0).mean(0)
            cost_giou = torch.stack(cost_giou, dim=0).mean(0)

            # Final cost matrix
            C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou

            if self.mask:
                # mask cost
                out_mask = src_masks[i]
                tgt_mask = target_masks[i]
                # Compute the focal loss between masks
                cost_mask = sigmoid_focal_coef(out_mask.transpose(0, 1), tgt_mask.unsqueeze(0))

                # Compute the dice loss between masks
                cost_dice = -dice_coef(out_mask.transpose(0, 1), tgt_mask.unsqueeze(0))

                # Compute the projection loss between masks
                # Boxinst: High-performance instance segmentation with box annotations
                src_y = out_mask.transpose(0, 1).max(dim=2, keepdim=True)[0]
                src_x = out_mask.transpose(0, 1).max(dim=3, keepdim=True)[0]
                tgt_y = tgt_mask.unsqueeze(0).max(dim=2, keepdim=True)[0]
                tgt_x = tgt_mask.unsqueeze(0).max(dim=3, keepdim=True)[0]
                cost_proj = -0.5 * (dice_coef(src_y, tgt_y) + dice_coef(src_x, tgt_x))

                C += self.cost_mask * cost_mask + self.cost_dice * cost_dice + self.cost_proj * cost_proj  # [q, 1]

            # Only has one tgt, MinCost Matcher
            _, src_ind = torch.min(C, dim=0)
            tgt_ind = torch.arange(1).to(src_ind)
            indices.append((src_ind.long(), tgt_ind.long()))

        # list[tuple], length is batch_size
        return indices


def build_matcher(args, num_classes=1):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou,
                            cost_mask=args.set_cost_mask,
                            cost_dice=args.set_cost_dice,
                            cost_proj=args.set_cost_proj,
                            num_classes=num_classes)
