import copy
import torch

from .GroundingDINO import build_groundingdino
from .matcher import build_matcher
from .criterion import SetCriterion
from .postprocessing import build_postprocessors


def build_model(args):
    if args.binary == 1:
        num_classes = 1
    else:
        if args.dataset_name == 'ref_youtube_vos':
            num_classes = 65
        elif args.dataset_name == 'davis':
            num_classes = 78
        elif args.dataset_name == 'a2d_sentences' or args.dataset_name == 'jhmdb_sentences':
            num_classes = 1
        else:
            num_classes = 91  # for coco
    device = torch.device(args.device)

    args.GroundingDINO.single_frame = args.dataset_name in ['coco', 'a2d_sentences', 'jhmdb_sentences']

    model = build_groundingdino(args.GroundingDINO, num_classes=num_classes)
    matcher = build_matcher(args, num_classes=num_classes)

    # prepare weight dict
    losses = ['labels', 'boxes', 'masks']
    weight_dict = {'loss_ce': args.cls_loss_coef,
                   'loss_bbox': args.bbox_loss_coef,
                   'loss_giou': args.giou_loss_coef,
                   "loss_mask": args.mask_loss_coef,
                   "loss_dice": args.dice_loss_coef,
                   "loss_proj": args.proj_loss_coef
                   }

    if args.GroundingDINO.aux_loss:
        aux_weight_dict = {}
        for i in range(args.GroundingDINO.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.GroundingDINO.two_stage_type != 'no':
        interm_weight_dict = {}
        interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items()})
        weight_dict.update(interm_weight_dict)

    criterion = SetCriterion(
            num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=args.eos_coef,
            losses=losses)
    criterion.to(device)

    postprocessors = build_postprocessors(args.dataset_name)

    return model, criterion, postprocessors
