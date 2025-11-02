# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
import torchvision

from models.dino_util import get_tokenlizer
from models.dino_util.misc import (
    NestedTensor,
    inverse_sigmoid,
    nested_tensor_from_tensor_list,
)

from .backbone import build_backbone
from .bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .transformer import build_transformer, DeformableTransformerDecoderLayer
from .utils import MLP, ContrastiveEmbed, _set_lora_linear
from models.segmentation import FPNSpatialDecoder
from ..dino_util.misc import clean_state_dict
from .utils import gen_sineembed_for_position, _get_clones
from einops import rearrange, repeat
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from .temporal_modules import TemporalDecoder

class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
            self,
            backbone,
            transformer,
            num_queries,
            aux_loss=False,
            iter_update=False,
            query_dim=2,
            num_feature_levels=1,
            nheads=8,
            # two stage
            two_stage_type="no",  # ['no', 'standard']
            dec_pred_bbox_embed_share=True,
            two_stage_class_embed_share=True,
            two_stage_bbox_embed_share=True,
            num_patterns=0,
            dn_number=100,
            dn_box_noise_scale=0.4,
            dn_label_noise_ratio=0.5,
            dn_labelbook_size=100,
            text_encoder_type="bert-base-uncased",
            sub_sentence_present=True,
            max_text_len=256,
            num_classes=1,
            dropout=0.0,
            # lora
            dec_lora=False,
            enc_lora=False,
            lora_rank=16,
            # segmentation
            pixel_decoder=None,
            mask_decoder=None,
            temporal_layer=3,
            tracking_alpha=0.1,
            full_tune=False,
            trainable_key_list=None,
            lora_exclued_list=None
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer__.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = 256
        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)

        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        # freeze

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        # prepare input projection layers
        if num_feature_levels > 1:
            # num_backbone_outs = len(backbone.num_channels)
            num_backbone_outs = len(backbone.num_channels[-3:])  # Not use the first-layer feature
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[-3:][_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-3:][0], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed()

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self.ref_point_head = self.transformer.decoder.ref_point_head
        self._reset_parameters()

        if enc_lora or dec_lora:
            if enc_lora:
                _set_lora_linear(self.transformer.encoder, lora_rank, lora_exclued_list)
            if dec_lora:
                _set_lora_linear(self.transformer.decoder, lora_rank, lora_exclued_list)

        self.pixel_decoder = pixel_decoder
        self.mask_decoder = mask_decoder

        # temporal aggregation
        _temporal_decoder = TemporalDecoder(hidden_dim, nheads, temporal_layer, dropout)
        temporal_decoder_list = [_temporal_decoder for i in range(transformer.num_decoder_layers)]
        self.temporal_decoder = nn.ModuleList(temporal_decoder_list)
        self.tracking_alpha = tracking_alpha

        # freeze untrained parameters
        trainable_key_list = trainable_key_list or []
        if not full_tune:
            for n, p in self.named_parameters():
                if any(keyword in n for keyword in trainable_key_list):
                    continue
                p.requires_grad_(False)

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    def forward(self, samples: NestedTensor, captions: List[str], targets=None, motions: NestedTensor=None, **kw):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        samples = copy.deepcopy(samples)
        B, T, H, W = samples.mask.shape
        device = samples.device
        samples.tensors = rearrange(samples.tensors, 'b t c h w -> (b t) c h w')
        samples.mask = rearrange(samples.mask, 'b t h w -> (b t) h w')
        captions = [t if t.endswith(".") else t + "." for t in captions]

        # encoder texts

        tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(
            device
        )
        one_hot_token = tokenized

        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                                        :, : self.max_text_len, : self.max_text_len
                                        ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

        encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195
        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                                        :, : self.max_text_len, : self.max_text_len
                                        ]

        text_dict = {
            "encoded_text": repeat(encoded_text, 'b ... -> (b t) ...', t=T),  # (b*t, l, d)
            "text_token_mask": repeat(text_token_mask, 'b ... -> (b t) ...', t=T),  # (b*t, l)
            "position_ids": repeat(position_ids, 'b ... -> (b t) ...', t=T),  # (b*t, l)
            "text_self_attention_masks": repeat(text_self_attention_masks, 'b ... -> (b t) ...', t=T),  # (b*t, l, l)
            "embed_text": encoded_text,  # preserve the unfused encoded_text
        }

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)
        poss = poss[-3:]

        srcs = []
        masks = []
        for l, feat in enumerate(features[-3:]):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        # encode motions in 1/8 size
        motions.tensors = rearrange(motions.tensors, "b (t n) ... -> (b t) n ...", t=T)
        motions.mask = rearrange(motions.mask, "b (t n) ... -> (b t) n ...", t=T)
        motion_dict = self.prepare_motion(motions, poss[0], size=poss[0].shape[-2:])

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal, memory_enc, selected_query_list = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict, motion_dict
        )

        # deformable-detr-like anchor update
        outputs_coord_list = []
        outputs_mask_list = []
        outputs_class_list = []

        memory_enc.insert(0, features[0].tensors)
        mask_features = self.pixel_decoder(memory_enc[-1], memory_enc[:-1][::-1])
        mask_features_flat = mask_features.flatten(2).permute(2, 0, 1)
        mask_features_padding = features[0].mask.flatten(1)
        start_index = torch.tensor(0, device=device)
        spatial_shapes = torch.as_tensor(mask_features.shape[-2:], device=device)[None, :]
        valid_ratios = self.transformer.get_valid_ratio(features[0].mask)
        valid_ratios = torch.cat([valid_ratios, valid_ratios], -1)[:, None, :]

        if T > 1:
            # sort the queries via tracking
            text_feat = rearrange(text_dict['encoded_text'], '(b t) l d -> t b l d', t=T)
            sent_feat = text_feat[:, :, 0]
            hs_tracked = []
            ref_tracked = []
            alpha = self.tracking_alpha
            for l in range(len(hs)):
                layer_hs = rearrange(hs[l], '(b t) q d -> t b q d', t=T)
                layer_ref = rearrange(reference[l], '(b t) q d -> t b q d', t=T)
                layer_hs_tracked = [layer_hs[0]]
                layer_ref_tracked = [layer_ref[0]]
                hs_memory = layer_hs[0]
                for t in range(1, T):
                    ind = self.match_from_embds(hs_memory, layer_hs[t])
                    ind = torch.tensor(ind, device=device)
                    hs_sorted = torch.gather(layer_hs[t], 1, ind[:, :, None].expand(-1, -1, self.hidden_dim))
                    ref_sorted = torch.gather(layer_ref[t], 1, ind[:, :, None].expand(-1, -1, 4))
                    conf = torch.matmul(hs_sorted, text_feat[t].transpose(1, 2)).max(-1)[0]
                    conf = (conf - conf.min(1)[0][:, None]) / (conf.max(1)[0] - conf.min(1)[0])[:, None]
                    conf = conf.unsqueeze(-1)
                    # conf = 1
                    hs_memory = (1 - alpha * conf) * hs_memory + alpha * conf * hs_sorted
                    layer_hs_tracked.append(hs_sorted)
                    layer_ref_tracked.append(ref_sorted)
                layer_hs_tracked = torch.stack(layer_hs_tracked, 0)
                layer_ref_tracked = torch.stack(layer_ref_tracked, 0)
                hs_tracked.append(rearrange(layer_hs_tracked, 't b q d -> (b t) q d'))
                ref_tracked.append(rearrange(layer_ref_tracked, 't b q d -> (b t) q d'))
            hs = hs_tracked
            reference[:-1] = ref_tracked


        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_cls_embed, layer_temp, layer_hs) in enumerate(
                zip(reference[:-1], self.bbox_embed, self.class_embed, self.temporal_decoder, hs)
        ):
            if T > 1:
                layer_sent_feat = repeat(sent_feat, 't b d -> t (b q) d', q=layer_hs.shape[1])
                layer_hs = rearrange(layer_hs, '(b t) q d -> t (b q) d', b=B)
                layer_hs = layer_temp(layer_hs, layer_sent_feat)  # t, b*q, d
                layer_hs = rearrange(layer_hs, 't (b q) d -> (b t) q d', b=B)

            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_coord_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_coord = layer_coord_unsig.sigmoid()
            outputs_coord_list.append(layer_coord)

            # prepare input for segmentation head
            query_ref_input = layer_ref_sig * valid_ratios  # use proposal
            query_sine_embed = gen_sineembed_for_position(query_ref_input)
            query_pos = self.ref_point_head(query_sine_embed).transpose(0, 1)  # nq, bs, 256
            query_ref_input = query_ref_input.transpose(0, 1)[:, :, None, :]
            mask_embed = layer_hs.transpose(0, 1)
            # begin segmentation
            for mask_layer in self.mask_decoder:
                mask_embed = mask_layer(
                    tgt=mask_embed,
                    tgt_query_pos=query_pos,
                    tgt_reference_points=query_ref_input,
                    memory_text=text_dict["encoded_text"],
                    text_attention_mask=~text_dict["text_token_mask"],
                    memory=mask_features_flat,
                    memory_key_padding_mask=mask_features_padding,
                    memory_level_start_index=start_index,
                    memory_spatial_shapes=spatial_shapes,
                )
            mask_embed = mask_embed.transpose(0, 1)
            layer_outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            outputs_mask_list.append(layer_outputs_mask)

            # classification
            layer_class = torch.max(layer_cls_embed(layer_hs, text_dict), dim=-1, keepdim=True)[0]
            outputs_class_list.append(layer_class)

        for l in range(len(hs)):
            outputs_class_list[l] = rearrange(outputs_class_list[l], '(b t) q k -> b t q k', b=B)
            outputs_coord_list[l] = rearrange(outputs_coord_list[l], '(b t) q c -> b t q c', b=B)

        out = {"pred_logits": outputs_class_list[-1], "pred_boxes": outputs_coord_list[-1]}

        for l in range(len(hs)):
            outputs_mask_list[l] = rearrange(outputs_mask_list[l], '(b t) q h w -> b t q h w', b=B)
        out.update({"pred_masks": outputs_mask_list[-1]})

        # for intermediate outputs
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class_list, outputs_coord_list, outputs_mask_list)

        # # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict).max(-1, keepdim=True)[0]
            interm_class = rearrange(interm_class, '(b t) q k -> b t q k', b=B)
            interm_coord = rearrange(ref_enc[-1], '(b t) q k -> b t q k', b=B)
            interm_proposal = rearrange(init_box_proposal, '(b t) q c -> b t q c', b=B)
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
            out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': interm_proposal}

            # prepare input for segmentation head
            query_ref_input = init_box_proposal * valid_ratios  # use proposal
            query_sine_embed = gen_sineembed_for_position(query_ref_input)
            query_pos = self.ref_point_head(query_sine_embed).transpose(0, 1)  # nq, bs, 256
            query_ref_input = query_ref_input.transpose(0, 1)[:, :, None, :]
            mask_embed = hs_enc[-1].transpose(0, 1)
            # begin segmentation
            for mask_layer in self.mask_decoder:
                mask_embed = mask_layer(
                    tgt=mask_embed,
                    tgt_query_pos=query_pos,
                    tgt_reference_points=query_ref_input,
                    memory_text=text_dict["encoded_text"],
                    text_attention_mask=~text_dict["text_token_mask"],
                    memory=mask_features_flat,
                    memory_key_padding_mask=mask_features_padding,
                    memory_level_start_index=start_index,
                    memory_spatial_shapes=spatial_shapes,
                )
            mask_embed = mask_embed.transpose(0, 1)
            interm_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            interm_mask = rearrange(interm_mask, '(b t) q h w -> b t q h w', b=B)
            out['interm_outputs'].update({"pred_masks": interm_mask})
            out['interm_outputs_for_matching_pre'].update({"pred_masks": interm_mask})

        # outputs['pred_logits'].shape
        # torch.Size([4, 900, 256])

        # outputs['pred_boxes'].shape
        # torch.Size([4, 900, 4])

        # outputs['text_mask'].shape
        # torch.Size([256])

        # outputs['text_mask']

        # outputs['aux_outputs'][0].keys()
        # dict_keys(['pred_logits', 'pred_boxes', 'one_hot', 'text_mask'])

        # outputs['aux_outputs'][img_idx]

        # outputs['token']
        # <class 'transformers.tokenization_utils_base.BatchEncoding'>

        # outputs['interm_outputs'].keys()
        # dict_keys(['pred_logits', 'pred_boxes', 'one_hot', 'text_mask'])

        # outputs['interm_outputs_for_matching_pre'].keys()
        # dict_keys(['pred_logits', 'pred_boxes'])

        # outputs['one_hot'].shape
        # torch.Size([4, 900, 256])
        return out

    def infer(self, all_samples: NestedTensor, captions, all_motions, targets, crop_len=36, **kw):
        all_samples = copy.deepcopy(all_samples)

        if isinstance(all_samples, (list, torch.Tensor)):
            all_samples = nested_tensor_from_tensor_list(all_samples)

        B, T, H, W = all_samples.mask.shape
        assert B == 1, "Inference only supports for batch size = 1"
        device = all_samples.device
        nl = self.transformer.num_decoder_layers - 1

        captions = [t if t.endswith(".") else t + "." for t in captions]

        # encoder texts

        tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(
            device
        )

        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                                        :, : self.max_text_len, : self.max_text_len
                                        ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

        encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195
        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                                        :, : self.max_text_len, : self.max_text_len
                                        ]

        all_features, all_hs, all_reference, all_memory_enc = [], [], [], []
        all_text_dict = {
            "encoded_text": [],
            "text_token_mask": [],
            "position_ids": [],
            "text_self_attention_masks": []
        }
        for clip_id in range(0, T, crop_len):
            samples = NestedTensor(
                all_samples.tensors[:, clip_id: clip_id + crop_len],
                all_samples.mask[:, clip_id: clip_id + crop_len]
            )

            samples.tensors = rearrange(samples.tensors, 'b t c h w -> (b t) c h w')
            samples.mask = rearrange(samples.mask, 'b t h w -> (b t) h w')

            t = samples.tensors.shape[0]
            text_dict = {
                "encoded_text": repeat(encoded_text, 'b ... -> (b t) ...', t=t),  # (b*t, l, d)
                "text_token_mask": repeat(text_token_mask, 'b ... -> (b t) ...', t=t),  # (b*t, l)
                "position_ids": repeat(position_ids, 'b ... -> (b t) ...', t=t),  # (b*t, l)
                "text_self_attention_masks": repeat(text_self_attention_masks, 'b ... -> (b t) ...', t=t),  # (b*t, l, l)
            }

            features, poss = self.backbone(samples)
            poss = poss[-3:]


            srcs = []
            masks = []
            for l, feat in enumerate(features[-3:]):
                src, mask = feat.decompose()
                srcs.append(self.input_proj[l](src))
                masks.append(mask)
                assert mask is not None
            if self.num_feature_levels > len(srcs):
                _len_srcs = len(srcs)
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:
                        src = self.input_proj[l](features[-1].tensors)
                    else:
                        src = self.input_proj[l](srcs[-1])
                    m = samples.mask
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src)
                    masks.append(mask)
                    poss.append(pos_l)

            # encode motions in 1/8 size
            motion = NestedTensor(
                all_motions.tensors[clip_id: clip_id + crop_len],
                all_motions.mask[clip_id: clip_id + crop_len]
            )
            motion_dict = self.prepare_motion(motion, poss[0], size=poss[0].shape[-2:])

            input_query_bbox = input_query_label = attn_mask = dn_meta = None
            output = self.transformer(
                srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict, motion_dict
            )
            hs, reference, hs_enc, ref_enc, init_box_proposal, memory_enc, selected_query_list = output
            all_hs.append(hs[nl])
            all_reference.append(reference[nl])
            all_memory_enc.append(memory_enc)
            all_features.append(features[0])
            for k, v in text_dict.items():
                all_text_dict[k].append(v)

        # merge into video features
        hs = torch.cat(all_hs, dim=0)   # (video_len, q, d)
        reference = torch.cat(all_reference, dim=0)   # (video_len, q, d)
        memory_enc = [torch.cat(mem, dim=0) for mem in list(zip(*all_memory_enc))]
        features = NestedTensor(*[torch.cat(x, dim=0) for x in list(zip(*[feat.decompose() for feat in all_features]))])
        text_dict = {k: torch.cat(v, dim=0) for k, v in all_text_dict.items()}

        # sort the queries via tracking
        text_feat = rearrange(text_dict['encoded_text'], '(b t) l d -> t b l d', t=T)
        sent_feat = repeat(text_feat[:, :, 0], 't b d -> t (b q) d', q=hs.shape[1])
        alpha = self.tracking_alpha
        hs = rearrange(hs, '(b t) q d -> t b q d', t=T)
        reference = rearrange(reference, '(b t) q d -> t b q d', t=T)
        hs_tracked = [hs[0]]
        ref_tracked = [reference[0]]
        hs_memory = hs[0]
        for t in range(1, T):
            ind = self.match_from_embds(hs_memory, hs[t])
            ind = torch.tensor(ind, device=device)
            hs_sorted = torch.gather(hs[t], 1, ind[:, :, None].expand(-1, -1, self.hidden_dim))
            ref_sorted = torch.gather(reference[t], 1, ind[:, :, None].expand(-1, -1, 4))
            conf = torch.matmul(hs_sorted, text_feat[t].transpose(1, 2)).max(-1)[0]
            conf = (conf - conf.min(1)[0][:, None]) / (conf.max(1)[0] - conf.min(1)[0])[:, None]
            conf = conf.unsqueeze(-1)
            # conf = 1
            hs_memory = (1 - alpha * conf) * hs_memory + alpha * conf * hs_sorted
            hs_tracked.append(hs_sorted)
            ref_tracked.append(ref_sorted)
        hs = rearrange(torch.stack(hs_tracked, 0), 't b q d -> t (b q) d')
        reference = rearrange(torch.stack(ref_tracked, 0), 't b q d -> (b t) q d')

        # temporal fusion
        hs = self.temporal_decoder[nl](hs, sent_feat)  # t, b*q, d
        hs = rearrange(hs, 't (b q) d -> (b t) q d', b=B)

        # logits
        logits = torch.max(self.class_embed[nl](hs, text_dict), dim=-1, keepdim=True)[0]

        # bbox
        delta_unsig = self.bbox_embed[nl](hs)
        coord_unsig = delta_unsig + inverse_sigmoid(reference)
        coord = coord_unsig.sigmoid()

        # segmentation
        memory_enc.insert(0, features.tensors)
        mask_features = self.pixel_decoder(memory_enc[-1], memory_enc[:-1][::-1])
        mask_features_flat = mask_features.flatten(2).permute(2, 0, 1)

        mask_features_padding = features.mask.flatten(1)
        start_index = torch.tensor(0, device=device)
        spatial_shapes = torch.as_tensor(mask_features.shape[-2:], device=device)[None, :]
        valid_ratios = self.transformer.get_valid_ratio(features.mask)
        valid_ratios = torch.cat([valid_ratios, valid_ratios], -1)[:, None, :]

        query_ref_input = reference * valid_ratios  # use proposal
        query_sine_embed = gen_sineembed_for_position(query_ref_input)
        query_pos = self.ref_point_head(query_sine_embed).transpose(0, 1)  # nq, bs, 256
        query_ref_input = query_ref_input.transpose(0, 1)[:, :, None, :]
        mask_embed = hs.transpose(0, 1)

        # we perform with clips since deformable attention only support max length of 36
        mask_embed_list = []
        for clip_id in range(0, T, crop_len):
            clip_mask_embed = mask_embed[:, clip_id: clip_id+crop_len]
            for mask_layer in self.mask_decoder:
                clip_mask_embed = mask_layer(
                    tgt=clip_mask_embed,
                    tgt_query_pos=query_pos[:, clip_id: clip_id+crop_len],
                    tgt_reference_points=query_ref_input[:, clip_id: clip_id+crop_len],
                    memory_text=text_dict["encoded_text"][clip_id: clip_id+crop_len],
                    text_attention_mask=~text_dict["text_token_mask"][clip_id: clip_id+crop_len],
                    memory=mask_features_flat[:, clip_id: clip_id+crop_len],
                    memory_key_padding_mask=mask_features_padding[clip_id: clip_id+crop_len],
                    memory_level_start_index=start_index,
                    memory_spatial_shapes=spatial_shapes,
                )
            mask_embed_list.append(clip_mask_embed.transpose(0, 1))
        mask_embed = torch.cat(mask_embed_list, dim=0)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        out = {"pred_logits": rearrange(logits, '(b t) q k -> b t q k', b=B),
               "pred_boxes": rearrange(coord, '(b t) q k -> b t q k', b=B),
               "pred_masks": rearrange(outputs_mask, '(b t) q h w -> b t q h w', b=B)
               }

        return out

    def prepare_motion(self, motions: NestedTensor, pos_embeds, size) -> dict:
        # motion : bs, t, 2, h, w
        # pos_embeds: bs, d, h, w
        x, mask = motions.decompose()
        bs, t, dim, h, w = x.shape
        dtype = x.dtype
        device = x.device
        assert dim == 2

        x = x.reshape(bs * t, dim, h, w)
        x = F.interpolate(x.float(), size, mode='bicubic').to(dtype)
        x = x.reshape(bs, t, *x.shape[1:])
        mask = F.interpolate(mask.float(), size, mode='nearest').bool()

        spatial_shapes = torch.as_tensor([size], dtype=torch.long, device=device)
        valid_ratios = self.transformer.get_valid_ratio(mask.flatten(0, 1)).unsqueeze(1)
        reference_points = self.transformer.encoder.get_reference_points(spatial_shapes, valid_ratios, device)

        motion_dict = {"motion": x, "mask": mask, "pos": pos_embeds, "spatial_shapes": spatial_shapes,
                       "reference_points": reference_points}
        return motion_dict

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_mask):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b, "pred_masks": c}
            for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_mask[:-1])
        ]

    @torch.no_grad()
    def match_from_embds(self, mem, cur):
        # mem, cur: b, q, d
        mem = F.normalize(mem, dim=-1)
        cur = F.normalize(cur, dim=-1)
        cos_sim = torch.matmul(mem, cur.transpose(1, 2))

        cost_embd = 1 - cos_sim

        C = 1.0 * cost_embd
        C = C.cpu()  # memory x current

        # permutation that makes current aligns to memory
        indices = np.stack([linear_sum_assignment(c)[1] for c in C], axis=0)

        return indices


def build_groundingdino(args, num_classes):
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    hidden_dim = transformer.d_model

    pixel_decoder = FPNSpatialDecoder(hidden_dim, 2 * [hidden_dim] + [backbone.num_channels[0]], args.SegHead.mask_dim)
    mask_dec_layer = DeformableTransformerDecoderLayer(
        hidden_dim,
        n_levels=1,
        n_heads=args.SegHead.n_heads,
        n_points=args.SegHead.n_points,
        dropout=args.SegHead.dropout,
        use_text_cross_attention=args.SegHead.use_text_cross_attention,
    )
    if not args.SegHead.use_self_attention:
        mask_dec_layer.rm_self_attn_modules()
    mask_decoder = _get_clones(mask_dec_layer, args.SegHead.n_layers)


    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    trainable_key_list = args.trainable_keys

    lora_exclued_list = args.lora_exclued_keys

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
        num_classes=num_classes,
        dec_lora=args.dec_lora,
        enc_lora=args.enc_lora,
        lora_rank=args.lora_rank,
        pixel_decoder=pixel_decoder,
        mask_decoder=mask_decoder,
        trainable_key_list=trainable_key_list,
        full_tune=args.full_tune,
        temporal_layer=args.temporal_layer,
        tracking_alpha=args.tracking_alpha,
        dropout=args.dropout,
        lora_exclued_list=lora_exclued_list,
    )
    print("load pretrained GroundingDINO from {} ...".format(args.pretrained_path))
    checkpoint = torch.load(args.pretrained_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    return model
