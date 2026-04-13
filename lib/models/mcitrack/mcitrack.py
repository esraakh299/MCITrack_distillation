"""
MCITrack Model
"""
import torch
import math
from torch import nn
import torch.nn.functional as F
from lib.models.mcitrack.encoder import build_encoder
from .decoder import build_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed
from .neck import build_neck
from collections import OrderedDict

class MCITrack(nn.Module):
    """ This is the base class for MCITrack """
    def __init__(self, encoder, decoder, neck,cfg,
                 num_frames=1, num_template=1, decoder_type="CENTER"):
        """ Initializes the model.
        Parameters:
            encoder: torch module of the encoder to be used. See encoder.py
            decoder: torch module of the decoder architecture. See decoder.py
        """
        super().__init__()
        self.encoder = encoder
        self.decoder_type = decoder_type
        self.neck = neck

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.decoder = decoder

        self.num_frames = num_frames
        self.num_template = num_template
        self.freeze_en = cfg.TRAIN.FREEZE_ENCODER
        self.interaction_indexes = cfg.MODEL.ENCODER.INTERACTION_INDEXES


    def forward(self, template_list=None, search_list=None, template_anno_list=None,enc_opt=None,neck_h_state=None, feature=None, mode="encoder"):
        """
        image_list: list of template and search images, template images should precede search images
        xz: feature from encoder
        seq: input sequence of the decoder
        mode: encoder or decoder.
        """
        if mode == "encoder":
            return self.forward_encoder(template_list, search_list, template_anno_list)
        elif mode == "neck":
            return self.forward_neck(enc_opt,neck_h_state)
        elif mode == "decoder":
            return self.forward_decoder(feature)
        else:
            raise ValueError

    def forward_encoder(self, template_list, search_list, template_anno_list):
        # Forward the encoder
        xz = self.encoder(template_list, search_list, template_anno_list)
        return xz
    def forward_neck(self,enc_out,neck_h_state):
        x = enc_out
        xs = x[:, 0:self.num_patch_x]
        x,xs,h = self.neck(x,xs,neck_h_state,self.encoder.body.blocks,self.interaction_indexes)
        x = self.encoder.body.fc_norm(x)
        xs = xs + x[:, 0:self.num_patch_x]
        return x,xs,h

    def forward_decoder(self, feature, gt_score_map=None):
        # feature = feature[0]
        # feature = feature[:,0:self.num_patch_x * self.num_frames] # (B, HW, C)
        bs, HW, C = feature.size()
        if self.decoder_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
        if self.decoder_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.decoder(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.decoder_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.decoder_type == "MLP":
            # run the mlp head
            score_map, bbox, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

def build_mcitrack(cfg):
    encoder = build_encoder(cfg)
    neck = build_neck(cfg,encoder)
    decoder = build_decoder(cfg, neck)
    model = MCITrack(
        encoder,
        decoder,
        neck,
        cfg,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
    )
    return model


class ADAPTIVE_NET(nn.Module):
    """Adaptive network for Target-aware Adaptive Distillation (TAD).
    Decides per-sample whether to apply distillation based on difficulty.
    Takes concatenated teacher+student features and outputs binary logits
    for Gumbel-Softmax selection.
    """
    def __init__(self, teacher_enc_chan, student_enc_chan, num_layer, cfg):
        super().__init__()
        self.num_layer = num_layer
        self.num_search = int((cfg.DATA.SEARCH.SIZE // cfg.MODEL.ENCODER.STRIDE) ** 2)
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(teacher_enc_chan + student_enc_chan, student_enc_chan),
                nn.ReLU(),
                nn.Linear(student_enc_chan, 2)  # output: whether to select this sample
            ) for _ in range(num_layer)
        ])

    def forward(self, feat_t_list, feat_s_list):
        B, _, CS = feat_s_list[0].shape
        B, _, CT = feat_t_list[0].shape
        logits_list = []
        for i in range(self.num_layer):
            ft = feat_t_list[i][:, 0:self.num_search].permute(0, 2, 1).view(
                B, CT, int(self.num_search ** 0.5), int(self.num_search ** 0.5))
            fs = feat_s_list[i][:, 0:self.num_search].permute(0, 2, 1).view(
                B, CS, int(self.num_search ** 0.5), int(self.num_search ** 0.5))
            if ft.dim() == 4:
                ft = F.adaptive_avg_pool2d(ft, 1).flatten(1)
            if fs.dim() == 4:
                fs = F.adaptive_avg_pool2d(fs, 1).flatten(1)
            feat_cat = torch.cat([ft, fs], dim=1)
            logits = self.mlps[i](feat_cat)
            logits_list.append(logits)
        return logits_list


def build_mcitrack_distill(cfg):
    """Build teacher, student, and adaptive net for TAD distillation.

    Teacher: uses cfg.TRAIN.TEACHER_TYPE encoder, frozen weights from checkpoint.
    Student: uses cfg.MODEL.ENCODER.TYPE encoder, trainable.
    Adaptive Net: small MLP that decides per-sample distillation.
    """
    from lib.models.mcitrack.encoder import build_encoder_teacher

    # Build teacher model
    teacher_encoder = build_encoder_teacher(cfg)
    teacher_neck = build_neck(cfg, teacher_encoder, d_model_override=teacher_encoder.num_channels)
    teacher_decoder = build_decoder(cfg, teacher_neck)
    teacher_model = MCITrack(
        teacher_encoder,
        teacher_decoder,
        teacher_neck,
        cfg,
        num_frames=cfg.DATA.SEARCH.NUMBER,
        num_template=cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
    )
    # Override teacher interaction indexes
    teacher_model.interaction_indexes = cfg.TRAIN.TEACHER_INTERACTION_INDEXES

    # Load teacher checkpoint and freeze
    teacher_checkpoint = torch.load(cfg.TRAIN.TEACHER_PATH, map_location="cpu")
    state_dict = teacher_checkpoint['net']
    teacher_model.load_state_dict(state_dict, strict=True)
    for p in teacher_model.parameters():
        p.requires_grad = False

    # Build student model
    student_encoder = build_encoder(cfg)
    student_neck = build_neck(cfg, student_encoder)
    student_decoder = build_decoder(cfg, student_neck)

    # Build adjust layers if channel mismatch
    if teacher_encoder.num_channels != student_encoder.num_channels:
        num_adjust = max(1, len(cfg.TRAIN.DISTILL_LAYER_T))
        adjust_layers = nn.ModuleList([
            nn.Linear(student_encoder.num_channels, teacher_encoder.num_channels, bias=False)
            for _ in range(num_adjust)
        ])
    else:
        adjust_layers = None

    student_model = MCITrack(
        student_encoder,
        student_decoder,
        student_neck,
        cfg,
        num_frames=cfg.DATA.SEARCH.NUMBER,
        num_template=cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
    )
    # Attach adjust layers to student model
    student_model.adjust_layers = adjust_layers

    # Build adaptive network
    adaptive_net = ADAPTIVE_NET(
        teacher_enc_chan=teacher_encoder.num_channels,
        student_enc_chan=student_encoder.num_channels,
        num_layer=max(1, len(cfg.TRAIN.DISTILL_LAYER_T)),
        cfg=cfg
    )

    return teacher_model, student_model, adaptive_net

