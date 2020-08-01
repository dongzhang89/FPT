import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from core.config import cfg
import utils.net as net_utils
import modeling.ResNet as ResNet
from modeling.generate_anchors import generate_anchors
from modeling.generate_proposals import GenerateProposalsOp
from modeling.collect_and_distribute_fpn_rpn_proposals import CollectAndDistributeFpnRpnProposalsOp
import nn as mynn
# from torch.nn import DataParallel  # or your customized DataParallel module
# from sync_batchnorm import SynchronizedBatchNorm1d, patch_replication_callback
from modeling.self_trans import SelfTrans
from modeling.rendering_trans import RenderTrans
from modeling.grounding_trans import GroundTrans

# ---------------------------------------------------------------------------- #
def ResNet50_conv5_body():
    return fpn(
        ResNet.ResNet50_conv5_body, 
        fpn_level_info_ResNet50_conv5()
    )

def ResNet50_conv5_body_fpt():
    return fpn(
        ResNet.ResNet50_conv5_body, 
        fpn_level_info_ResNet50_conv5(),
        fpt_rendering=True
    )

def ResNet50_conv5_P2only_body():
    return fpn(
        ResNet.ResNet50_conv5_body,
        fpn_level_info_ResNet50_conv5(),
        P2only=True
    )

def ResNet101_conv5_body():
    return fpn(
        ResNet.ResNet101_conv5_body, 
        fpn_level_info_ResNet101_conv5()
    )

def ResNet101_conv5_P2only_body():
    return fpn(
        ResNet.ResNet101_conv5_body, 
        fpn_level_info_ResNet101_conv5(),
        P2only=True
    )

def ResNet152_conv5_body():
    return fpn(
        ResNet.ResNet152_conv5_body, 
        fpn_level_info_ResNet152_conv5()
    )

def ResNet152_conv5_P2only_body():
    return fpn(
        ResNet.ResNet152_conv5_body,
        fpn_level_info_ResNet152_conv5(),
        P2only=True
    )
# ---------------------------------------------------------------------------- #
class fpn(nn.Module):
    def __init__(self, conv_body_func, fpn_level_info, P2only=False, fpt_rendering=False):
        super().__init__()
        self.fpn_level_info = fpn_level_info
        self.P2only = P2only
        self.fpt_rendering = fpt_rendering
        self.st = SelfTrans(in_channels=cfg.FPN.DIM, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        self.rt = RenderTrans(channels_high=cfg.FPN.DIM, channels_low=cfg.FPN.DIM, upsample=False)
        self.dim_out = fpn_dim = cfg.FPN.DIM
        min_level, max_level = get_min_max_levels()
        self.num_backbone_stages = len(fpn_level_info.blobs) - (min_level - 2)
        fpn_dim_lateral = fpn_level_info.dims
        self.spatial_scale = []

        self.conv_top = nn.Conv2d(fpn_dim_lateral[0], fpn_dim, 1, 1, 0)
        if cfg.FPN.USE_GN:
            self.conv_top = nn.Sequential(
                nn.Conv2d(fpn_dim_lateral[0], fpn_dim, 1, 1, 0, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(fpn_dim), fpn_dim, eps=cfg.GROUP_NORM.EPSILON))
        else:
            self.conv_top = nn.Conv2d(fpn_dim_lateral[0], fpn_dim, 1, 1, 0)

        self.ground_lateral_modules = nn.ModuleList()
        self.posthoc_modules = nn.ModuleList()

        for i in range(self.num_backbone_stages - 1):
            self.ground_lateral_modules.append(
                ground_lateral_module(fpn_dim, fpn_dim_lateral[i+1])
            )

        for i in range(self.num_backbone_stages):
            if cfg.FPN.USE_GN:
                self.posthoc_modules.append(nn.Sequential(
                    nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1, bias=False),
                    nn.GroupNorm(net_utils.get_group_gn(fpn_dim), fpn_dim,
                                 eps=cfg.GROUP_NORM.EPSILON),
                    nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1, bias=False),
                    nn.ReLU(inplace=True)
                ))
            else:
                self.posthoc_modules.append(
                    nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1, bias=False),
                    nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1, bias=False),
                    nn.ReLU(inplace=True)
                )

            self.spatial_scale.append(fpn_level_info.spatial_scales[i])

        if self.fpt_rendering:
            self.fpt_rendering_conv1_modules = nn.ModuleList()
            self.fpt_rendering_conv2_modules = nn.ModuleList()

            for i in range(self.num_backbone_stages - 1):
                if cfg.FPN.USE_GN:
                    self.fpt_rendering_conv1_modules.append(nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, 3, 2, 1, bias=True),
                        nn.GroupNorm(net_utils.get_group_gn(fpn_dim), fpn_dim,
                                    eps=cfg.GROUP_NORM.EPSILON), nn.ReLU(inplace=True)
                    ))
                    self.fpt_rendering_conv2_modules.append(nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1, bias=True),
                        nn.GroupNorm(net_utils.get_group_gn(fpn_dim), fpn_dim, eps=cfg.GROUP_NORM.EPSILON),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    self.fpt_rendering_conv1_modules.append(
                        nn.Conv2d(fpn_dim, fpn_dim, 3, 2, 1)
                    )
                    self.fpt_rendering_conv2_modules.append(
                        nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1))
 
        if not cfg.FPN.EXTRA_CONV_LEVELS and max_level == 6:
            self.maxpool_p6 = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)
            self.spatial_scale.insert(0, self.spatial_scale[0] * 0.5)

        if cfg.FPN.EXTRA_CONV_LEVELS and max_level > 5:
            self.extra_pyramid_modules = nn.ModuleList()
            dim_in = fpn_level_info.dims[0]
            for i in range(6, max_level + 1):
                self.extra_pyramid_modules(
                    nn.Conv2d(dim_in, fpn_dim, 3, 2, 1)
                )
                dim_in = fpn_dim
                self.spatial_scale.insert(0, self.spatial_scale[0] * 0.5)

        if self.P2only:
            self.spatial_scale = self.spatial_scale[-1]

        self._init_weights()

        self.conv_body = conv_body_func()  # e.g resnet

    def _init_weights(self):
        def init_func(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.XavierFill(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        for child_m in self.children():
            if (not isinstance(child_m, nn.ModuleList) or
                not isinstance(child_m[0], ground_lateral_module)):
                child_m.apply(init_func)

    def detectron_weight_mapping(self):
        conv_body_mapping, orphan_in_detectron = self.conv_body.detectron_weight_mapping()
        mapping_to_detectron = {}
        for key, value in conv_body_mapping.items():
            mapping_to_detectron['conv_body.'+key] = value

        d_prefix = 'fpn_inner_' + self.fpn_level_info.blobs[0]
        if cfg.FPN.USE_GN:
            mapping_to_detectron['conv_top.0.weight'] = d_prefix + '_w'
            mapping_to_detectron['conv_top.1.weight'] = d_prefix + '_gn_s'
            mapping_to_detectron['conv_top.1.bias'] = d_prefix + '_gn_b'
        else:
            mapping_to_detectron['conv_top.weight'] = d_prefix + '_w'
            mapping_to_detectron['conv_top.bias'] = d_prefix + '_b'
        for i in range(self.num_backbone_stages - 1):
            p_prefix = 'ground_lateral_modules.%d.conv_lateral' % i
            d_prefix = 'fpn_inner_' + self.fpn_level_info.blobs[i+1] + '_lateral'
            if cfg.FPN.USE_GN:
                mapping_to_detectron.update({
                    p_prefix + '.0.weight' : d_prefix + '_w',
                    p_prefix + '.1.weight' : d_prefix + '_gn_s',
                    p_prefix + '.1.bias': d_prefix + '_gn_b'
                })
            else:
                mapping_to_detectron.update({
                    p_prefix + '.weight' : d_prefix + '_w',
                    p_prefix + '.bias': d_prefix + '_b'
                })

        for i in range(self.num_backbone_stages):
            p_prefix = 'posthoc_modules.%d' % i
            d_prefix = 'fpn_' + self.fpn_level_info.blobs[i]
            if cfg.FPN.USE_GN:
                mapping_to_detectron.update({
                    p_prefix + '.0.weight': d_prefix + '_w',
                    p_prefix + '.1.weight': d_prefix + '_gn_s',
                    p_prefix + '.1.bias': d_prefix + '_gn_b'
                })
            else:
                mapping_to_detectron.update({
                    p_prefix + '.weight': d_prefix + '_w',
                    p_prefix + '.bias': d_prefix + '_b'
                })

        if hasattr(self, 'extra_pyramid_modules'):
            for i in len(self.extra_pyramid_modules):
                p_prefix = 'extra_pyramid_modules.%d' % i
                d_prefix = 'fpn_%d' % (6 + i)
                mapping_to_detectron.update({
                    p_prefix + '.weight': d_prefix + '_w',
                    p_prefix + '.bias': d_prefix + '_b'
                })

        return mapping_to_detectron, orphan_in_detectron

    def forward(self, x):
        conv_body_blobs = [self.conv_body.res1(x)]
        for i in range(1, self.conv_body.convX):
            conv_body_blobs.append(
                getattr(self.conv_body, 'res%d' % (i+1))(conv_body_blobs[-1])
            )
        fpn_inner_blobs = [self.st(self.conv_top(conv_body_blobs[-1]))]

        for i in range(self.num_backbone_stages - 1):
            fpn_inner_blobs.append(
                self.ground_lateral_modules[i](fpn_inner_blobs[-1], conv_body_blobs[-(i+2)])
            )
        fpn_output_blobs = []

        if self.fpt_rendering:
            fpn_middle_blobs = []

        for i in range(self.num_backbone_stages):
            if not self.fpt_rendering:
                fpn_output_blobs.append(
                    self.posthoc_modules[i](fpn_inner_blobs[i])
                )
            else:
                fpn_middle_blobs.append(
                    self.posthoc_modules[i](fpn_inner_blobs[i])
                )

        if self.fpt_rendering:
            fpn_output_blobs.append(fpn_middle_blobs[-1])
            for i in range(2, self.num_backbone_stages + 1):
                rend_tmp = self.fpt_rendering_conv1_modules[i - 2](fpn_output_blobs[0])
                print(fpn_middle_blobs[self.num_backbone_stages - i].size())
                rend_tmp = rend_tmp + fpn_middle_blobs[self.num_backbone_stages - i]
                # rend_tmp = self.rt(fpn_middle_blobs[self.num_backbone_stages - i], rend_tmp)
                rend_tmp = self.fpt_rendering_conv2_modules[i - 2](rend_tmp)
                fpn_output_blobs.insert(0, rend_tmp)        

        if hasattr(self, 'maxpool_p6'):
            fpn_output_blobs.insert(0, self.maxpool_p6(fpn_output_blobs[0]))

        if hasattr(self, 'extra_pyramid_modules'):
            blob_in = conv_body_blobs[-1]
            fpn_output_blobs.insert(0, self.extra_pyramid_modules(blob_in))
            for module in self.extra_pyramid_modules[1:]:
                fpn_output_blobs.insert(0, module(F.relu(fpn_output_blobs[0], inplace=True)))

        if self.P2only:
            return fpn_output_blobs[-1]
        else:
            return fpn_output_blobs

class ground_lateral_module(nn.Module):
    def __init__(self, dim_in_top, dim_in_lateral):
        super().__init__()
        self.dim_in_top = dim_in_top
        self.dim_in_lateral = dim_in_lateral
        self.dim_out = dim_in_top
        if cfg.FPN.USE_GN:
            self.conv_lateral = nn.Sequential(
                nn.Conv2d(dim_in_lateral, self.dim_out, 3, 1, 1, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(self.dim_out), self.dim_out,
                             eps=cfg.GROUP_NORM.EPSILON),
                nn.Conv2d(dim_in_lateral, self.dim_out, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv_lateral = nn.Sequential(
                nn.Conv2d(dim_in_lateral, self.dim_out, 3, 1, 1, bias=False),
                nn.Conv2d(dim_in_lateral, self.dim_out, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True)
            )

        self._init_weights()
        self.st = SelfTrans(in_channels=cfg.FPN.DIM, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        self.gt = GroundTrans(in_channels=cfg.FPN.DIM, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
    def _init_weights(self):
        if cfg.FPN.USE_GN:
            conv = self.conv_lateral[0]
        else:
            conv = self.conv_lateral

        if cfg.FPN.ZERO_INIT_LATERAL:
            init.constant_(conv.weight, 0)
        else:
            mynn.init.XavierFill(conv.weight)
        if conv.bias is not None:
            init.constant_(conv.bias, 0)

    def forward(self, top_blob, lateral_blob):
        lat = self.conv_lateral(lateral_blob)
        lat = self.st(lat)
        td = top_blob
        return self.gt(top_blob, lat)


def get_min_max_levels():
    min_level = 2
    max_level = 5
    if cfg.FPN.MULTILEVEL_RPN and not cfg.FPN.MULTILEVEL_ROIS:
        max_level = cfg.FPN.RPN_MAX_LEVEL
        min_level = cfg.FPN.RPN_MIN_LEVEL
    if not cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
        max_level = cfg.FPN.ROI_MAX_LEVEL
        min_level = cfg.FPN.ROI_MIN_LEVEL
    if cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
        max_level = max(cfg.FPN.RPN_MAX_LEVEL, cfg.FPN.ROI_MAX_LEVEL)
        min_level = min(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.ROI_MIN_LEVEL)
    return min_level, max_level

# ---------------------------------------------------------------------------- #
class fpn_rpn_outputs(nn.Module):
    """Add RPN on FPN specific outputs."""
    def __init__(self, dim_in, spatial_scales):
        super().__init__()
        self.dim_in = dim_in
        self.spatial_scales = spatial_scales
        self.dim_out = self.dim_in
        num_anchors = len(cfg.FPN.RPN_ASPECT_RATIOS)

        # Create conv ops shared by all FPN levels
        self.FPN_RPN_conv = nn.Conv2d(dim_in, self.dim_out, 3, 1, 1)
        dim_score = num_anchors * 2 if cfg.RPN.CLS_ACTIVATION == 'softmax' \
            else num_anchors
        self.FPN_RPN_cls_score = nn.Conv2d(self.dim_out, dim_score, 1, 1, 0)
        self.FPN_RPN_bbox_pred = nn.Conv2d(self.dim_out, 4 * num_anchors, 1, 1, 0)

        self.GenerateProposals_modules = nn.ModuleList()
        k_max = cfg.FPN.RPN_MAX_LEVEL  # coarsest level of pyramid
        k_min = cfg.FPN.RPN_MIN_LEVEL  # finest level of pyramid
        for lvl in range(k_min, k_max + 1):
            sc = self.spatial_scales[k_max - lvl]  # in reversed order
            lvl_anchors = generate_anchors(
                stride=2.**lvl,
                sizes=(cfg.FPN.RPN_ANCHOR_START_SIZE * 2.**(lvl - k_min), ),
                aspect_ratios=cfg.FPN.RPN_ASPECT_RATIOS
            )
            self.GenerateProposals_modules.append(GenerateProposalsOp(lvl_anchors, sc))

        self.CollectAndDistributeFpnRpnProposals = CollectAndDistributeFpnRpnProposalsOp()

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.FPN_RPN_conv.weight, std=0.01)
        init.constant_(self.FPN_RPN_conv.bias, 0)
        init.normal_(self.FPN_RPN_cls_score.weight, std=0.01)
        init.constant_(self.FPN_RPN_cls_score.bias, 0)
        init.normal_(self.FPN_RPN_bbox_pred.weight, std=0.01)
        init.constant_(self.FPN_RPN_bbox_pred.bias, 0)

    def detectron_weight_mapping(self):
        k_min = cfg.FPN.RPN_MIN_LEVEL
        mapping_to_detectron = {
            'FPN_RPN_conv.weight': 'conv_rpn_fpn%d_w' % k_min,
            'FPN_RPN_conv.bias': 'conv_rpn_fpn%d_b' % k_min,
            'FPN_RPN_cls_score.weight': 'rpn_cls_logits_fpn%d_w' % k_min,
            'FPN_RPN_cls_score.bias': 'rpn_cls_logits_fpn%d_b' % k_min,
            'FPN_RPN_bbox_pred.weight': 'rpn_bbox_pred_fpn%d_w' % k_min,
            'FPN_RPN_bbox_pred.bias': 'rpn_bbox_pred_fpn%d_b' % k_min
        }
        return mapping_to_detectron, []

    def forward(self, blobs_in, im_info, roidb=None):
        k_max = cfg.FPN.RPN_MAX_LEVEL  # coarsest level of pyramid
        k_min = cfg.FPN.RPN_MIN_LEVEL  # finest level of pyramid
        assert len(blobs_in) == k_max - k_min + 1
        return_dict = {}
        rois_blobs = []
        score_blobs = []
        for lvl in range(k_min, k_max + 1):
            slvl = str(lvl)
            bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order

            fpn_rpn_conv = F.relu(self.FPN_RPN_conv(bl_in), inplace=True)
            fpn_rpn_cls_score = self.FPN_RPN_cls_score(fpn_rpn_conv)
            fpn_rpn_bbox_pred = self.FPN_RPN_bbox_pred(fpn_rpn_conv)
            return_dict['rpn_cls_logits_fpn' + slvl] = fpn_rpn_cls_score
            return_dict['rpn_bbox_pred_fpn' + slvl] = fpn_rpn_bbox_pred

            if not self.training or cfg.MODEL.FASTER_RCNN:
                # Proposals are needed during:
                #  1) inference (== not model.train) for RPN only and Faster R-CNN
                #  OR
                #  2) training for Faster R-CNN
                # Otherwise (== training for RPN only), proposals are not needed
                if cfg.RPN.CLS_ACTIVATION == 'softmax':
                    B, C, H, W = fpn_rpn_cls_score.size()
                    fpn_rpn_cls_probs = F.softmax(
                        fpn_rpn_cls_score.view(B, 2, C // 2, H, W), dim=1)
                    fpn_rpn_cls_probs = fpn_rpn_cls_probs[:, 1].squeeze(dim=1)
                else:  # sigmoid
                    fpn_rpn_cls_probs = F.sigmoid(fpn_rpn_cls_score)

                fpn_rpn_rois, fpn_rpn_roi_probs = self.GenerateProposals_modules[lvl - k_min](
                    fpn_rpn_cls_probs, fpn_rpn_bbox_pred, im_info)
                rois_blobs.append(fpn_rpn_rois)
                score_blobs.append(fpn_rpn_roi_probs)
                return_dict['rpn_rois_fpn' + slvl] = fpn_rpn_rois
                return_dict['rpn_rois_prob_fpn' + slvl] = fpn_rpn_roi_probs

        if cfg.MODEL.FASTER_RCNN:
            # CollectAndDistributeFpnRpnProposals also labels proposals when in training mode
            blobs_out = self.CollectAndDistributeFpnRpnProposals(rois_blobs + score_blobs, roidb, im_info)
            return_dict.update(blobs_out)

        return return_dict


def fpn_rpn_losses(**kwargs):
    """Add RPN on FPN specific losses."""
    losses_cls = []
    losses_bbox = []
    for lvl in range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1):
        slvl = str(lvl)
        # Spatially narrow the full-sized RPN label arrays to match the feature map shape
        b, c, h, w = kwargs['rpn_cls_logits_fpn' + slvl].shape
        rpn_labels_int32_fpn = kwargs['rpn_labels_int32_wide_fpn' + slvl][:, :, :h, :w]
        h, w = kwargs['rpn_bbox_pred_fpn' + slvl].shape[2:]
        rpn_bbox_targets_fpn = kwargs['rpn_bbox_targets_wide_fpn' + slvl][:, :, :h, :w]
        rpn_bbox_inside_weights_fpn = kwargs[
            'rpn_bbox_inside_weights_wide_fpn' + slvl][:, :, :h, :w]
        rpn_bbox_outside_weights_fpn = kwargs[
            'rpn_bbox_outside_weights_wide_fpn' + slvl][:, :, :h, :w]

        if cfg.RPN.CLS_ACTIVATION == 'softmax':
            rpn_cls_logits_fpn = kwargs['rpn_cls_logits_fpn' + slvl].view(
                b, 2, c // 2, h, w).permute(0, 2, 3, 4, 1).contiguous().view(-1, 2)
            rpn_labels_int32_fpn = rpn_labels_int32_fpn.contiguous().view(-1).long()
            # the loss is averaged over non-ignored targets
            loss_rpn_cls_fpn = F.cross_entropy(
                rpn_cls_logits_fpn, rpn_labels_int32_fpn, ignore_index=-1)
        else:  # sigmoid
            weight = (rpn_labels_int32_fpn >= 0).float()
            loss_rpn_cls_fpn = F.binary_cross_entropy_with_logits(
                kwargs['rpn_cls_logits_fpn' + slvl], rpn_labels_int32_fpn.float(), weight,
                size_average=False)
            loss_rpn_cls_fpn /= cfg.TRAIN.RPN_BATCH_SIZE_PER_IM * cfg.TRAIN.IMS_PER_BATCH

        # Normalization by (1) RPN_BATCH_SIZE_PER_IM and (2) IMS_PER_BATCH is
        # handled by (1) setting bbox outside weights and (2) SmoothL1Loss
        # normalizes by IMS_PER_BATCH
        loss_rpn_bbox_fpn = net_utils.smooth_l1_loss(
            kwargs['rpn_bbox_pred_fpn' + slvl], rpn_bbox_targets_fpn,
            rpn_bbox_inside_weights_fpn, rpn_bbox_outside_weights_fpn,
            beta=1/9)

        losses_cls.append(loss_rpn_cls_fpn)
        losses_bbox.append(loss_rpn_bbox_fpn)

    return losses_cls, losses_bbox


# ---------------------------------------------------------------------------- #
FpnLevelInfo = collections.namedtuple(
    'FpnLevelInfo',
    ['blobs', 'dims', 'spatial_scales']
)

def fpn_level_info_ResNet50_conv5():
    return FpnLevelInfo(
        blobs=('res5_2_sum', 'res4_5_sum', 'res3_3_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )

def fpn_level_info_ResNet101_conv5():
    return FpnLevelInfo(
        blobs=('res5_2_sum', 'res4_22_sum', 'res3_3_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_ResNet152_conv5():
    return FpnLevelInfo(
        blobs=('res5_2_sum', 'res4_35_sum', 'res3_7_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )