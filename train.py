# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import argparse

import mxnet as mx
from mxnet.module import Module

from symdata.loader import AnchorGenerator, AnchorSampler, AnchorLoader
from symnet.model import load_param, infer_data_shape, check_shape, initialize_frcnn, get_fixed_params
from symnet.metric import RPNAccMetric, RPNLogLossMetric, RPNL1LossMetric, RCNNAccMetric, RCNNLogLossMetric, \
    RCNNL1LossMetric
from symimdb.pascal_voc import PascalVOC


def train_net(sym, roidb, args):
    img_long_side = 1000
    img_short_side = 600

    # setup multi-gpu
    batch_size = 1

    # load training data
    feat_sym = sym.get_internals()['rpn_cls_score_output']
    ag = AnchorGenerator(feat_stride=16, anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios)
    asp = AnchorSampler(batch_rois=256, fg_fraction=0.5, fg_overlap=0.7, bg_overlap=0.3)
    train_data = AnchorLoader(roidb, batch_size, img_short_side, img_long_side,
                              args.img_pixel_means, args.img_pixel_stds, feat_sym, ag, asp, shuffle=True)

    # produce shape max possible
    _, out_shape, _ = feat_sym.infer_shape(data=(1, 3, img_long_side, img_long_side))
    feat_height, feat_width = out_shape[0][-2:]
    rpn_num_anchors = len(args.rpn_anchor_scales) * len(args.rpn_anchor_ratios)
    data_names = ['data', 'im_info', 'gt_boxes']
    label_names = ['label', 'bbox_target', 'bbox_weight']
    data_shapes = [('data', (batch_size, 3, img_long_side, img_long_side)),
                   ('im_info', (batch_size, 3)),
                   ('gt_boxes', (batch_size, 100, 5))]
    label_shapes = [('label', (batch_size, 1, rpn_num_anchors * feat_height, feat_width)),
                    ('bbox_target', (batch_size, 4 * rpn_num_anchors, feat_height, feat_width)),
                    ('bbox_weight', (batch_size, 4 * rpn_num_anchors, feat_height, feat_width))]

    # print shapes
    data_shape_dict, out_shape_dict = infer_data_shape(sym, data_shapes + label_shapes)
    mx.viz.print_summary(sym, shape=dict(data_shapes + label_shapes))

    # load and initialize params
    arg_params, aux_params = load_param(args.pretrained)
    arg_params, aux_params = initialize_frcnn(sym, data_shapes, arg_params, aux_params)

    # check parameter shapes
    check_shape(sym, data_shapes + label_shapes, arg_params, aux_params)

    # check fixed params
    fixed_param_names = get_fixed_params(sym, args.net_fixed_params)

    # metric
    eval_metrics = mx.metric.CompositeEvalMetric()
    eval_metrics.add(RPNAccMetric())
    eval_metrics.add(RPNLogLossMetric())
    eval_metrics.add(RPNL1LossMetric())
    eval_metrics.add(RCNNAccMetric())
    eval_metrics.add(RCNNLogLossMetric())
    eval_metrics.add(RCNNL1LossMetric())

    # learning schedule
    base_lr = 0.001
    lr_factor = 0.1
    lr_epoch = [7]

    lr_epoch_diff = lr_epoch
    lr = base_lr

    lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)
    # optimizer
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.0005,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': (1.0 / batch_size),
                        'clip_gradient': 5}

    # train
    mod = Module(sym, data_names=data_names, label_names=label_names,
                 fixed_param_names=fixed_param_names)
    mod.fit(train_data, eval_metric=eval_metrics, kvstore='device',
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=0,
            num_epoch=10)


def parse_args():
    # faster rcnn params
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.rpn_anchor_scales = (8, 16, 32)
    args.rpn_anchor_ratios = (0.5, 1, 2)

    return args


def get_voc(args):
    args.rcnn_num_classes = len(PascalVOC.classes)

    iset = '2007_trainval'
    imdb = PascalVOC(iset, 'data', 'data/VOCdevkit')
    imdb.append_flipped_images()
    return imdb.roidb


def get_vgg16_train(args):
    from symnet.symbol_vgg import get_vgg_train
    args.pretrained = 'model/vgg16-0000.params'
    args.img_pixel_means = (123.68, 116.779, 103.939)
    args.img_pixel_stds = (1.0, 1.0, 1.0)
    args.net_fixed_params = ['conv1', 'conv2']
    args.rcnn_feat_stride = 16
    args.rcnn_pooled_size = (7, 7)
    return get_vgg_train(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                         rpn_feature_stride=16, rpn_pre_topk=12000,
                         rpn_post_topk=2000, rpn_nms_thresh=0.7,
                         rpn_min_size=16, rpn_batch_rois=256,
                         num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                         rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=1,
                         rcnn_batch_rois=128, rcnn_fg_fraction=0.25,
                         rcnn_fg_overlap=0.5, rcnn_bbox_stds=(0.1, 0.1, 0.2, 0.2))


def main():
    args = parse_args()
    roidb = get_voc(args)
    sym = get_vgg16_train(args)
    train_net(sym, roidb, args)


if __name__ == '__main__':
    main()
