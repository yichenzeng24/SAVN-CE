#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modifications Copyright (c) 2026 Yichen Zeng, Wuhan University, Email: zengyichen@whu.edu.cn
# Description: Adapted for semantic audio-visual navigation in continuous environment (SAVN-CE).

import logging
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class AudioGoalPredictor(nn.Module):
    def __init__(self):
        super(AudioGoalPredictor, self).__init__()
        self.input_shape_printed = False
        self.predictor = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.predictor.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.predictor.fc = nn.Linear(512, 21)

    def forward(self, audio_feature):
        if not self.input_shape_printed:
            logging.info('Audiogoal predictor input audio feature shape: {}'.format(audio_feature.shape))
            self.input_shape_printed = True
        return self.predictor(audio_feature)

