#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modifications Copyright (c) 2026 Yichen Zeng, Wuhan University, Email: zengyichen@whu.edu.cn
# Description: Adapted for semantic audio-visual navigation in continuous environment (SAVN-CE).

import torch
import torch.nn as nn
from savnce_baselines.magnet.models.smt_resnet import custom_resnet18


class VisualEncoder(nn.Module):
    r"""A modified ResNet-18 architecture from https://arxiv.org/abs/1903.03878.

    Takes in observations and produces an embedding of the rgb and/or depth
    and/or semantic components.

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(self, observation_space, embedding_size=64):
        super().__init__()

        self.embedding_size = 0
        self.rgb_enabled = False
        self.depth_enabled = False

        if "rgb" in observation_space.spaces:
            self.rgb_enabled = True
            n_input_rgb = observation_space.spaces["rgb"].shape[2]
            self.rgb_encoder = custom_resnet18(num_input_channels=n_input_rgb, num_classes=embedding_size)
            self.embedding_size += embedding_size

        if "depth" in observation_space.spaces:
            self.depth_enabled = True
            n_input_depth = observation_space.spaces["depth"].shape[2]
            self.depth_encoder = custom_resnet18(num_input_channels=n_input_depth, num_classes=embedding_size)
            self.embedding_size += embedding_size

        assert self.rgb_enabled or self.depth_enabled, "VisualEncoder: No visual modalities enabled"
        
        self.layer_init()

    def layer_init(self):
        def weights_init(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, nn.init.calculate_gain("relu")
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

        self.apply(weights_init)

    def forward(self, observations):
        visual_features = []
        if self.rgb_enabled:
            rgb_observations = observations["rgb"]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2) # (batch_size, H, W, C) -> (batch_size, C, H, W)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            visual_features.append(self.rgb_encoder(rgb_observations)) # (batch_size, rgb_embedding_size)

        if self.depth_enabled:
            depth_observations = observations["depth"]
            depth_observations = depth_observations.permute(0, 3, 1, 2) # (batch_size, H, W, C) -> (batch_size, C, H, W)
            visual_features.append(self.depth_encoder(depth_observations)) # (batch_size, depth_embedding_size)

        visual_features = torch.cat(visual_features, dim=1) # (batch_size, visual_embedding_size)

        return visual_features

