#!/usr/bin/env python3

# Copyright (c) 2026 Yichen Zeng, Wuhan University

import torch
import torch.nn as nn
from savnce.tasks.nav import PoseSensor

class PoseEncoder(nn.Module):
    def __init__(self, pose_dim, embedding_size=16, dist_scale=20, time_scale=500):
        super().__init__()
        self.dist_scale = dist_scale
        self.time_scale = time_scale
        self.pose_encoder = nn.Linear(pose_dim, embedding_size)
        self.embedding_size = embedding_size
        self.prev_pose = None

    def format_pose(self, pose):
        # pose: (batch_size, 4), containing x, y, heading, time
        # return: (batch_size, 5)
        if not isinstance(pose, torch.Tensor):
            pose = torch.tensor(pose)
        x, y, heading, time = pose[:, 0], pose[:, 1], pose[:, 2], pose[:, 3]
        cos_heading, sin_heading = torch.cos(heading), torch.sin(heading)
        normalized_x, normalized_y = x / self.dist_scale, y / self.dist_scale
        normalized_time = time / self.time_scale
        formatted_pose = torch.stack([normalized_x, normalized_y, cos_heading, sin_heading, normalized_time], -1)
        return formatted_pose
    
    def forward(self, observations):
        pose = observations[PoseSensor.cls_uuid].clone()
        # pose: (batch_size, 4)
        pose = self.format_pose(pose) # (batch_size, 5)
        pose = self.pose_encoder(pose) # (batch_size, pose_embedding_size)
        return pose

