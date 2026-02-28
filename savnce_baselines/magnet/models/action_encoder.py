#!/usr/bin/env python3

# Copyright (c) 2026 Yichen Zeng, Wuhan University

import torch
import torch.nn as nn


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, embedding_size=16):
        super().__init__()

        self.action_encoder = nn.Embedding(action_dim, embedding_size)
        self.embedding_size = embedding_size

    def forward(self, prev_actions):
        # action: (batch_size, 1)
        if isinstance(prev_actions, list):
            prev_actions = torch.tensor(prev_actions)
        if prev_actions.ndim > 1:
            prev_actions = prev_actions.squeeze(-1)
        action_embedding = self.action_encoder(prev_actions) # (batch_size, action_embedding_size)
        return action_embedding
    