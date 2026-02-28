#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modifications Copyright (c) 2026 Yichen Zeng, Wuhan University, Email: zengyichen@whu.edu.cn
# Description: Adapted for semantic audio-visual navigation in continuous environment (SAVN-CE).

import torch
import torch.nn as nn


class SAVNCE_StateEncoder(nn.Module):
    """
    The core Scene Memory Transformer block from https://arxiv.org/abs/1903.03878
    """
    def __init__(
        self,
        feature_size: int,
        embedding_size: int,
        nhead: int = 8,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        dropout: float = 0.1,
        activation: str = "relu",
        pretraining: bool = False
    ):
        r"""A Transformer for encoding the state in RL and decoding features based on
        the observation and goal encodings.

        Supports masking the hidden state during various timesteps in the forward pass

        Args:
            feature_size: The feature size of the SMT
            nhead: The number of encoding and decoding attention heads
            num_encoder_layers: The number of encoder layers
            num_decoder_layers: The number of decoder layers
            dim_feedforward: The hidden size of feedforward layers in the transformer
            dropout: The dropout value after each attention layer
            activation: The activation to use after each linear layer
        """

        super().__init__()
        self._embedding_size = embedding_size
        self._nhead = nhead
        self._num_encoder_layers = num_encoder_layers
        self._num_decoder_layers = num_decoder_layers
        self._dropout = dropout
        self._activation = activation
        self._pretraining = pretraining

        self.fusion_encoder = nn.Sequential(
            nn.Linear(feature_size, self._embedding_size),
            nn.ReLU(),
            nn.Linear(self._embedding_size, self._embedding_size),
        )

        self.transformer = nn.Transformer(
            d_model=self._embedding_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=self._embedding_size,
            dropout=dropout,
            activation=activation,
        )


    def _convert_masks_to_transformer_format(self, memory_masks):
        r"""The memory_masks is a FloatTensor with
            -   zeros for invalid locations, and
            -   ones for valid locations.

        The required format is a BoolTensor with
            -   True for invalid locations, and
            -   False for valid locations
        """
        return (1 - memory_masks) > 0

    def single_forward(self, x, memory, memory_masks, goal=None):
        r"""Forward for a non-sequence input

        Args:
            x: (N, input_size) Tensor
            memory: The memory of encoded observations in the episode. It is a
                (M, N, input_size) Tensor.
            memory_masks: The masks indicating the set of valid memory locations
                for the current observations. It is a (N, M) Tensor.
            goal: (N, goal_dims) Tensor (optional)
        """
        # If memory_masks is all zeros for a data point, x_att will be NaN.
        # In these cases, just set memory_mask to ones and replace x_att with x.
        # all_zeros_mask = (memory_masks.sum(dim=1) == 0).float().unsqueeze(1)
        # memory_masks = 1.0 * all_zeros_mask + memory_masks * (1 - all_zeros_mask)
        if self._pretraining:
            memory_masks = torch.cat(
                [torch.zeros_like(memory_masks), torch.ones([memory_masks.shape[0], 1], device=memory_masks.device)],
                dim=1)
        else:
            memory_masks = torch.cat([memory_masks, torch.ones([memory_masks.shape[0], 1], device=memory_masks.device)],
                                     dim=1)

        # Compress features
        memory = torch.cat([memory, x.unsqueeze(0)])
        M, bs = memory.shape[:2]
        memory = self.fusion_encoder(memory.view(M*bs, -1)).view(M, bs, -1)

        # Transformer operations
        t_masks = self._convert_masks_to_transformer_format(memory_masks)
        if goal is not None:
            x_att = self.transformer(
                memory,
                goal.unsqueeze(0),
                src_key_padding_mask=t_masks,
                memory_key_padding_mask=t_masks,
            )[-1]
        else:
            decode_memory = False
            if decode_memory:
                x_att = self.transformer(
                    memory,
                    memory,
                    src_key_padding_mask=t_masks,
                    tgt_key_padding_mask=t_masks,
                    memory_key_padding_mask=t_masks,
                )[-1]
            else:
                x_att = self.transformer(
                    memory,
                    memory[-1:],
                    src_key_padding_mask=t_masks,
                    memory_key_padding_mask=t_masks,
                )[-1]

        return x_att

    @property
    def hidden_state_size(self):
        return self._embedding_size

    def forward(self, x, memory, *args, **kwargs):
        """
        Single input case:
            Inputs:
                x - (N, input_size)
                memory - (M, N, input_size)
                memory_masks - (N, M)
        Sequential input case:
            Inputs:
                x - (T*N, input_size)
                memory - (M, N, input_size)
                memory_masks - (T*N, M)
        """
        assert x.size(0) == memory.size(1)
        return self.single_forward(x, memory, *args, **kwargs)
