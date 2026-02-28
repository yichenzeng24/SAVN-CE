# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn

from savnce_baselines.common.utils import Flatten
from savnce_baselines.av_nav.models.visual_cnn import conv_output_dim, layer_init


class AudioCNN(nn.Module):
    r"""A Simple 3-Conv CNN for processing audio spectrogram features

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(self, observation_space, output_size, audiogoal_sensor):
        super(AudioCNN, self).__init__()
        self._n_input_audio = observation_space.spaces[audiogoal_sensor].shape[0] # (n_channels, n_bins, n_frames)
        self._audiogoal_sensor = audiogoal_sensor

        cnn_dims = np.array(
            observation_space.spaces[audiogoal_sensor].shape[1:], dtype=np.float32
        )

        if cnn_dims[0] < 30 or cnn_dims[1] < 30:
            self._cnn_layers_kernel_size = [(5, 5), (3, 3), (3, 3)]
            self._cnn_layers_stride = [(2, 2), (2, 2), (1, 1)]
        else:
            self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
            self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self._n_input_audio,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[0],
                stride=self._cnn_layers_stride[0],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[1],
                stride=self._cnn_layers_stride[1],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[2],
                stride=self._cnn_layers_stride[2],
            ),
            #  nn.ReLU(True),
            Flatten(),
            nn.Linear(64 * cnn_dims[0] * cnn_dims[1], output_size),
            nn.ReLU(True),
        )

        layer_init(self.cnn)

    def forward(self, observations):
        audio_observations = observations[self._audiogoal_sensor] # (batch, n_channels, n_bins, n_frames)
        return self.cnn(audio_observations)
