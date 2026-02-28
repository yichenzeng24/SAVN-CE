#!/usr/bin/env python3

# Copyright (c) 2026 Yichen Zeng, Wuhan University

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Convolutional block with Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout.
    Designed for feature extraction from audio input.
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), pool_size=(4, 2), dropout=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        x = self.dropout(x)
        return x
    

class AudioEncoder(nn.Module):
    r"""
    A Simple CRNN model, which takes in audio features and produces an embedding of the audio features.

    Args:
        feature_extractor_uuid: The uuid of the audio feature extractor
        (audio_feature_dim, time_steps, num_bins) -> (1, audio_feature_dim, time_steps, num_bins)
        output_size: The size of the embedding vector
    Returns:
        x: The embedding of the audio features
        (1, output_size)
    """

    def __init__(self, input_channels, feature_extractor_uuid, embedding_size=128, has_distractor_sound=False):
        super().__init__()
        self._feature_extractor_uuid = feature_extractor_uuid
        self.embedding_size = embedding_size
        self._has_distractor_sound = has_distractor_sound
        if has_distractor_sound:
            input_channels += 21

        self.conv_block1 = ConvBlock(in_channels=input_channels, out_channels=32, pool_size=(4, 2))
        self.conv_block2 = ConvBlock(in_channels=32, out_channels=64, pool_size=(4, 2))
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=128, pool_size=(4, 2))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1536, out_features=embedding_size),
        )
    
    def forward(self, observations):
        audio = observations[self._feature_extractor_uuid] # (batch, n_channels, n_bins, n_frames)
        if self._has_distractor_sound:
            labels = observations['category']
            expanded_labels = labels.reshape(labels.shape + (1, 1)).expand(labels.shape + audio.shape[-2:])
            audio = torch.cat([audio, expanded_labels], dim=1)
        audio = self.conv_block1(audio) 
        audio = self.conv_block2(audio) 
        audio = self.conv_block3(audio) 
        audio_features = self.fc(audio) # (batch_size, embedding_size)
        return audio_features

