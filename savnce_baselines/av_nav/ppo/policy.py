#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modifications Copyright (c) 2026 Yichen Zeng, Wuhan University, Email: zengyichen@whu.edu.cn
# Description: Adapted for semantic audio-visual navigation in continuous environment (SAVN-CE).

import abc
import os

import torch
import torch.nn as nn

from habitat import logger
from savnce_baselines.common.utils import CategoricalNet, load_pretrained_weights
from savnce_baselines.av_nav.models.rnn_state_encoder import RNNStateEncoder
from savnce_baselines.av_nav.models.visual_cnn import VisualCNN
from savnce_baselines.av_nav.models.audio_cnn import AudioCNN

DUAL_GOAL_DELIMITER = ','


class Policy(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        # print('Features: ', features.cpu().numpy())
        distribution = self.action_distribution(features)
        # print('Distribution: ', distribution.logits.cpu().numpy())
        value = self.critic(features)
        # print('Value: ', value.item())

        if deterministic:
            action = distribution.mode()
            # print('Deterministic action: ', action.item())
        else:
            action = distribution.sample()
            # print('Sample action: ', action.item())

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class AudioNavBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid,
        hidden_size=512,
        extra_rgb=False,
        use_pretrained=False,
        pretrained_path=''
    ):
        super().__init__(
            AudioNavBaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                extra_rgb=extra_rgb,
                use_pretrained=use_pretrained,
                pretrained_path=pretrained_path
            ),
            action_space.n,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class AudioNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self, 
        observation_space, 
        hidden_size, 
        goal_sensor_uuid, 
        extra_rgb=False, 
        use_pretrained=False, 
        pretrained_path=''
    ):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self._audiogoal = False
        self._pointgoal = False
        self._objectgoal = False
        self._n_pointgoal = 0
        self._n_objectgoal = 0

        if 'pointgoal_with_gps_compass' in self.goal_sensor_uuid:
            self._pointgoal = True
            self._n_pointgoal = observation_space.spaces['pointgoal_with_gps_compass'].shape[0]
        if 'category' in self.goal_sensor_uuid:
            self._objectgoal = True
            self._n_objectgoal = observation_space.spaces['category'].shape[0]
        if 'spectrogram' in self.goal_sensor_uuid:
            self._audiogoal = True

        self.visual_encoder = VisualCNN(observation_space, hidden_size, extra_rgb)
        if self._audiogoal:
            if 'audiogoal' in self.goal_sensor_uuid:
                audiogoal_sensor = 'audiogoal'
            elif 'spectrogram' in self.goal_sensor_uuid:
                audiogoal_sensor = 'spectrogram'
            self.audio_encoder = AudioCNN(observation_space, hidden_size, audiogoal_sensor)

        rnn_input_size = (0 if self.is_blind else self._hidden_size) + \
                         (self._n_pointgoal if self._pointgoal else 0) + \
                         (self._n_objectgoal if self._objectgoal else 0) + \
                         (self._hidden_size if self._audiogoal else 0)
        self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)

        if use_pretrained:
            assert(pretrained_path != '')
            load_pretrained_weights(
                self,
                pretrained_path,
                dict_key='state_dict',
                prefix='actor_critic.net.',
                logger=logger
            )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []

        if self._pointgoal:
            x.append(observations['pointgoal_with_gps_compass'])
        if self._objectgoal:
            x.append(observations['category'])
        if self._audiogoal:
            x.append(self.audio_encoder(observations))
        if not self.is_blind:
            x.append(self.visual_encoder(observations))

        x1 = torch.cat(x, dim=1)
        x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        if torch.isnan(x2).any().item():
            for key in observations:
                print(key, torch.isnan(observations[key]).any().item())
            print('rnn_old', torch.isnan(rnn_hidden_states).any().item())
            print('rnn_new', torch.isnan(rnn_hidden_states1).any().item())
            print('mask', torch.isnan(masks).any().item())
            assert True
        return x2, rnn_hidden_states1
    