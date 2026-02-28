#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modifications Copyright (c) 2026 Yichen Zeng, Wuhan University, Email: zengyichen@whu.edu.cn
# Description: Adapted for semantic audio-visual navigation in continuous environment (SAVN-CE).

import abc
import itertools
import os
import torch
import torch.nn as nn
from torchsummary import summary

from habitat import logger
from savnce_baselines.common.utils import CategoricalNet, load_pretrained_weights
from savnce_baselines.magnet.models.action_encoder import ActionEncoder
from savnce_baselines.magnet.models.audio_encoder import AudioEncoder
from savnce_baselines.magnet.models.pose_encoder import PoseEncoder
from savnce_baselines.magnet.models.visual_encoder import VisualEncoder
from savnce_baselines.magnet.models.savnce_state_encoder import SAVNCE_StateEncoder

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
        ext_memory,
        ext_memory_masks,
        deterministic=False,
    ):
        features, rnn_hidden_states, ext_memory_feats = self.net(
            observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, ext_memory_feats

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks):
        features, _, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks
        )
        return self.critic(features)

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        ext_memory,
        ext_memory_masks,
    ):
        features, rnn_hidden_states, ext_memory_feats = self.net(
            observations, rnn_hidden_states, prev_actions,
            masks, ext_memory, ext_memory_masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, ext_memory_feats


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class SAVNCE_Policy(Policy):
    def __init__(self, observation_space, action_space, **kwargs):
        super().__init__(
            SAVNCE_Net(
                observation_space,
                action_space,
                **kwargs
            ),
            action_space.n
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


class SAVNCE_Net(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN. Implements the
    policy from Scene Memory Transformer: https://arxiv.org/abs/1903.03878
    """

    def __init__(
        self,
        observation_space,
        action_space,
        smt_cfg = None,
        use_goal_descriptor = False,
        use_goal_as_target = False,
        goal_descriptor_cfg = None,
        goal_sensor_uuids = ['spectrogram'],
        use_category_input = False,
    ):
        super().__init__()
        assert smt_cfg is not None, "smt_cfg is required"
        assert goal_descriptor_cfg is not None, "goal_descriptor_cfg is required"
        assert isinstance(goal_sensor_uuids, list), "goal_sensor_uuids must be a list"
        self._use_residual_connection = False
        self._use_goal_descriptor = use_goal_descriptor
        self._use_goal_as_target = use_goal_descriptor and use_goal_as_target
        self._gd_config = goal_descriptor_cfg
        self._action_size= action_space.n
        self._use_category_input = use_category_input or 'oracle_category' in goal_sensor_uuids

        self.visual_encoder = VisualEncoder(observation_space, embedding_size=64)
        self.pose_encoder = PoseEncoder(5, embedding_size=16)
        self.action_encoder = ActionEncoder(action_space.n, embedding_size=16)
        self.feature_size = self.visual_encoder.embedding_size + \
            self.pose_encoder.embedding_size + self.action_encoder.embedding_size

        self._audiogoal = False
        self._pointgoal = False
        self._objectgoal = False
        self._oracle_position = False
        self._oracle_category = False
        self._oracle_accddoa = False
        for goal_sensor_uuid in goal_sensor_uuids:
            if goal_sensor_uuid == 'spectrogram' or 'extractor' in goal_sensor_uuid:
                audiogoal_uuid = goal_sensor_uuid
                assert audiogoal_uuid in observation_space.spaces
                self._audiogoal = True
                break
        if self._audiogoal:
            audio_input_channels = observation_space.spaces[audiogoal_uuid].shape[0]
            self.audio_encoder = AudioEncoder(audio_input_channels, audiogoal_uuid, embedding_size=128)
            self.feature_size += self.audio_encoder.embedding_size
        if 'pointgoal_with_gps_compass' in goal_sensor_uuids:
            self._pointgoal = True
            assert 'pointgoal_with_gps_compass' in observation_space.spaces
            self.feature_size += observation_space.spaces['pointgoal_with_gps_compass'].shape[0]
        if 'oracle_position' in goal_sensor_uuids:
            self._oracle_position = True
            assert 'oracle_position' in observation_space.spaces
            self.feature_size += observation_space.spaces['oracle_position'].shape[0]
        if self._use_category_input:
            self._oracle_category = True
            assert 'oracle_category' in observation_space.spaces
            self.feature_size += observation_space.spaces['oracle_category'].shape[0]
        if 'oracle_accddoa' in goal_sensor_uuids:
            self._oracle_accddoa = True
            assert 'oracle_accddoa' in observation_space.spaces
            self.feature_size += observation_space.spaces['oracle_accddoa'].shape[0]    

        if self._use_goal_descriptor:
            if self._use_goal_as_target:
                if self._gd_config.encoding_goal_descriptor:
                    if self._gd_config.use_goal_descriptor_embedding:
                        self.goal_descriptor_encoder = nn.Linear(self._gd_config.embedding_size, 128)      
                    else:
                        self.goal_descriptor_encoder = nn.Linear(self._gd_config.output_size, 128)      
            else:
                if self._gd_config.use_goal_descriptor_embedding:
                    if self._gd_config.encoding_goal_descriptor:
                        self.goal_descriptor_encoder = nn.Linear(self._gd_config.embedding_size, 128)
                        self.feature_size += 128
                    else:
                        self.feature_size += self._gd_config.embedding_size
                else:
                    if self._gd_config.encoding_goal_descriptor:
                        self.goal_descriptor_encoder = nn.Linear(self._gd_config.output_size, 128)
                        self.feature_size += 128
                    else:
                        self.feature_size += self._gd_config.output_size

        self.smt_state_encoder = SAVNCE_StateEncoder(
            feature_size=self.feature_size,
            embedding_size=smt_cfg.embedding_size,
            nhead=smt_cfg.nhead,
            num_encoder_layers=smt_cfg.num_encoder_layers,
            num_decoder_layers=smt_cfg.num_decoder_layers,
            dropout=smt_cfg.dropout,
            activation=smt_cfg.activation,
            pretraining=smt_cfg.pretraining
        )

        if smt_cfg.use_pretrained:
            assert(smt_cfg.pretrained_path != '')
            load_pretrained_weights(
                self,
                smt_cfg.pretrained_path,
                dict_key='state_dict',
                prefix='actor_critic.net.',
                logger=logger
            )
        self.train()

    @property
    def memory_dim(self):
        return self.feature_size

    @property
    def output_size(self):
        size = self.smt_state_encoder.hidden_state_size
        if self._use_residual_connection:
            size += self.feature_size
        return size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return -1

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks):
        x = self.get_features(observations, prev_actions)
        if self._use_goal_as_target:
            if self._gd_config.use_goal_descriptor_embedding:
                goal = observations['goal_descriptor'][:, :self._gd_config.embedding_size]
            else:
                goal = observations['goal_descriptor'][:, self._gd_config.embedding_size:]
                if self._gd_config.encoding_goal_descriptor:
                    goal = self.goal_descriptor_encoder(goal)
        else:
            goal = None
        x_att = self.smt_state_encoder(x, ext_memory, ext_memory_masks, goal=goal)
        if self._use_residual_connection:
            x_att = torch.cat([x_att, x], 1)

        return x_att, rnn_hidden_states, x

    def freeze_encoders(self):
        """Freeze audio, visual, pose and action encoders."""
        logger.info(f'SAVNCE_Net ===> Freezing all encoders!')
        params_to_freeze = []
        params_to_freeze.append(self.visual_encoder.parameters())
        params_to_freeze.append(self.action_encoder.parameters())
        if self._audiogoal:
            params_to_freeze.append(self.audio_encoder.parameters())
        # if self._use_goal_descriptor and self._gd_config.encoding_goal_descriptor:
        #     params_to_freeze.append(self.goal_descriptor_encoder.parameters())
        for p in itertools.chain(*params_to_freeze):
            p.requires_grad = False

    def set_eval_encoders(self):
        """Sets the all encoders to eval mode."""
        self.visual_encoder.eval()
        self.pose_encoder.eval()
        self.action_encoder.eval()
        if self._audiogoal:
            self.audio_encoder.eval()
        if self._use_goal_descriptor and self._gd_config.encoding_goal_descriptor:
            self.goal_descriptor_encoder.eval()

    def get_features(self, observations, prev_actions):
        x = []
        x.append(self.visual_encoder(observations))
        x.append(self.action_encoder(prev_actions))
        if self._audiogoal:
            x.append(self.audio_encoder(observations))
        if self._pointgoal:
            x.append(observations['pointgoal_with_gps_compass'])
        if self._oracle_position:
            x.append(observations['oracle_position'])
        if self._use_category_input:
            x.append(observations['oracle_category'])
        if self._oracle_accddoa:
            x.append(observations['oracle_accddoa'])
        x.append(self.pose_encoder(observations))

        if self._use_goal_descriptor and not self._use_goal_as_target:
            if self._gd_config.use_goal_descriptor_embedding:
                gd_embedding = observations['goal_descriptor'][:, :self._gd_config.embedding_size]
                if self._gd_config.encoding_goal_descriptor:
                    x.append(self.goal_descriptor_encoder(gd_embedding))
                else:
                    x.append(gd_embedding)
            else:
                gd_predict = observations['goal_descriptor'][:, self._gd_config.embedding_size:]
                if self._gd_config.encoding_goal_descriptor:
                    x.append(self.goal_descriptor_encoder(gd_predict))
                else:
                    x.append(gd_predict)

        x = torch.cat(x, dim=-1)
        assert not torch.isnan(x).any().item()

        return x