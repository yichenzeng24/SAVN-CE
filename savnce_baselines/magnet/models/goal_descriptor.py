#!/usr/bin/env python3

# Copyright (c) 2026 Yichen Zeng, Wuhan University
# Description: Adapted for semantic audio-visual navigation in continuous environment (SAVN-CE).

import math
import torch
import torch.nn as nn

from habitat import Config, logger
from savnce_baselines.magnet.models.action_encoder import ActionEncoder
from savnce_baselines.magnet.models.audio_encoder import AudioEncoder
from savnce_baselines.magnet.models.pose_encoder import PoseEncoder
from savnce_baselines.magnet.models.visual_encoder import VisualEncoder
from savnce_baselines.magnet.models.metrics import SELDMetrics
from savnce_baselines.common.utils import load_pretrained_weights


class GoalDescriptor(nn.Module):
    def __init__(
        self, 
        observation_space, 
        action_space = None, 
        batch_size = None, 
        config = None, 
        device = None,
        has_distractor_sound = False,
        pose_embedding_size = 16,
        action_embedding_size = 16,
        num_classes = 21
    ):
        super().__init__()
        assert config is not None, "Config is required for goal descriptor"
        self.config = config
        self.device = device
        self._observation_space = observation_space
        self._output_format = self.config.output_format
        self._audio_goal_uuid = self.config.audio_goal_uuid
        self._has_distractor_sound = has_distractor_sound
        assert self._audio_goal_uuid in observation_space.spaces, f"Invalid audio goal sensor: {self._audio_goal_uuid}"
        audio_input_channels = observation_space.spaces[self._audio_goal_uuid].shape[0]
        self.audio_encoder = AudioEncoder(audio_input_channels, self._audio_goal_uuid, embedding_size=256)
        self.feature_size = self.audio_encoder.embedding_size
        if self.config.use_visual_encoding:
            assert 'rgb' in observation_space.spaces or 'depth' in observation_space.spaces, f"Invalid visual sensor: {self.config.use_visual_encoding}"
            self.visual_encoder = VisualEncoder(observation_space)
            self.feature_size += self.visual_encoder.embedding_size
        if self.config.use_pose_encoding:
            assert 'pose' in observation_space.spaces, f"Invalid pose sensor: {self.config.use_pose_encoding}"
            self.pose_encoder = PoseEncoder(5, pose_embedding_size)
            self.feature_size += pose_embedding_size
        if self.config.use_action_encoding:
            assert action_space is not None, "Action space is required for action encoding"
            self.action_encoder = ActionEncoder(action_space.n, action_embedding_size)
            self.feature_size += action_embedding_size
        if self._has_distractor_sound:
            assert 'oracle_category' in observation_space.spaces, f"Invalid oracle category sensor: {self._has_distractor_sound}"
            self.feature_size += num_classes
        # use mlp to project the feature to the embedding size
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.feature_size, self.config.embedding_size),
            nn.ReLU(),
            nn.Linear(self.config.embedding_size, self.config.embedding_size)
        )
        self.fusion_norm = nn.LayerNorm(self.config.embedding_size) 
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.config.embedding_size, 
            nhead=8, 
            dim_feedforward=512, 
            dropout=0.1, 
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)
        self.pos_encoder = PositionalEncoding(self.config.embedding_size)

        if self.config.output_format == 'ACCDDOA':
            self.output_head = ACCDDOA_head(self.config.embedding_size, num_classes=num_classes)
            self.metrics = SELDMetrics()
            extra_sensors = ['oracle_accddoa']
        else:
            raise ValueError(f"Invalid output format: {self.output_format}")
        if self.config.use_visual_encoding:
            extra_sensors.extend(['rgb', 'depth'])
        if self._has_distractor_sound and 'oracle_category' not in extra_sensors:
            extra_sensors.append('oracle_category')      
        if self.config.use_pretrained:
            load_pretrained_weights(
                self,
                self.config.pretrained_path,
                dict_key='goal_descriptor',
                prefix='',
                logger=logger
            )
        self.episodic_memory = EpisodicMemory(
            batch_size,
            self.config.memory_size,
            self.config.embedding_size,
            self.device
        )
        if self.config.online_training:
            self.data_generator = EpisodeGenerator(
                batch_size,
                self.config.min_episode_steps,
                device=self.device,
                extra_sensors=extra_sensors
            )
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
            self.loss_fn = SELDLoss(self.config.output_format, num_classes=num_classes)
        else:
            self.freeze_weights()  
        self.to(self.device)          

    def get_features(self, observations, prev_actions = None):
        features = []
        features.append(self.audio_encoder(observations))
        if self.config.use_visual_encoding :
            features.append(self.visual_encoder(observations))
        if self.config.use_pose_encoding:
            features.append(self.pose_encoder(observations))
        if self.config.use_action_encoding:
            features.append(self.action_encoder(prev_actions))
        if self._has_distractor_sound:
            features.append(observations['oracle_category'])
        features = torch.cat(features, dim=-1)
        return features
    
    def forward(self, observations, prev_actions = None, padding_mask = None):
        features = self.get_features(observations, prev_actions)
        seq_len = features.shape[0]
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device), diagonal=1)
        embedding = self.fusion_norm(self.fusion_proj(features))
        if padding_mask is not None:
            embedding = self.encoder(self.pos_encoder(embedding), mask=mask, src_key_padding_mask=padding_mask)
        else:
            embedding = self.encoder(self.pos_encoder(embedding), mask=mask)
        predict = self.output_head(embedding)
        return predict   

    def update(self, observations, dones, prev_actions = None):
        """
        Single forward pass for updating the episode embedding.
        Args:
            observations: tensor(batch_size, embedding_size)
            dones: list of bool (batch_size,)
            prev_actions: tensor(batch_size,) (optional)
        Returns:
            embedding: tensor(batch_size, embedding_size)
            predict: tensor(batch_size, output_size)
        """
        with torch.no_grad():
            features = self.get_features(observations, prev_actions)
            step_embedding = self.fusion_norm(self.fusion_proj(features))
            episode_embedding, episode_padding_mask = self.episodic_memory.insert(step_embedding)
            steps = self.episodic_memory.steps - 1 # the current step
            embedding = self.encoder(self.pos_encoder(episode_embedding), src_key_padding_mask=episode_padding_mask)
            self.episodic_memory.after_update(dones)
            predict = self.output_head(embedding)
        batch_idx = torch.arange(embedding.shape[0], device=self.device)
        step_embedding, step_predict = embedding[batch_idx, steps], predict[batch_idx, steps]
        gd_observation = torch.cat([step_embedding, step_predict], dim=-1)
        observations['goal_descriptor'] = gd_observation
        return gd_observation

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        logger.info("Freezing goal descriptor")

    def set_eval_encoders(self):
        self.audio_encoder.eval()
        if self.config.use_visual_encoding:
            self.visual_encoder.eval()
        if self.config.use_pose_encoding:
            self.pose_encoder.eval()
        if self.config.use_action_encoding:
            self.action_encoder.eval()


class DecentralizedDistributedMixinBelief:
    def init_distributed(self, find_unused_params: bool = True) -> None:
        r"""Initializes distributed training for the model

        1. Broadcasts the model weights from world_rank 0 to all other workers
        2. Adds gradient hooks to the model

        :param find_unused_params: Whether or not to filter out unused parameters
                                   before gradient reduction.  This *must* be True if
                                   there are any parameters in the model that where unused in the
                                   forward pass, otherwise the gradient reduction
                                   will not work correctly.
        """
        # NB: Used to hide the hooks from the nn.Module,
        # so they don't show up in the state_dict
        class Guard:
            def __init__(self, model, device):
                if torch.cuda.is_available():
                    self.ddp = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids=[device], output_device=device
                    )
                else:
                    self.ddp = torch.nn.parallel.DistributedDataParallel(model)

        self._ddp_hooks = Guard(self, self.device)

        self.reducer = self._ddp_hooks.ddp.reducer
        self.find_unused_params = find_unused_params

    def before_backward(self, loss):
        if self.find_unused_params:
            self.reducer.prepare_for_backward([loss])
        else:
            self.reducer.prepare_for_backward([])


class GoalDescriptorDDP(GoalDescriptor, DecentralizedDistributedMixinBelief):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EpisodeGenerator:
    def __init__(
        self, 
        batch_size, 
        min_episode_steps = 30, 
        max_episode_steps = 256,
        device = "cpu", 
        extra_sensors = ['oracle_accddoa']
    ):
        self.batch_size = batch_size
        self.min_episode_steps = min_episode_steps
        self.max_episode_steps = max_episode_steps
        self.device = device

        self.buffers = {
            'binaural_extractor': [torch.empty((0,), dtype=torch.float32, device=device) for _ in range(batch_size)],
            'pose': [torch.empty((0,), dtype=torch.float32, device=device) for _ in range(batch_size)],
            'prev_action': [torch.empty((0,), dtype=torch.long, device=device) for _ in range(batch_size)],
            **{sensor: [torch.empty((0,), dtype=torch.float32, device=device) for _ in range(batch_size)] for sensor in extra_sensors}
        }

    def process(self, rollouts):
        """
        Split rollouts into complete episodes. An complete episode starts with prev_action is 0, ends with next prev_action is 0.
        For example, if the prev_action is [0, 1, 3, 0, 2, 3, 3, 1, 0], the episodes will be [0, 1, 3], [0, 2, 3, 3, 1].
        Drop episodes shorter than min_episode_steps and keep unfinished episode steps in buffer.
        """
        # Step 1: append new rollouts into buffer
        for i in range(self.batch_size):
            for sensor, data in rollouts.observations.items():
                if sensor in self.buffers:
                    self.buffers[sensor][i] = torch.cat([self.buffers[sensor][i], data[1:, i]])

            self.buffers['prev_action'][i] = torch.cat([self.buffers['prev_action'][i], rollouts.prev_actions[1:, i]])
        any_episode = False
        max_episode_step = 0
        for i in range(self.batch_size):
            # Step 2: process each batch
            done_steps = (self.buffers['prev_action'][i] == 0).nonzero(as_tuple=False)[:, 0].tolist()
            start = 0
            for end in done_steps:
                episode_length = end - start
                if episode_length < self.min_episode_steps:
                    if episode_length > max_episode_step:
                        max_episode_step = episode_length
                        max_step_episode = {sensor: self.buffers[sensor][i][start:end] for sensor in self.buffers}
                    start = end
                    continue
                # clip the episode length to max_episode_steps, an episode don't have done mask if it's steps larger than 500
                clipped_end = min(end, start + self.max_episode_steps) 
                episode = {sensor: self.buffers[sensor][i][start:clipped_end] for sensor in self.buffers}
                yield episode
                any_episode = True
                start = end

            # Step 3: update buffer and done mask with unfinished episode steps
            for sensor in self.buffers:
                self.buffers[sensor][i] = self.buffers[sensor][i][start:]

        # if no episode generated, yield the episode with the most steps
        if not any_episode:
            yield max_step_episode  


class EpisodicMemory:
    def __init__(self, batch_size, memory_size = 128, embedding_size = 256, device = None):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.memory_size = memory_size
        self.device = device
        self.steps = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self.episode_embedding = torch.zeros(
            self.batch_size,
            self.memory_size,
            self.embedding_size,
            device=self.device
        )
        self.episode_padding_mask = torch.ones(
            self.batch_size,
            self.memory_size,
            dtype=torch.bool,
            device=self.device
        )

    def insert(self, step_embedding):
        """
        Insert the episode embedding.
        Args:
            step_embedding: tensor(batch_size, embedding_size)
        Returns:
            episode_embedding: tensor(batch_size, memory_size, embedding_size)
            episode_padding_mask: tensor(batch_size, memory_size), True for padding, False for not padding
        """
        step_embedding = step_embedding.to(self.device)
        for batch_index in range(self.batch_size): 
            self.episode_embedding[batch_index, self.steps[batch_index]].copy_(step_embedding[batch_index])
            self.episode_padding_mask[batch_index, self.steps[batch_index]] = False
            self.steps[batch_index] = self.steps[batch_index] + 1 % self.memory_size
        return self.episode_embedding, self.episode_padding_mask

    def after_update(self, step_done_mask):
        """
        After update, reset the episode embedding and padding mask if the episode has done, 
        or shift the episode embedding and padding mask if it is too long.
        Args:
            step_done_mask: list of bool (batch_size) or tensor(batch_size, 1), True for done, False for not done
        """
        for batch_index in range(self.batch_size):
            if step_done_mask[batch_index]:
                self.reset_at(batch_index)
            else:
                if self.steps[batch_index] >= self.memory_size:
                    self.shift_at(batch_index)
    
    def reset_at(self, batch_index):
        """
        Reset the episode embedding and padding mask, and set the step to 0.
        Args:
            batch_index: int
        """
        self.episode_embedding[batch_index] = torch.zeros(
            self.memory_size,
            self.embedding_size,
            device=self.device
        )
        self.episode_padding_mask[batch_index] = torch.ones(
            self.memory_size,
            dtype=torch.bool,
            device=self.device
        )
        self.steps[batch_index] = 0
    
    def shift_at(self, batch_index):
        """
        Shift the episode embedding and padding mask for the next step if the memory is full.
        The last step will be replaced by the new step.
        """
        self.steps[batch_index] = self.steps[batch_index] - 1
        self.episode_embedding[batch_index] = torch.cat(
            [
                self.episode_embedding[batch_index, 1:],
                self.episode_embedding[batch_index, :1]
            ]
        )
        self.episode_padding_mask[batch_index] = torch.cat(
            [
                self.episode_padding_mask[batch_index, 1:],
                self.episode_padding_mask[batch_index, :1]
            ]
        )

    def pop_at(self, envs_to_pause):
        """
        Pop the episode embedding and padding mask if the env is paused.
        """
        keep_indices = [
            index for index in range(self.batch_size)
            if index not in envs_to_pause
        ]
        self.steps= self.steps[keep_indices]
        self.episode_embedding = self.episode_embedding[keep_indices]
        self.episode_padding_mask = self.episode_padding_mask[keep_indices]
        self.batch_size = len(keep_indices) 


class ACCDDOA_head(nn.Module):
    def __init__(self, input_size, num_classes = 21):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 4 * self.num_classes)
        )
        self.doa_act = nn.Tanh()
        self.dist_act = nn.Sigmoid()
    
    def forward(self, x):
        x = self.mlp(x)
        doa = self.doa_act(x[..., : 3 * self.num_classes])
        dist = self.dist_act(x[..., 3 * self.num_classes : 4 * self.num_classes])
        return torch.cat([doa, dist], dim=-1) # (batch_size, time_steps, 4 * num_classes)


class SELDLoss(nn.Module):
    def __init__(self, output_format = 'ACCDDOA', num_classes = 21):
        super().__init__()
        self.output_format = output_format
        if self.output_format == 'ACCDDOA':
            self.accddoa_loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Invalid output format: {self.output_format}")
        self.num_classes = num_classes

    def forward(self, pred, label):    
        if self.output_format == 'ACCDDOA':
            loss = self.accddoa_loss_fn(pred, label)
        else:
            raise ValueError(f"Invalid output format: {self.output_format}")
        return loss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """
        x: (seq_len, d_model) or (batch_size, seq_len, d_model)
        """
        if x.ndim == 2: # (seq_len, d_model)
            x = x + self.pe[0, :x.size(0), :]
        else: # (batch_size, seq_len, d_model)
            x = x + self.pe[:, :x.size(1), :]
        return x