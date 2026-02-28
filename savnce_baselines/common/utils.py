#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modifications Copyright (c) 2026 Yichen Zeng, Wuhan University, Email: zengyichen@whu.edu.cn
# Description: Adapted for semantic audio-visual navigation in continuous environment (SAVN-CE).

import glob
import os
from collections import defaultdict
from typing import Dict, List, Optional
import random
import copy
import numbers
import json

import numpy as np
import cv2
from scipy.io import wavfile
import torch
import torch.nn as nn
import torch.nn.functional as f
import moviepy.editor as mpy
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from gym.spaces import Box
from moviepy.audio.AudioClip import CompositeAudioClip, AudioArrayClip

from habitat.utils.visualizations.utils import images_to_video
from habitat import logger
from habitat_sim.utils.common import d3_40_colors_rgb
from savnce_baselines.common.tensorboard_utils import TensorboardWriter
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import draw_collision

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class CustomFixedCategorical(torch.distributions.Categorical):
    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class CategoricalNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return CustomFixedCategorical(logits=x)


class CategoricalNetWithMask(nn.Module):
    def __init__(self, num_inputs, num_outputs, masking):
        super().__init__()
        self.masking = masking

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, features, action_maps):
        probs = f.softmax(self.linear(features))
        if self.masking:
            probs = probs * torch.reshape(action_maps, (action_maps.shape[0], -1)).float()

        return CustomFixedCategorical(probs=probs)


def linear_decay(epoch: int, total_num_updates: int) -> float:
    r"""Returns a multiplicative factor for linear value decay

    Args:
        epoch: current epoch number
        total_num_updates: total number of epochs

    Returns:
        multiplicative factor that decreases param value linearly
    """
    return 1 - (epoch / float(total_num_updates))


def exponential_decay(epoch: int, total_num_updates: int, decay_lambda: float) -> float:
    r"""Returns a multiplicative factor for linear value decay

    Args:
        epoch: current epoch number
        total_num_updates: total number of epochs
        decay_lambda: decay lambda

    Returns:
        multiplicative factor that decreases param value linearly
    """
    return np.exp(-decay_lambda * (epoch / float(total_num_updates)))


def to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


def batch_obs(
    observations: List[Dict], device: Optional[torch.device] = None, skip_list = []
) -> Dict[str, torch.Tensor]:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of lists of observations.
    """
    batch = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            if sensor in skip_list:
                continue
            batch[sensor].append(to_tensor(obs[sensor]).float())

    for sensor in batch:
        batch[sensor] = torch.stack(batch[sensor], dim=0).to(
            device=device, dtype=torch.float
        )

    return batch


def poll_checkpoint_folder(
    checkpoint_folder: str, previous_ckpt_ind: int, eval_interval: int
) -> Optional[str]:
    r""" Return (previous_ckpt_ind + 1)th checkpoint in checkpoint folder
    (sorted by time of last modification).

    Args:
        checkpoint_folder: directory to look for checkpoints.
        previous_ckpt_ind: index of checkpoint last returned.
        eval_interval: number of checkpoints between two evaluation

    Returns:
        return checkpoint path if (previous_ckpt_ind + 1)th checkpoint is found
        else return None.
    """
    assert os.path.isdir(checkpoint_folder), (
        f"invalid checkpoint folder " f"path {checkpoint_folder}"
    )
    models_paths = list(
        filter(os.path.isfile, glob.glob(checkpoint_folder + "/*"))
    )
    models_paths.sort(key=os.path.getmtime)
    ind = previous_ckpt_ind + eval_interval
    if ind < len(models_paths):
        return models_paths[ind]
    return None


def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[np.ndarray],
    scene_name: str,
    sound: str,
    sr: int,
    episode_id: int,
    checkpoint_idx: int,
    metric_name: str,
    metric_value: float,
    tb_writer: TensorboardWriter,
    fps: int = 10,
    audios: List[str] = None
) -> None:
    r"""Generate video according to specified information.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
        audios: raw audio files
    Returns:
        None
    """
    if len(images) < 1:
        return

    video_name = f"{scene_name}_{episode_id}_{sound}_{metric_name}{metric_value:.2f}"
    if "disk" in video_option:
        assert video_dir is not None
        if audios is None:
            images_to_video(images, video_dir, video_name)
        else:
            images_to_video_with_audio(images, video_dir, video_name, audios, sr, fps=fps)
    if "tensorboard" in video_option:
        tb_writer.add_video_from_np_images(
            f"episode{episode_id}", checkpoint_idx, images, fps=fps
        )


def plot_top_down_map(info, dataset='replica', pred=None):
    top_down_map = info["top_down_map"]["map"]
    top_down_map = maps.colorize_topdown_map(
        top_down_map, info["top_down_map"]["fog_of_war_mask"]
    )
    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    if dataset == 'replica':
        agent_radius_px = top_down_map.shape[0] // 16
    else:
        agent_radius_px = top_down_map.shape[0] // 25
    top_down_map = maps.draw_agent(
        image=top_down_map,
        agent_center_coord=map_agent_pos,
        agent_rotation=info["top_down_map"]["agent_angle"],
        agent_radius_px=agent_radius_px
    )
    # if pred is not None:
    #     from habitat.utils.geometry_utils import quaternion_rotate_vector
    #     from habitat_sim.utils.common import quat_from_coeffs
    #     ### code need to be tested
    #     agent_angle = info["top_down_map"]["agent_angle"]
    #     source_rotation = [0.0, np.sin(agent_angle / 2), 0, np.cos(agent_angle / 2)] # format: [x, y, z, w]
    #     source_rotation = quat_from_coeffs(source_rotation)

    #     rounded_pred = np.round(pred[1])
    #     direction_vector_agent = np.array([rounded_pred[1], 0, -rounded_pred[0]])
    #     direction_vector = quaternion_rotate_vector(source_rotation, direction_vector_agent)

    #     grid_size = (
    #         top_down_map.shape[0] / 10000,
    #         top_down_map.shape[1] / 10000,
    #     )
    #     ### code need to be tested
    #     delta_x = int(-direction_vector[0] / grid_size[0])
    #     delta_y = int(direction_vector[2] / grid_size[1])

    #     x = np.clip(map_agent_pos[0] + delta_x, a_min=0, a_max=top_down_map.shape[0])
    #     y = np.clip(map_agent_pos[1] + delta_y, a_min=0, a_max=top_down_map.shape[1])
    #     point_padding = 12
    #     for m in range(x - point_padding, x + point_padding + 1):
    #         for n in range(y - point_padding, y + point_padding + 1):
    #             if np.linalg.norm(np.array([m - x, n - y])) <= point_padding and \
    #                     0 <= m < top_down_map.shape[0] and 0 <= n < top_down_map.shape[1]:
    #                 top_down_map[m, n] = (0, 255, 255)
    #     if np.linalg.norm(rounded_pred) < 1:
    #         assert delta_x == 0 and delta_y == 0

    if top_down_map.shape[0] > top_down_map.shape[1]:
        top_down_map = np.rot90(top_down_map, 1)
    return top_down_map


def images_to_video_with_audio(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    audios: List[str],
    sr: int,
    fps: int = 1,
    quality: Optional[float] = 5,
    threads: int = 8,
    **kwargs
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        audios: raw audio files
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    """

    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    multiplier = 0.5
    audios = np.concatenate(audios, axis=0)
    audios = audios / np.max(np.abs(audios)) * multiplier
    audio_clip = AudioArrayClip(audios, fps=sr)
    video_clip = mpy.ImageSequenceClip(images, fps=fps)
    video_with_new_audio = video_clip.set_audio(audio_clip)
    video_with_new_audio.write_videofile(os.path.join(output_dir, video_name), fps=fps, threads=threads)

def resize_observation(observations, model_resolution):
    obs_copy = copy.deepcopy(observations)
    for observation in obs_copy:
        if 'rgb' in observation:
            observation['rgb'] = cv2.resize(observation['rgb'], (model_resolution, model_resolution))
        if 'depth' in observation:
            observation['depth'] = np.expand_dims(cv2.resize(observation['depth'], (model_resolution, model_resolution)), axis=-1)
        if 'semantic' in observation:
            observation['semantic'] = cv2.resize(observation['semantic'], (model_resolution, model_resolution))
    return obs_copy

def convert_semantics_to_rgb(semantics):
    r"""Converts semantic IDs to RGB images.
    """
    semantics = semantics.long() % 40
    mapping_rgb = torch.from_numpy(d3_40_colors_rgb).to(semantics.device)
    semantics_r = torch.take(mapping_rgb[:, 0], semantics)
    semantics_g = torch.take(mapping_rgb[:, 1], semantics)
    semantics_b = torch.take(mapping_rgb[:, 2], semantics)
    semantics_rgb = torch.stack([semantics_r, semantics_g, semantics_b], -1)

    return semantics_rgb


class ResizeCenterCropper(nn.Module):
    def __init__(self, size, channels_last: bool = False):
        r"""An nn module the resizes and center crops your input.
        Args:
            size: A sequence (w, h) or int of the size you wish to resize/center_crop.
                    If int, assumes square crop
            channels_list: indicates if channels is the last dimension
        """
        super().__init__()
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        assert len(size) == 2, "forced input size must be len of 2 (w, h)"
        self._size = size
        self.channels_last = channels_last

    def transform_observation_space(
        self, observation_space, trans_keys=["rgb", "depth", "semantic"]
    ):
        size = self._size
        observation_space = copy.deepcopy(observation_space)
        # observation_space = copy.copy(observation_space)
        if size:
            for key in observation_space.spaces:
                if (
                    key in trans_keys
                    and observation_space.spaces[key].shape != size
                ):
                    logger.info(
                        "Overwriting CNN input size of %s: %s" % (key, size)
                    )
                    observation_space.spaces[key] = overwrite_gym_box_shape(
                        observation_space.spaces[key], size
                    )
        self.observation_space = observation_space
        return observation_space

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._size is None:
            return input

        return center_crop(
            image_resize_shortest_edge(
                input, max(self._size), channels_last=self.channels_last
            ),
            self._size,
            channels_last=self.channels_last,
        )


def image_resize_shortest_edge(
    img, size: int, channels_last: bool = False
) -> torch.Tensor:
    """Resizes an img so that the shortest side is length of size while
        preserving aspect ratio.

    Args:
        img: the array object that needs to be resized (HWC) or (NHWC)
        size: the size that you want the shortest edge to be resize to
        channels: a boolean that channel is the last dimension
    Returns:
        The resized array as a torch tensor.
    """
    img = to_tensor(img)
    no_batch_dim = len(img.shape) == 3
    if len(img.shape) < 3 or len(img.shape) > 5:
        raise NotImplementedError()
    if no_batch_dim:
        img = img.unsqueeze(0)  # Adds a batch dimension
    if channels_last:
        h, w = img.shape[-3:-1]
        if len(img.shape) == 4:
            # NHWC -> NCHW
            img = img.permute(0, 3, 1, 2)
        else:
            # NDHWC -> NDCHW
            img = img.permute(0, 1, 4, 2, 3)
    else:
        # ..HW
        h, w = img.shape[-2:]

    # Percentage resize
    scale = size / min(h, w)
    h = int(h * scale)
    w = int(w * scale)
    img = torch.nn.functional.interpolate(
        img.float(), size=(h, w), mode="area"
    ).to(dtype=img.dtype)
    if channels_last:
        if len(img.shape) == 4:
            # NCHW -> NHWC
            img = img.permute(0, 2, 3, 1)
        else:
            # NDCHW -> NDHWC
            img = img.permute(0, 1, 3, 4, 2)
    if no_batch_dim:
        img = img.squeeze(dim=0)  # Removes the batch dimension
    return img


def center_crop(img, size, channels_last: bool = False):
    """Performs a center crop on an image.

    Args:
        img: the array object that needs to be resized (either batched or unbatched)
        size: A sequence (w, h) or a python(int) that you want cropped
        channels_last: If the channels are the last dimension.
    Returns:
        the resized array
    """
    if channels_last:
        # NHWC
        h, w = img.shape[-3:-1]
    else:
        # NCHW
        h, w = img.shape[-2:]

    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    assert len(size) == 2, "size should be (h,w) you wish to resize to"
    cropx, cropy = size

    startx = w // 2 - (cropx // 2)
    starty = h // 2 - (cropy // 2)
    if channels_last:
        return img[..., starty : starty + cropy, startx : startx + cropx, :]
    else:
        return img[..., starty : starty + cropy, startx : startx + cropx]


def overwrite_gym_box_shape(box: Box, shape) -> Box:
    if box.shape == shape:
        return box
    shape = list(shape) + list(box.shape[len(shape) :])
    low = box.low if np.isscalar(box.low) else np.min(box.low)
    high = box.high if np.isscalar(box.high) else np.max(box.high)
    return Box(low=low, high=high, shape=shape, dtype=box.dtype)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def observations_to_image(observation: Dict, info: Dict, pred=None) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    if "rgb" in observation:
        observation_size = observation["rgb"].shape[0]
        rgb = observation["rgb"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        egocentric_view.append(rgb)

    # draw depth map if observation has depth info
    if "depth" in observation:
        observation_size = observation["depth"].shape[0]
        depth_map = observation["depth"].squeeze() * 255.0
        if not isinstance(depth_map, np.ndarray):
            depth_map = depth_map.cpu().numpy()

        depth_map = depth_map.astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        egocentric_view.append(depth_map)

    assert (
        len(egocentric_view) > 0
    ), "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=1)

    # draw collision
    if "collisions" in info and info["collisions"]["is_collision"]:
        egocentric_view = draw_collision(egocentric_view)

    frame = egocentric_view

    if "top_down_map" in info:
        # top_down_map = info["top_down_map"]["map"]
        # top_down_map = maps.colorize_topdown_map(
        #     top_down_map, info["top_down_map"]["fog_of_war_mask"]
        # )
        map_agent_pos = info["top_down_map"]["agent_map_coord"]
        # top_down_map = maps.draw_agent(
        #     image=top_down_map,
        #     agent_center_coord=map_agent_pos,
        #     agent_rotation=info["top_down_map"]["agent_angle"],
        #     agent_radius_px=top_down_map.shape[0] // 16,
        # )

        top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map"], observation_size
        )

        # if pred is not None:
        #     from habitat.utils.geometry_utils import quaternion_rotate_vector
        #     from habitat_sim.utils.common import quat_from_coeffs
        #     ### code need to be tested
        #     agent_angle = info["top_down_map"]["agent_angle"]
        #     source_rotation = [0.0, np.sin(agent_angle / 2), 0, np.cos(agent_angle / 2)] # format: [x, y, z, w]
        #     source_rotation = quat_from_coeffs(source_rotation)

        #     rounded_pred = np.round(pred[1])
        #     direction_vector_agent = np.array([rounded_pred[1], 0, -rounded_pred[0]])
        #     direction_vector = quaternion_rotate_vector(source_rotation, direction_vector_agent)

        #     grid_size = (
        #         top_down_map.shape[0] / 10000,
        #         top_down_map.shape[1] / 10000,
        #     )
        #     ### code need to be tested
        #     delta_x = int(-direction_vector[0] / grid_size[0])
        #     delta_y = int(direction_vector[2] / grid_size[1])

        #     x = np.clip(map_agent_pos[0] + delta_x, a_min=0, a_max=top_down_map.shape[0])
        #     y = np.clip(map_agent_pos[1] + delta_y, a_min=0, a_max=top_down_map.shape[1])
        #     point_padding = 12
        #     for m in range(x - point_padding, x + point_padding + 1):
        #         for n in range(y - point_padding, y + point_padding + 1):
        #             if np.linalg.norm(np.array([m - x, n - y])) <= point_padding and \
        #                     0 <= m < top_down_map.shape[0] and 0 <= n < top_down_map.shape[1]:
        #                 top_down_map[m, n] = (0, 255, 255)
        #     if np.linalg.norm(rounded_pred) < 1:
        #         assert delta_x == 0 and delta_y == 0

        if top_down_map.shape[0] > top_down_map.shape[1]:
            top_down_map = np.rot90(top_down_map, 1)

        # scale top down map to align with rgb view
        # if pred is None:
        old_h, old_w, _ = top_down_map.shape
        top_down_height = observation_size
        top_down_width = int(float(top_down_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        top_down_map = cv2.resize(
            top_down_map.astype(np.float32),
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )
        # else:
        #     # draw label
        #     CATEGORY_INDEX_MAPPING = {
        #         'chair': 0,
        #         'table': 1,
        #         'picture': 2,
        #         'cabinet': 3,
        #         'cushion': 4,
        #         'sofa': 5,
        #         'bed': 6,
        #         'chest_of_drawers': 7,
        #         'plant': 8,
        #         'sink': 9,
        #         'toilet': 10,
        #         'stool': 11,
        #         'towel': 12,
        #         'tv_monitor': 13,
        #         'shower': 14,
        #         'bathtub': 15,
        #         'counter': 16,
        #         'fireplace': 17,
        #         'gym_equipment': 18,
        #         'seating': 19,
        #         'clothes': 20
        #     }
        #     index2label = {v: k for k, v in CATEGORY_INDEX_MAPPING.items()}
        #     pred_label = index2label[pred[0]]
        #     text_height = int(observation_size * 0.1)

        #     old_h, old_w, _ = top_down_map.shape
        #     top_down_height = observation_size - text_height
        #     top_down_width = int(float(top_down_height) / old_h * old_w)
        #     # cv2 resize (dsize is width first)
        #     top_down_map = cv2.resize(
        #         top_down_map.astype(np.float32),
        #         (top_down_width, top_down_height),
        #         interpolation=cv2.INTER_CUBIC,
        #     )

        #     top_down_map = np.concatenate(
        #         [np.ones([text_height, top_down_map.shape[1], 3], dtype=np.int32) * 255, top_down_map], axis=0)
        #     top_down_map = cv2.putText(top_down_map, 'C_t: ' + pred_label.replace('_', ' '), (10, text_height - 10),
        #                                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 2, cv2.LINE_AA)

        frame = np.concatenate((egocentric_view, top_down_map), axis=1)
    return frame

def find_best_ckpt_idx(event_dir_path, min_step=-1, max_step=10000):
    import tensorflow as tf
    
    events = os.listdir(event_dir_path)
    max_value = 0
    max_index = -1
    for event in events:
        if "events" not in event:
            continue
        iterator = tf.compat.v1.train.summary_iterator(os.path.join(event_dir_path, event))
        for e in iterator:
            if len(e.summary.value) == 0:
                continue
            if not e.summary.value[0].tag.startswith('val'):
                break
            # if 'softspl' not in e.summary.value[0].tag:
            if 'spl' not in e.summary.value[0].tag or 'softspl' in e.summary.value[0].tag:
                continue
            if not min_step <= e.step <= max_step:
                continue
            if len(e.summary.value) > 0 and e.summary.value[0].simple_value > max_value:
                max_value = e.summary.value[0].simple_value
                max_index = e.step

    if max_index == -1:
        print('No max index is found in {}'.format(event_dir_path))
    else:
        print('The best index in {} is {}'.format(event_dir_path, max_index))

    return max_index

def load_pretrained_weights(
    model: nn.Module, 
    checkpoint_file: str, 
    dict_key: str = 'state_dict', 
    prefix: str = 'actor_critic.net.', 
    logger = None
):
    if checkpoint_file is not None and os.path.isfile(checkpoint_file):
        ckpt_dict = torch.load(checkpoint_file, weights_only=False)
        if len(dict_key) > 0 and dict_key in ckpt_dict:
            ckpt_dict = ckpt_dict[dict_key]
        cleaned_ckpt_dict = {k[len(prefix):]: v for k, v in ckpt_dict.items() if prefix in k}
        model_dict = model.state_dict()
        filtered_dict = {}
        missing_keys = []
        unmatched_keys = []
        unexpected_keys = []
        for key, value in model_dict.items():
            if key in cleaned_ckpt_dict:
                if value.shape == cleaned_ckpt_dict[key].shape:
                    filtered_dict[key] = cleaned_ckpt_dict[key]
                else:
                    if logger is not None:
                        logger.info(f"Size mismatch for {key}: model {value.shape} != checkpoint {cleaned_ckpt_dict[key].shape}")
                    unmatched_keys.append(key)
            else:
                missing_keys.append(key)
        
        for key in cleaned_ckpt_dict.keys():
            if key not in model_dict:
                unexpected_keys.append(key)
        
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        if logger is not None:
            logger.info(f"{model.__class__.__name__} ===> successfully loaded weights from {checkpoint_file}, "
                    f"missing keys: {len(missing_keys)}, "
                    f"unexpected keys: {len(unexpected_keys)}, "
                    f"unmatched keys: {len(unmatched_keys)}")
            if len(missing_keys) > 0:
                logger.info(f"Missing keys: {missing_keys}")
            if len(unexpected_keys) > 0:
                logger.info(f"Unexpected keys: {unexpected_keys}")
    else:
        if logger is not None:
            logger.info(f"{model.__class__.__name__} ===> checkpoint file is missing, unable to load weights!")

def format_number(num):
    units = ['', 'K', 'M', 'G', 'T', 'P']
    i = 0
    while num >= 1000 and i < len(units)-1:
        num /= 1000
        i += 1
    if i == 0:
        return f"{num:>3d}"
    else:
        return f"{num:.3f} {units[i]}"
