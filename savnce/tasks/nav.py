# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modifications Copyright (c) 2026 Yichen Zeng, Wuhan University, Email: zengyichen@whu.edu.cn
# Description: Adapted for semantic audio-visual navigation in continuous environment (SAVN-CE).

from typing import Any, List, Type, Union

import numpy as np
import torch
import cv2
import librosa
from gym import spaces
from skimage.measure import block_reduce

from habitat.config import Config
from habitat.core.dataset import Episode

from habitat.tasks.nav.nav import DistanceToGoal, Measure, EmbodiedTask, Success
from habitat.core.registry import registry
from habitat.core.simulator import (
    Sensor,
    SensorTypes,
    Simulator,
)
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.tasks.utils import cartesian_to_polar
from savnce.mp3d_utils import CATEGORY_INDEX_MAPPING
from savnce.utils import convert_semantic_object_to_rgb, normalize_distance
from savnce.mp3d_utils import HouseReader


@registry.register_measure
class NormalizedDistanceToGoal(Measure):
    r""" Distance to goal the episode ends
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._start_end_episode_distance = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "normalized_distance_to_goal"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._start_end_episode_distance = episode.info["geodesic_distance"]
        self._metric = None

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        distance_to_goal = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        self._metric = distance_to_goal / self._start_end_episode_distance


@registry.register_measure
class SWS(Measure):
    r"""Success when the audio goal is silent
    """
    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "sws"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._metric = None

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()
        self._metric = ep_success * self._sim.is_silent


@registry.register_measure
class SNA(Measure):
    r"""SPL (Success weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._start_end_num_action = None
        self._agent_num_action = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "sna"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._start_end_num_action = episode.info["num_action"]
        self._agent_num_action = 0
        self._metric = None

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()
        self._agent_num_action += 1

        self._metric = ep_success * (
            self._start_end_num_action
            / max(
                self._start_end_num_action, self._agent_num_action
            )
        )


@registry.register_measure
class NA(Measure):
    r""" Number of actions

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._agent_num_action = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "na"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._agent_num_action = 0
        self._metric = None

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        self._agent_num_action += 1
        self._metric = self._agent_num_action


@registry.register_sensor(name="Collision")
class Collision(Sensor):
    def __init__(
        self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any
    ):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "collision"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=bool
        )

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        return [self._sim.previous_step_collided]
    

@registry.register_sensor(name="EgoMap")
class EgoMap(Sensor):
    r"""Estimates the top-down occupancy based on current depth-map.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: contains the MAP_RESOLUTION, MAP_SIZE, HEIGHT_THRESH fields to
                decide grid-size, extents of the projection, and the thresholds
                for determining obstacles and explored space.
    """

    def __init__(
        self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim

        super().__init__(config=config)

        # Map statistics
        self.map_size = self.config.MAP_SIZE
        self.map_res = self.config.MAP_RESOLUTION

        # Agent height for pointcloud transformation
        self.sensor_height = self.config.POSITION[1]

        # Compute intrinsic matrix
        hfov = float(self._sim.config.DEPTH_SENSOR.HFOV) * np.pi / 180
        self.intrinsic_matrix = np.array([[1 / np.tan(hfov / 2.), 0., 0., 0.],
                                          [0., 1 / np.tan(hfov / 2.), 0., 0.],
                                          [0., 0.,  1, 0],
                                          [0., 0., 0, 1]])
        self.inverse_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)

        # Height thresholds for obstacles
        self.height_thresh = self.config.HEIGHT_THRESH

        # Depth processing
        self.min_depth = float(self._sim.config.DEPTH_SENSOR.MIN_DEPTH)
        self.max_depth = float(self._sim.config.DEPTH_SENSOR.MAX_DEPTH)

        # Pre-compute a grid of locations for depth projection
        W = self._sim.config.DEPTH_SENSOR.WIDTH
        H = self._sim.config.DEPTH_SENSOR.HEIGHT
        self.proj_xs, self.proj_ys = np.meshgrid(
                                          np.linspace(-1, 1, W),
                                          np.linspace(1, -1, H)
                                     )

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "ego_map"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self.config.MAP_SIZE, self.config.MAP_SIZE, 2)
        return spaces.Box(
            low=0,
            high=1,
            shape=sensor_shape,
            dtype=np.uint8,
        )

    def convert_to_pointcloud(self, depth):
        """
        Inputs:
            depth = (H, W, 1) numpy array
        Returns:
            xyz_camera = (N, 3) numpy array for (X, Y, Z) in egocentric world coordinates
        """

        depth_float = depth.astype(np.float32)[..., 0]

        # =========== Convert to camera coordinates ============
        W = depth.shape[1]
        xs = self.proj_xs.reshape(-1)
        ys = self.proj_ys.reshape(-1)
        depth_float = depth_float.reshape(-1)

        # Filter out invalid depths
        max_forward_range = self.map_size * self.map_res
        valid_depths = (depth_float != 0.0) & (depth_float <= max_forward_range)
        xs = xs[valid_depths]
        ys = ys[valid_depths]
        depth_float = depth_float[valid_depths]

        # Unproject
        # negate depth as the camera looks along -Z
        xys = np.vstack((xs * depth_float,
                         ys * depth_float,
                         -depth_float, np.ones(depth_float.shape)))
        inv_K = self.inverse_intrinsic_matrix
        xyz_camera = np.matmul(inv_K, xys).T # XYZ in the camera coordinate system
        xyz_camera = xyz_camera[:, :3] / xyz_camera[:, 3][:, np.newaxis]

        return xyz_camera

    def safe_assign(self, im_map, x_idx, y_idx, value):
        try:
            im_map[x_idx, y_idx] = value
        except IndexError:
            valid_idx1 = np.logical_and(x_idx >= 0, x_idx < im_map.shape[0])
            valid_idx2 = np.logical_and(y_idx >= 0, y_idx < im_map.shape[1])
            valid_idx = np.logical_and(valid_idx1, valid_idx2)
            im_map[x_idx[valid_idx], y_idx[valid_idx]] = value

    def _get_depth_projection(self, sim_depth):
        """
        Project pixels visible in depth-map to ground-plane
        """

        if self._sim.config.DEPTH_SENSOR.NORMALIZE_DEPTH:
            depth = sim_depth * (self.max_depth - self.min_depth) + self.min_depth
        else:
            depth = sim_depth

        XYZ_ego = self.convert_to_pointcloud(depth)

        # Adding agent's height to the point cloud
        XYZ_ego[:, 1] += self.sensor_height

        # Convert to grid coordinate system
        V = self.map_size
        Vby2 = V // 2
        points = XYZ_ego

        grid_x = (points[:, 0] / self.map_res) + Vby2
        grid_y = (points[:, 2] / self.map_res) + V

        # Filter out invalid points
        valid_idx = (grid_x >= 0) & (grid_x <= V-1) & (grid_y >= 0) & (grid_y <= V-1)
        points = points[valid_idx, :]
        grid_x = grid_x[valid_idx].astype(int)
        grid_y = grid_y[valid_idx].astype(int)

        # Create empty maps for the two channels
        obstacle_mat = np.zeros((self.map_size, self.map_size), np.uint8)
        explore_mat = np.zeros((self.map_size, self.map_size), np.uint8)

        # Compute obstacle locations
        high_filter_idx = points[:, 1] < self.height_thresh[1]
        low_filter_idx = points[:, 1] > self.height_thresh[0]
        obstacle_idx = np.logical_and(low_filter_idx, high_filter_idx)

        self.safe_assign(obstacle_mat, grid_y[obstacle_idx], grid_x[obstacle_idx], 1)

        # Compute explored locations
        explored_idx = high_filter_idx
        self.safe_assign(explore_mat, grid_y[explored_idx], grid_x[explored_idx], 1)

        # Smoothen the maps
        kernel = np.ones((3, 3), np.uint8)

        obstacle_mat = cv2.morphologyEx(obstacle_mat, cv2.MORPH_CLOSE, kernel)
        explore_mat = cv2.morphologyEx(explore_mat, cv2.MORPH_CLOSE, kernel)

        # Ensure all expanded regions in obstacle_mat are accounted for in explored_mat
        explore_mat = np.logical_or(explore_mat, obstacle_mat)

        return np.stack([obstacle_mat, explore_mat], axis=2)

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        # convert to numpy array
        ego_map_gt = self._sim.get_egomap_observation()
        if ego_map_gt is None:
            sim_depth = asnumpy(observations['depth'])
            ego_map_gt = self._get_depth_projection(sim_depth)
            self._sim.cache_egomap_observation(ego_map_gt)

        return ego_map_gt


@registry.register_sensor(name="Category")
class Category(Sensor):
    cls_uuid: str = "category"

    def __init__(
        self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any
    ):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(len(CATEGORY_INDEX_MAPPING.keys()),),
            dtype=bool
        )

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        # index = CATEGORY_INDEX_MAPPING[episode.object_category]
        index = CATEGORY_INDEX_MAPPING[episode.goals[0].object_category]
        onehot = np.zeros(len(CATEGORY_INDEX_MAPPING.keys()))
        onehot[index] = 1

        return onehot


@registry.register_sensor(name="CategoryBelief")
class CategoryBelief(Sensor):
    cls_uuid: str = "category_belief"

    def __init__(
        self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any
    ):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(len(CATEGORY_INDEX_MAPPING.keys()),),
            dtype=bool
        )

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        belief = np.zeros(len(CATEGORY_INDEX_MAPPING.keys()))

        return belief


@registry.register_sensor(name="LocationBelief")
class LocationBelief(Sensor):
    cls_uuid: str = "location_belief"

    def __init__(
        self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any
    ):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(2,),
            dtype=bool
        )

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        belief = np.zeros(2)
        return belief
    

@registry.register_sensor(name="MPCAT40Index")
class MPCAT40Index(Sensor):
    def __init__(
        self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any
    ):
        self.config = config
        self._category_mapping = {
                'chair': 3,
                'table': 5,
                'picture': 6,
                'cabinet': 7,
                'cushion': 8,
                'sofa': 10,
                'bed': 11,
                'chest_of_drawers': 13,
                'plant': 14,
                'sink': 15,
                'toilet': 18,
                'stool': 19,
                'towel': 20,
                'tv_monitor': 22,
                'shower': 23,
                'bathtub': 25,
                'counter': 26,
                'fireplace': 27,
                'gym_equipment': 33,
                'seating': 34,
                'clothes': 38
            }
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "mpcat40_index"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=bool
        )

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        index = self._category_mapping[episode.object_category]
        encoding = np.array([index])

        return encoding


@registry.register_sensor(name="SemanticObjectSensor")
class SemanticObjectSensor(Sensor):
    r"""Lists the object categories for each pixel location.

    Args:
        sim: reference to the simulator for calculating task observations.
    """
    cls_uuid: str = "semantic_object"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._current_episode_id = None
        self.mapping = None
        self._initialize_category_mappings()

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _initialize_category_mappings(self):
        self.category_to_task_category_id = {
            'chair': 0,
            'table': 1,
            'picture': 2,
            'cabinet': 3,
            'cushion': 4,
            'sofa': 5,
            'bed': 6,
            'chest_of_drawers': 7,
            'plant': 8,
            'sink': 9,
            'toilet': 10,
            'stool': 11,
            'towel': 12,
            'tv_monitor': 13,
            'shower': 14,
            'bathtub': 15,
            'counter': 16,
            'fireplace': 17,
            'gym_equipment': 18,
            'seating': 19,
            'clothes': 20
        }
        self.category_to_mp3d_category_id = {
            'chair': 3,
            'table': 5,
            'picture': 6,
            'cabinet': 7,
            'cushion': 8,
            'sofa': 10,
            'bed': 11,
            'chest_of_drawers': 13,
            'plant': 14,
            'sink': 15,
            'toilet': 18,
            'stool': 19,
            'towel': 20,
            'tv_monitor': 22,
            'shower': 23,
            'bathtub': 25,
            'counter': 26,
            'fireplace': 27,
            'gym_equipment': 33,
            'seating': 34,
            'clothes': 38
        }
        self.num_task_categories = np.max(
            list(self.category_to_task_category_id.values())
        ) + 1
        self.mp3d_id_to_task_id = np.ones((200, ), dtype=np.int64) * -1
        for k in self.category_to_task_category_id.keys():
            v1 = self.category_to_task_category_id[k]
            v2 = self.category_to_mp3d_category_id[k]
            self.mp3d_id_to_task_id[v2] = v1
        # Map unknown classes to a new category
        self.mp3d_id_to_task_id[
            self.mp3d_id_to_task_id == -1
        ] = self.num_task_categories

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        if self.config.CONVERT_TO_RGB:
            observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.config.HEIGHT, self.config.WIDTH, 3),
                dtype=np.uint8,
            )
        else:
            observation_space = spaces.Box(
                low=np.iinfo(np.uint32).min,
                high=np.iinfo(np.uint32).max,
                shape=(self.config.HEIGHT, self.config.WIDTH),
                dtype=np.uint32,
            )
        return observation_space

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if self._current_episode_id != episode_uniq_id:
            self._current_episode_id = episode_uniq_id
            reader = HouseReader(self._sim._current_scene.replace('.glb', '.house'))
            instance_id_to_mp3d_id = reader.compute_object_to_category_index_mapping()
            self.instance_id_to_mp3d_id = np.array([instance_id_to_mp3d_id[i] for i in range(len(instance_id_to_mp3d_id))])

        # Pre-process semantic observations to remove invalid values
        semantic = np.copy(observations["semantic"])
        semantic[semantic >= self.instance_id_to_mp3d_id.shape[0]] = 0
        # Map from instance id to semantic id
        semantic_object = np.take(self.instance_id_to_mp3d_id, semantic)
        # Map from semantic id to task id
        semantic_object = np.take(self.mp3d_id_to_task_id, semantic_object)
        if self.config.CONVERT_TO_RGB:
            semantic_object = SemanticObjectSensor.convert_semantic_map_to_rgb(
                semantic_object
            )

        return semantic_object

    @staticmethod
    def convert_semantic_map_to_rgb(semantic_map):
        return convert_semantic_object_to_rgb(semantic_map)


@registry.register_sensor(name="PoseSensor")
class PoseSensor(Sensor):
    r"""The agents current location and heading in the coordinate frame defined by the
    episode, i.e. the axis it faces along and the origin is defined by its state at
    t=0. Additionally contains the time-step of the episode.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """
    cls_uuid: str = "pose"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._episode_time = 0
        self._current_episode_id = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(4,),
            dtype=np.float32,
        )

    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        # return np.array([phi], dtype=np.float32)
        return phi

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id != self._current_episode_id:
            self._episode_time = 0.0
            self._current_episode_id = episode_uniq_id

        agent_state = self._sim.get_agent_state()

        origin = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        agent_position_xyz = agent_state.position
        rotation_world_agent = agent_state.rotation

        agent_position_xyz = quaternion_rotate_vector(
            rotation_world_start.inverse(), agent_position_xyz - origin
        )

        agent_heading = self._quat_to_xy_heading(
            rotation_world_agent.inverse() * rotation_world_start
        )

        ep_time = self._episode_time
        self._episode_time += 1.0

        return np.array(
            [-agent_position_xyz[2], agent_position_xyz[0], agent_heading, ep_time],
            dtype=np.float32
        )


@registry.register_sensor
class ProximitySensor(Sensor):
    r"""Sensor for observing the distance to the closest obstacle

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    cls_uuid: str = "proximity"

    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._max_detection_radius = getattr(
            config, "MAX_DETECTION_RADIUS", 2.0
        )
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0.0,
            high=self._max_detection_radius,
            shape=(1,),
            dtype=np.float32,
        )

    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position

        return np.array(
            [
                self._sim.distance_to_closest_obstacle(
                    current_position, self._max_detection_radius
                )
            ],
            dtype=np.float32,
        )


@registry.register_sensor
class OracleActionSensor(Sensor):
    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "oracle_action_sensor"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_oracle_action()

def asnumpy(v):
    if torch.is_tensor(v):
        return v.cpu().numpy()
    elif isinstance(v, np.ndarray):
        return v
    else:
        raise ValueError('Invalid input')


@registry.register_sensor
class AudioGoalSensor(Sensor):
    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "audiogoal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.AUDIO

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._sim._num_samples_per_step, self._sim._audio_channel_count)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_audiogoal_observation()


@registry.register_sensor
class SpectrogramSensor(Sensor):
    cls_uuid: str = "spectrogram"
    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "spectrogram"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.AUDIO

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        spectrogram = self.compute_spectrogram(np.ones((self._sim._num_samples_per_step, self._sim._audio_channel_count)))

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=spectrogram.shape,
            dtype=np.float32,
        )

    @staticmethod
    def compute_spectrogram(audio_data):
        def compute_stft(signal):
            n_fft = 512
            hop_length = 160
            win_length = 400
            stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
            stft = block_reduce(stft, block_size=(4, 1), func=np.mean)
            return stft
        
        spectra = []
        for channel in range(audio_data.shape[1]):
            spectra.append(np.log1p(compute_stft(audio_data[:, channel])))
        spectra = np.stack(spectra) # (n_channels, n_bins, n_frames)

        return spectra

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        audiogoal = self._sim.get_current_audiogoal_observation()
        spectra = self.compute_spectrogram(audiogoal)
        return spectra
    

class AudioFeatureExtractor:
    """
    Base class for audio feature extraction.
    """
    supported_audio_types: List[str] = ["mono", "binaural", "ambisonics"]

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self.mic_type = self._sim._audio_mic_type
        if self.mic_type not in self.supported_audio_types:
            raise ValueError(f"Audio type {self.mic_type} not supported, supported audio types: {self.supported_audio_types}")
        self.num_samples_per_step = self._sim._num_samples_per_step
        self.n_channels = self._sim._audio_channel_count

        self.sr = self._sim.config.AUDIO.RIR_SAMPLING_RATE # Hz
        self.c = 343 # m/s
        self.hop_len_s = 0.01 # seconds
        self.hop_len = int(self.sr * self.hop_len_s) # samples
        self.n_fft = 512 # samples
        self.eps = 1e-8
        self.win_len = self.n_fft # samples
        self.num_fre_bin = self.n_fft // 2 + 1 # bins

    def stft(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute the Short-Time Fourier Transform (STFT) of the input signal.
        :param signal: <np.ndarray: (n_samples, n_channels)>.
        :return: <np.ndarray: (n_channels, n_bins, n_frames)>.
        """
        spectra = []
        # signal = signal / np.max(signal + self.config.eps, keepdims=True)
        for channel in range(signal.shape[1]):
            stft_ch = librosa.stft(
                signal[:, channel], 
                n_fft=self.n_fft, 
                hop_length=self.hop_len,
                win_length=self.win_len, 
                ) # (n_fft // 2 + 1, n_frames)
            spectra.append(stft_ch)
        return np.array(spectra) # (n_channels, n_bins, n_frames)

    def extract(self, audio_input: np.ndarray) -> np.ndarray:
        """
        :param audio_input: <np.ndarray: (n_samples, n_channels)>.
        :return: <np.ndarray: (n_channels, n_bins, n_frames)>.
        """
        raise NotImplementedError("This method is not implemented")

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.AUDIO

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        features = self.extract(np.ones((self.num_samples_per_step, self.n_channels)))

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=features.shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        audio_goal = self._sim.get_current_audiogoal_observation()
        # audio_goal: (num_samples_per_step, n_channels)
        features = self.extract(audio_goal)
        return features
    

@registry.register_sensor
class BinauralFeatureExtractor(AudioFeatureExtractor, Sensor):

    cls_uuid: str = "binaural_extractor"
    supported_audio_types = ["binaural"]

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        AudioFeatureExtractor.__init__(self, *args, sim=sim, config=config, **kwargs)
        Sensor.__init__(self, *args, sim=sim, config=config, **kwargs)

    def extract(self, audio_input: np.ndarray) -> np.ndarray:
        """
        :param audio_input: <np.ndarray: (n_samples, n_channels)>.
        :return: <np.ndarray: (4, n_bins, n_frames)>.
        """
        spec = self.stft(audio_input)[:, 1:, :] # (n_channels, n_bins, n_frames)
        nChs, nBins, n_frames = spec.shape
        features = np.zeros((4, nBins, n_frames))
        features[0, :, :] = np.mean(np.abs(spec), axis=0) # mean magnitude spectrogram
        phase_diff = np.angle(spec[0, :, :]) - np.angle(spec[1, :, :]) # interchannel phase difference
        features[1, :, :] = np.cos(phase_diff) # interchannel cosine phase difference
        features[2, :, :] = np.sin(phase_diff) # interchannel sine phase difference
        left_channel_level = np.abs(spec[0, :, :])
        right_channel_level = np.abs(spec[1, :, :])
        # ILD_dB = 10 * np.log10((left_channel_power + self.eps) / (right_channel_power + self.eps)) # interchannel level difference, in dB
        ILD_dB = np.log10((left_channel_level + self.eps) / (right_channel_level + self.eps)) # interchannel level difference, in dB
        features[3, :, :] = np.clip(ILD_dB, -4, 4) # clip to -40 dB to 40 dB
        return features.astype(np.float32)


@registry.register_sensor
class OraclePositionSensor(Sensor):
    r"""Sensor for oracle goal position observations which are used in SAVN-CE.

    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the OraclePosition sensor. Can contain field for
            GOAL_FORMAT which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.

            Also contains a DIMENSIONALITY field which specifes the number
            of dimensions ued to specify the goal, must be in [2, 3]

    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
        _dimensionality: number of dimensions used to specify the goal
        _everlasting: bool, if True, the goal position is everlasting
    """
    cls_uuid: str = "oracle_position"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim

        self._goal_format = config.GOAL_FORMAT
        assert self._goal_format in ["CARTESIAN", "POLAR"]
        self._dimensionality = config.DIMENSIONALITY
        assert self._dimensionality in [2, 3]
        self._everlasting = config.EVERLASTING
        self._step_time = config.STEP_TIME
        self._goals_offset_step = config.MAX_EPISODE_STEPS
        self._use_normalized_distance = config.USE_NORMALIZED_DISTANCE
        self._max_distance = config.MAX_DISTANCE

        self._episode_step_index = 0
        self._goals_onset_step = 0
        self._current_episode_id = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def _compute_oracle_position(
        self, agent_position, agent_rotation, goal_position
    ):
        delta = goal_position - agent_position
        if self._use_normalized_distance:
            delta = delta / self._max_distance
        if self._goal_format == "POLAR":
            direction_vector = np.array([0, 0, -1])
            heading_vector = quaternion_rotate_vector(agent_rotation, direction_vector)
            global_azi = -np.arctan2(delta[0], -delta[2])
            heading = -np.arctan2(heading_vector[0], -heading_vector[2])
            phi = (global_azi - heading + np.pi) % (2 * np.pi) - np.pi
            
            if self._dimensionality == 2:
                distance = np.sqrt(delta[0] ** 2 + delta[2] ** 2)
                return np.array([distance, phi], dtype=np.float32)
            else:
                theta = np.arccos(heading_vector[1] / np.linalg.norm(heading_vector))
                distance = np.linalg.norm(delta)
                return np.array([distance, phi, theta], dtype=np.float32)
        else:
            if self._dimensionality == 2:
                return np.array([delta[0], -delta[2]], dtype=np.float32)
            else:
                return np.array([delta[0], -delta[2], delta[1]], dtype=np.float32)

    def get_observation(
        self,
        observations,
        episode: Episode,
        *args: Any,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id}_{episode.episode_id}"
        if episode_uniq_id != self._current_episode_id:
            self._episode_step_index = 0
            self._current_episode_id = episode_uniq_id
            self._goals_onset_step = int(episode.goals[0].onset / self._step_time)
            if not self._everlasting:
                self._goals_offset_step = int(episode.goals[0].offset / self._step_time)
        else:
            self._episode_step_index += 1
        if self._goals_onset_step <= self._episode_step_index < self._goals_offset_step:
            agent_state = self._sim.get_agent(0).get_state()
            agent_position = agent_state.position
            agent_rotation = agent_state.rotation
            goal_position = np.array(episode.goals[0].position, dtype=np.float32)

            return self._compute_oracle_position(
                agent_position, agent_rotation, goal_position
            )
        else:
            return np.zeros(self._dimensionality, dtype=np.float32)


@registry.register_sensor
class OracleCategorySensor(Sensor):
    r"""Sensor for oracle goal category observations which are used in SAVN-CE.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.

    Attributes:
        _num_classes: int, number of classes
        _everlasting: bool, if True, the goal category is everlasting
    """
    cls_uuid: str = "oracle_category"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._categories = CATEGORY_INDEX_MAPPING
        self._num_classes = len(self._categories.keys())
        self._everlasting = config.EVERLASTING
        self._step_time = config.STEP_TIME
        self._goals_offset_step = config.MAX_EPISODE_STEPS
        self._episode_step_index = 0
        self._goals_onset_step = 0
        self._current_episode_id = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self._num_classes,),
            dtype=np.float32,
        )

    def get_observation(
        self,
        observations,
        episode: Episode,
        *args: Any,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id}_{episode.episode_id}"
        if episode_uniq_id != self._current_episode_id:
            self._episode_step_index = 0
            self._current_episode_id = episode_uniq_id
            self._goals_onset_step = int(episode.goals[0].onset / self._step_time)
            if not self._everlasting:
                self._goals_offset_step = int(episode.goals[0].offset / self._step_time)
        else:
            self._episode_step_index += 1
        
        onehot = np.zeros(self._num_classes)
        if self._goals_onset_step <= self._episode_step_index < self._goals_offset_step:
            index = CATEGORY_INDEX_MAPPING[episode.goals[0].object_category]
            onehot[index] = 1
        return onehot


@registry.register_sensor
class OracleACCDDOASensor(Sensor):
    r"""Sensor for oracle goal accddoa observations which are used in SAVN-CE.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.

    Attributes:
        _num_classes: int, number of classes
        _everlasting: bool, if True, the goal accddoa is everlasting
    """

    cls_uuid: str = "oracle_accddoa"

    def __init__(
        self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._dimensionality = config.DIMENSIONALITY
        assert self._dimensionality in [2, 3]
        self._goal_format = config.GOAL_FORMAT
        assert self._goal_format in ['POLAR', 'CARTESIAN']
        self._num_classes = len(CATEGORY_INDEX_MAPPING.keys()) # int
        self._step_time = config.STEP_TIME # float
        self._min_distance = config.MIN_DISTANCE # float
        self._max_distance = config.MAX_DISTANCE # float
        self._goals_offset_step = config.MAX_EPISODE_STEPS
        self._everlasting = config.EVERLASTING

        self._episode_step_index = 0
        self._goals_onset_step = 0
        self._current_episode_id = None

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(4 * self._num_classes,),
            dtype=np.float32
        )

    def get_oracle_accddoa_label(self, episode):
        agent_state = self._sim.get_agent(0).get_state()
        agent_position = agent_state.position
        agent_rotation = agent_state.rotation
        goal_position = episode.goals[0].position

        x_labels = np.zeros(self._num_classes) # (num_classes)
        y_labels = np.zeros(self._num_classes) # (num_classes)
        z_labels = np.zeros(self._num_classes) # (num_classes)
        dist_labels = np.zeros(self._num_classes) # (num_classes)
        category_index = CATEGORY_INDEX_MAPPING[episode.goals[0].object_category]
        delta = goal_position - agent_position
        direction_vector = np.array([0, 0, -1])
        heading_vector = quaternion_rotate_vector(agent_rotation, direction_vector)
        global_azi = -np.arctan2(delta[0], -delta[2])
        heading = -np.arctan2(heading_vector[0], -heading_vector[2])
        phi = (global_azi - heading + np.pi) % (2 * np.pi) - np.pi
        # here x-axis is rightward, y-axis is upward, z-axis is backward
        # azi (-pi, pi) -z -> 0, -x -> pi/2, z -> pi, x -> -pi/2
        # ele (0, pi) xoz -> pi/2, y -> 0, -y -> pi
        if self._goal_format == "POLAR":
            if self._dimensionality == 2:
                distance = np.sqrt(delta[0] ** 2 + delta[2] ** 2)
                x_labels[category_index] = phi
                dist_labels[category_index] = normalize_distance(distance, self._min_distance, self._max_distance) # dist
            else:
                theta = np.arccos(heading_vector[1] / np.linalg.norm(heading_vector))
                distance = np.linalg.norm(delta)
                x_labels[category_index] = phi
                y_labels[category_index] = theta
                dist_labels[category_index] = normalize_distance(distance, self._min_distance, self._max_distance) # dist
        else: # CARTESIAN
            if self._dimensionality == 2:
                distance = np.sqrt(delta[0] ** 2 + delta[2] ** 2)
                theta = np.pi / 2
            else:
                distance = np.linalg.norm(delta)
                theta = np.arccos(heading_vector[1] / np.linalg.norm(heading_vector))
            temp_labels = np.sin(theta)
            # here x-axis is rightward, y-axis is forward, z-axis is upward
            # azi (-pi, pi) y -> 0, -x -> pi/2, -y -> pi, x -> -pi/2
            # ele (0, pi) xoy -> pi/2, z -> 0, -z -> pi
            x_labels[category_index] = -temp_labels * np.sin(phi) # doa_x
            y_labels[category_index] = temp_labels * np.cos(phi) # doa_y
            z_labels[category_index] = np.cos(theta) # doa_z
            dist_labels[category_index] = normalize_distance(distance, self._min_distance, self._max_distance) # dist
        accddoa_label = np.concatenate([x_labels, y_labels, z_labels, dist_labels]) # (4 * num_classes,)
        return accddoa_label.astype(np.float32)
    
    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ):
        episode_uniq_id = f"{episode.scene_id}_{episode.episode_id}"
        if episode_uniq_id != self._current_episode_id:
            self._episode_step_index = 0
            self._current_episode_id = episode_uniq_id
            self._goals_onset_step = int(episode.goals[0].onset / self._step_time)
            if not self._everlasting:
                self._goals_offset_step = int(episode.goals[0].offset / self._step_time)
        else:
            self._episode_step_index += 1
        if self._goals_onset_step <= self._episode_step_index < self._goals_offset_step:
            accddoa_label = self.get_oracle_accddoa_label(episode).astype(np.float32)
            return accddoa_label
        else:
            return np.zeros(4 * self._num_classes, dtype=np.float32)
    

@registry.register_sensor(name="GoalDescriptor")
class GoalDescriptor(Sensor):
    r"""Sensor for storing goal descriptor observations which are used in MAGNet.
    It will be updated by the goal descriptor network.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.

    Attributes:
        _num_classes: int, number of classes
        _output_size: int, output size of the goal descriptor network
    """
    cls_uuid: str = "goal_descriptor"

    def __init__(
        self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any
    ):
        self.output_size = config.EMBEDDING_SIZE + config.OUTPUT_SIZE
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self.output_size,),
            dtype=np.float32
        )

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        goal = np.zeros(self.output_size, dtype=np.float32)
        return goal


@registry.register_measure
class DistractorSuccess(Measure):
    r"""Whether or not the agent stopped at distractor.
    """

    cls_uuid: str = "distractor_success"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._metric = None
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def get_distance_to_distractor(self, episode):
        assert episode.distractor is not None, "Distractor is not found in episode"
        return self._sim.geodesic_distance(
            self._sim.get_agent_state().position,
            [episode.distractor.position]
        )

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        distance_to_distractor = self.get_distance_to_distractor(episode)
        if (
            hasattr(task, "is_stop_called")
            and task.is_stop_called  # type: ignore
            and distance_to_distractor < self._config.DISTRACTOR_SUCCESS_DISTANCE
        ):
            self._metric = 1.0
        else:
            self._metric = 0.0