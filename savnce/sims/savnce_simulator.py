# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modifications Copyright (c) 2026 Yichen Zeng, Wuhan University, Email: zengyichen@whu.edu.cn
# Description: Adapted for semantic audio-visual navigation in continuous environment (SAVN-CE).

import glob
from typing import Any, List, Optional
from abc import ABC
import logging
import os

import librosa
from scipy.signal import fftconvolve
import numpy as np
from gym import spaces

from habitat import config
from habitat.core.registry import registry
import habitat_sim
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSimSensor, overwrite_config
from habitat.core.simulator import (
    AgentState,
    Config,
    Observations,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
)


@registry.register_simulator()
class SAVNCE_DummySimulator(Simulator, ABC):
    r"""Changes made to simulator wrapper over habitat-sim

    This simulator is without audio observations.
    Args:
        config: configuration for initializing the simulator.
    """

    def action_space_shortest_path(self, source: AgentState, targets: List[AgentState], agent_id: int = 0) -> List[
            ShortestPathPoint]:
        pass

    def __init__(self, config: Config) -> None:
        self.config = self.habitat_config = config
        agent_config = self._get_agent_config()
        sim_sensors = []
        for sensor_name in agent_config.SENSORS:
            sensor_cfg = getattr(self.config, sensor_name)
            sensor_type = registry.get_sensor(sensor_cfg.TYPE)

            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            sim_sensors.append(sensor_type(sensor_cfg))

        self._sensor_suite = SensorSuite(sim_sensors)
        self.sim_config = self.create_sim_config(self._sensor_suite)
        self._current_scene = self.sim_config.sim_cfg.scene_id
        self._action_space = spaces.Discrete(
            len(self.sim_config.agents[0].action_space)
        )
        self._prev_sim_obs = None
        self._sound_onset_step = None
        self._sound_offset_step = None
        self._episode_step_count = None
        self._is_episode_active = None
        self._previous_step_collided = False

        self._step_time = self.config.STEP_TIME   
        self._max_episode_steps = 500
        self._sim = habitat_sim.Simulator(config=self.sim_config)

    def create_sim_config(
        self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        sim_config = habitat_sim.SimulatorConfiguration()
        # Check if Habitat-Sim is post Scene Config Update
        if not hasattr(sim_config, "scene_id"):
            raise RuntimeError(
                "Incompatible version of Habitat-Sim detected, please upgrade habitat_sim"
            )
        overwrite_config(
            config_from=self.config.HABITAT_SIM_V0,
            config_to=sim_config,
            # Ignore key as it gets propogated to sensor below
            ignore_keys={"gpu_gpu"},
        )
        sim_config.scene_id = self.config.SCENE
        sim_config.enable_physics = False
        # sim_config.scene_dataset_config_file = 'data/scene_datasets/mp3d/mp3d.scene_dataset_config.json'
        # sim_config.scene_dataset_config_file = 'data/scene_datasets/replica/replica.scene_dataset_config.json'
        agent_config = habitat_sim.AgentConfiguration()
        overwrite_config(
            config_from=self._get_agent_config(),
            config_to=agent_config,
            # These keys are only used by Hab-Lab
            ignore_keys={
                "is_set_start_state",
                # This is the Sensor Config. Unpacked below
                "sensors",
                "start_position",
                "start_rotation",
                "goal_position",
                "onset",
                "offset",
                "duration",
                "sound_id",
                "distractor_sound_id",
                "distractor_position",
                "mass",
                "linear_acceleration",
                "angular_acceleration",
                "linear_friction",
                "angular_friction",
                "coefficient_of_restitution",
            },
        )

        sensor_specifications = []
        for sensor in _sensor_suite.sensors.values():
            assert isinstance(sensor, HabitatSimSensor)
            sim_sensor_cfg = sensor._get_default_spec()  # type: ignore[misc]
            overwrite_config(
                config_from=sensor.config,
                config_to=sim_sensor_cfg,
                # These keys are only used by Hab-Lab
                # or translated into the sensor config manually
                ignore_keys=sensor._config_ignore_keys,
                # TODO consider making trans_dict a sensor class var too.
                trans_dict={
                    "sensor_model_type": lambda v: getattr(
                        habitat_sim.FisheyeSensorModelType, v
                    ),
                    "sensor_subtype": lambda v: getattr(
                        habitat_sim.SensorSubType, v
                    ),
                },
            )
            sim_sensor_cfg.uuid = sensor.uuid
            sim_sensor_cfg.resolution = list(
                sensor.observation_space.shape[:2]
            )

            # TODO(maksymets): Add configure method to Sensor API to avoid
            # accessing child attributes through parent interface
            # We know that the Sensor has to be one of these Sensors
            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type
            sim_sensor_cfg.gpu2gpu_transfer = (
                self.config.HABITAT_SIM_V0.GPU_GPU
            )
            sensor_specifications.append(sim_sensor_cfg)

        agent_config.sensor_specifications = sensor_specifications
        agent_config.action_space = registry.get_action_space_configuration(
            self.config.ACTION_SPACE_CONFIG
        )(self.config).get()

        return habitat_sim.Configuration(sim_config, [agent_config])

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    def _update_agents_state(self) -> bool:
        is_updated = False
        for agent_id, _ in enumerate(self.config.AGENTS):
            agent_cfg = self._get_agent_config(agent_id)
            if agent_cfg.IS_SET_START_STATE:
                self.set_agent_state(
                    agent_cfg.START_POSITION,
                    agent_cfg.START_ROTATION,
                    agent_id,
                )
                is_updated = True

        return is_updated

    def _get_agent_config(self, agent_id: Optional[int] = None) -> Any:
        if agent_id is None:
            agent_id = self.config.DEFAULT_AGENT_ID
        agent_name = self.config.AGENTS[agent_id]
        agent_config = getattr(self.config, agent_name)
        return agent_config

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        return self._sim.get_agent(agent_id).get_state()

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        agent = self._sim.get_agent(agent_id)
        new_state = self.get_agent_state(agent_id)
        new_state.position = position
        new_state.rotation = rotation

        # NB: The agent state also contains the sensor states in _absolute_
        # coordinates. In order to set the agent's body to a specific
        # location and have the sensors follow, we must not provide any
        # state for the sensors. This will cause them to follow the agent's
        # body
        new_state.sensor_states = {}
        agent.set_state(new_state, reset_sensors)
        return True
    
    @property
    def current_scene_name(self):
        # config.SCENE (_current_scene) looks like 'data/scene_datasets/replica/office_1/habitat/mesh_semantic.ply'
        return self._current_scene.split('/')[3]
        
    @property
    def is_silent(self):
        return not (self._sound_onset_step <= self._episode_step_count < self._sound_offset_step)

    @property
    def pathfinder(self):
        return self._sim.pathfinder

    def get_agent(self, agent_id):
        return self._sim.get_agent(agent_id)

    def reconfigure(self, config: Config) -> None:
        # was called before the start of the episode
        logging.debug('Reconfigure')
        self.config = config
        if self.config.AUDIO.EVERLASTING:
            self._sound_onset_step = 0
            self._sound_offset_step = self._max_episode_steps
        else:
            self._sound_onset_step = int(self.config.AGENT_0.ONSET / self._step_time)
            self._sound_offset_step = int(self.config.AGENT_0.OFFSET / self._step_time)  
        logging.debug(f'Sound onset step: {self._sound_onset_step} and offset step: {self._sound_offset_step}')
        is_same_scene = config.SCENE == self._current_scene
        if not is_same_scene:
            self._current_scene = config.SCENE
            # logging.debug('Current scene: {}'.format(self.current_scene_name))

            self._sim.close()
            del self._sim
            self.sim_config = self.create_sim_config(self._sensor_suite)
            self._sim = habitat_sim.Simulator(self.sim_config)
            logging.debug(f'Loaded scene {self.current_scene_name}')

        self._update_agents_state()
        # reset some variables when the episode is reset
        self._episode_step_count = 0

    def reset(self):
        logging.debug('Reset simulation')
        sim_obs = self._sim.reset()
        if self._update_agents_state():
            sim_obs = self._sim.get_sensor_observations()

        self._is_episode_active = True
        self._prev_sim_obs = sim_obs
        self._previous_step_collided = False
        # Encapsule data under Observations class
        observations = self._sensor_suite.get_observations(sim_obs)

        return observations

    def step(self, action):
        """
        All angle calculations in this function is w.r.t habitat coordinate frame, on X-Z plane
        where +Y is upward, -Z is forward and +X is rightward.
        Angle 0 corresponds to +X, angle 90 corresponds to +y and 290 corresponds to 270.

        :param action: action to be taken
        :return:
        Dict of observations
        """
        assert self._is_episode_active, (
            "episode is not active, environment not RESET or "
            "STOP action called previously"
        )
        self._episode_step_count += 1
        sim_obs = self._sim.step(action)
        self._prev_sim_obs = sim_obs
        observations = self._sensor_suite.get_observations(sim_obs)
        logging.debug(f'Step simulation, episode step count: {self._episode_step_count}')
        return observations

    def geodesic_distance(self, position_a, position_bs, episode=None):
        if episode is None or episode._shortest_path_cache is None:
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_ends = np.array(
                [np.array(position_bs[0], dtype=np.float32)]
            )
        else:
            path = episode._shortest_path_cache

        path.requested_start = np.array(position_a, dtype=np.float32)

        self.pathfinder.find_path(path)

        if episode is not None:
            episode._shortest_path_cache = path

        return path.geodesic_distance

    @property
    def previous_step_collided(self):
        return self._previous_step_collided

    def seed(self, seed):
        self._sim.seed(seed)

    def get_observations_at(
            self,
            position: Optional[List[float]] = None,
            rotation: Optional[List[float]] = None,
            keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        current_state = self.get_agent_state()
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(
                position, rotation, reset_sensors=False
            )

        if success:
            sim_obs = self._sim.get_sensor_observations()

            self._prev_sim_obs = sim_obs

            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    reset_sensors=False,
                )
            return observations
        else:
            return None

    def get_straight_shortest_path_points(self, position_a, position_b):
        path = habitat_sim.ShortestPath()
        path.requested_start = position_a
        path.requested_end = position_b
        self.pathfinder.find_path(path)
        return path.points

    def make_greedy_follower(self, *args, **kwargs):
        return self._sim.make_greedy_follower(*args, **kwargs)


@registry.register_simulator()
class SAVNCE_Simulator(Simulator, ABC):
    r"""Changes made to simulator wrapper over habitat-sim

    This simulator is adapted for semantic audio-visual navigation in continuous environment (SAVN-CE). 
    1. The agent can move anywhere navigable in the continuous environment and collect audio-visual observations rather than move among predefined nodes, but the action space is still defined discretely.
    3. The action space is forward 0.25m, turn left 15 degrees, turn right 15 degrees and stop.
    4. The heavy precomputed binaural RIRs are not needed anymore, but the simulation would be slower due to the RIR computation on the fly.
    Args:
        config: configuration for initializing the simulator.
    """

    def action_space_shortest_path(self, source: AgentState, targets: List[AgentState], agent_id: int = 0) -> List[
            ShortestPathPoint]:
        pass

    def __init__(self, config: Config) -> None:
        self.config = self.habitat_config = config
        agent_config = self._get_agent_config()
        sim_sensors = []
        for sensor_name in agent_config.SENSORS:
            sensor_cfg = getattr(self.config, sensor_name)
            sensor_type = registry.get_sensor(sensor_cfg.TYPE)

            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            sim_sensors.append(sensor_type(sensor_cfg))

        self._sensor_suite = SensorSuite(sim_sensors)
        self.sim_config = self.create_sim_config(self._sensor_suite)
        self._current_scene = self.sim_config.sim_cfg.scene_id
        self._action_space = spaces.Discrete(
            len(self.sim_config.agents[0].action_space)
        )
        self._prev_sim_obs = None

        self._current_sound = None
        self._sound_onset_step = None
        self._sound_offset_step = None
        self._source_sound_dict = dict()
        self._episode_step_count = None
        self._is_episode_active = None
        self._previous_step_collided = False

        self._audio_distractor_sensor_enabled = False
        self._max_episode_steps = 500
        self._audio_mic_type = self.config.AUDIO.MIC_TYPE
        self._sampling_rate = self.config.AUDIO.RIR_SAMPLING_RATE
        self._step_time = self.config.STEP_TIME
        self._num_samples_per_step = int(self._sampling_rate * self._step_time)
        self._audio_goal_sensor_uuid = "audio_goal_sensor"
        self._audio_distractor_sensor_uuid = "audio_distractor_sensor"
        self._sim = habitat_sim.Simulator(config=self.sim_config)
        self.add_acoustic_config(self._audio_mic_type, self._audio_goal_sensor_uuid)
        self._audio_index = 0
        self._current_sound_sample_index = 0
        self._default_audio_length = self._sampling_rate * 15
        # ensure to have enough space for the reverb from the previous steps
        self._filtered_source_signal = np.zeros(
            (
                self._num_samples_per_step * (self._max_episode_steps + 20),
                self._audio_channel_count
            ),
            dtype=np.float32
        )
        if self.config.AUDIO.HAS_DISTRACTOR_SOUND:
            self.add_acoustic_config(self._audio_mic_type, self._audio_distractor_sensor_uuid)
            self._audio_distractor_sensor_enabled = True
            self._current_distractor_sound = None
            self._current_distractor_sample_index = 0
            self._distractor_sound_dict = dict()
            # ensure to have enough space for the reverb from the previous steps
            self._filtered_distractor_signal = np.zeros(
                (
                    self._num_samples_per_step * (self._max_episode_steps + 20),
                    self._audio_channel_count
                ),
                dtype=np.float32
            )
        if self.config.AUDIO.HAS_NOISE_SOUND:
            self._current_noise_sound = None
            self._current_noise_sample_index = 0
            self._default_SNR = 20 # dB
            self._noise_factor = np.power(10, -self._default_SNR / 20)
            self._noise_sound_dict = dict()
            self._load_noise_sounds()
        self._has_computed_audiogoal = False
        self._current_audiogoal = None

    def add_acoustic_config(self, mic_type, sensor_uuid):
        audio_sensor_spec = habitat_sim.AudioSensorSpec()
        if mic_type == "mono":
            audio_sensor_spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Ambisonics
            audio_sensor_spec.channelLayout.channelCount = 1
            self._audio_channel_count = 1
        elif mic_type == "binaural":
            audio_sensor_spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
            audio_sensor_spec.channelLayout.channelCount = 2
            self._audio_channel_count = 2
        elif mic_type == "ambisonics":
            audio_sensor_spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Ambisonics
            audio_sensor_spec.channelLayout.channelCount = 4
            self._audio_channel_count = 4
        else:
            raise ValueError(f"Invalid audio type: {mic_type}")
        audio_sensor_spec.uuid = sensor_uuid
        audio_sensor_spec.enableMaterials = False
        audio_sensor_spec.acousticsConfig.sampleRate = self._sampling_rate
        audio_sensor_spec.acousticsConfig.threadCount = self.config.AUDIO.THREADS
        audio_sensor_spec.acousticsConfig.indirectRayCount = 500
        audio_sensor_spec.acousticsConfig.temporalCoherence = True
        audio_sensor_spec.acousticsConfig.transmission = True
        self._sim.add_sensor(audio_sensor_spec)

    def create_sim_config(
        self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        sim_config = habitat_sim.SimulatorConfiguration()
        # Check if Habitat-Sim is post Scene Config Update
        if not hasattr(sim_config, "scene_id"):
            raise RuntimeError(
                "Incompatible version of Habitat-Sim detected, please upgrade habitat_sim"
            )
        overwrite_config(
            config_from=self.config.HABITAT_SIM_V0,
            config_to=sim_config,
            # Ignore key as it gets propogated to sensor below
            ignore_keys={"gpu_gpu"},
        )
        sim_config.scene_id = self.config.SCENE
        sim_config.enable_physics = False
        # sim_config.scene_dataset_config_file = 'data/scene_datasets/mp3d/mp3d.scene_dataset_config.json'
        # sim_config.scene_dataset_config_file = 'data/scene_datasets/replica/replica.scene_dataset_config.json'
        agent_config = habitat_sim.AgentConfiguration()
        overwrite_config(
            config_from=self._get_agent_config(),
            config_to=agent_config,
            # These keys are only used by Hab-Lab
            ignore_keys={
                "is_set_start_state",
                # This is the Sensor Config. Unpacked below
                "sensors",
                "start_position",
                "start_rotation",
                "goal_position",
                "onset",
                "offset",
                "duration",
                "sound_id",
                "distractor_sound_id",
                "distractor_position",
                "mass",
                "linear_acceleration",
                "angular_acceleration",
                "linear_friction",
                "angular_friction",
                "coefficient_of_restitution",
            },
        )

        sensor_specifications = []
        for sensor in _sensor_suite.sensors.values():
            assert isinstance(sensor, HabitatSimSensor)
            sim_sensor_cfg = sensor._get_default_spec()  # type: ignore[misc]
            overwrite_config(
                config_from=sensor.config,
                config_to=sim_sensor_cfg,
                # These keys are only used by Hab-Lab
                # or translated into the sensor config manually
                ignore_keys=sensor._config_ignore_keys,
                # TODO consider making trans_dict a sensor class var too.
                trans_dict={
                    "sensor_model_type": lambda v: getattr(
                        habitat_sim.FisheyeSensorModelType, v
                    ),
                    "sensor_subtype": lambda v: getattr(
                        habitat_sim.SensorSubType, v
                    ),
                },
            )
            sim_sensor_cfg.uuid = sensor.uuid
            sim_sensor_cfg.resolution = list(
                sensor.observation_space.shape[:2]
            )

            # TODO(maksymets): Add configure method to Sensor API to avoid
            # accessing child attributes through parent interface
            # We know that the Sensor has to be one of these Sensors
            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type
            sim_sensor_cfg.gpu2gpu_transfer = (
                self.config.HABITAT_SIM_V0.GPU_GPU
            )
            sensor_specifications.append(sim_sensor_cfg)

        agent_config.sensor_specifications = sensor_specifications
        agent_config.action_space = registry.get_action_space_configuration(
            self.config.ACTION_SPACE_CONFIG
        )(self.config).get()

        return habitat_sim.Configuration(sim_config, [agent_config])

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    def _update_agents_state(self) -> bool:
        is_updated = False
        for agent_id, _ in enumerate(self.config.AGENTS):
            agent_cfg = self._get_agent_config(agent_id)
            if agent_cfg.IS_SET_START_STATE:
                self.set_agent_state(
                    agent_cfg.START_POSITION,
                    agent_cfg.START_ROTATION,
                    agent_id,
                )
                is_updated = True

        return is_updated

    def _get_agent_config(self, agent_id: Optional[int] = None) -> Any:
        if agent_id is None:
            agent_id = self.config.DEFAULT_AGENT_ID
        agent_name = self.config.AGENTS[agent_id]
        agent_config = getattr(self.config, agent_name)
        return agent_config

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        return self._sim.get_agent(agent_id).get_state()

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        agent = self._sim.get_agent(agent_id)
        new_state = self.get_agent_state(agent_id)
        new_state.position = position
        new_state.rotation = rotation

        # NB: The agent state also contains the sensor states in _absolute_
        # coordinates. In order to set the agent's body to a specific
        # location and have the sensors follow, we must not provide any
        # state for the sensors. This will cause them to follow the agent's
        # body
        new_state.sensor_states = {}
        agent.set_state(new_state, reset_sensors)
        return True

    @property
    def source_sound_dir(self):
        return self.config.AUDIO.SOURCE_SOUND_DIR

    @property
    def distractor_sound_dir(self):
        return self.config.AUDIO.DISTRACTOR_SOUND_DIR

    @property
    def noise_sound_dir(self):
        return self.config.AUDIO.NOISE_SOUND_DIR

    @property
    def current_scene_name(self):
        # config.SCENE (_current_scene) looks like 'data/scene_datasets/replica/office_1/habitat/mesh_semantic.ply'
        return self._current_scene.split('/')[3]

    @property
    def current_source_sound(self):
        return self._source_sound_dict[self._current_sound]

    @property
    def current_distractor_sound(self):
        return self._distractor_sound_dict[self._current_distractor_sound]
    
    @property
    def current_noise_sound(self):
        return self._noise_sound_dict[self._current_noise_sound]
    
    @property
    def is_silent(self):
        return not (self._sound_onset_step <= self._episode_step_count < self._sound_offset_step)

    @property
    def pathfinder(self):
        return self._sim.pathfinder

    def get_agent(self, agent_id):
        return self._sim.get_agent(agent_id)

    def reconfigure(self, config: Config) -> None:
        # was called before the start of the episode
        logging.debug('Reconfigure')
        self.config = config
        if self.config.AUDIO.EVERLASTING:
            self._sound_onset_step = 0
            self._sound_offset_step = self._max_episode_steps
        else:
            self._sound_onset_step = int(self.config.AGENT_0.ONSET / self._step_time)
            self._sound_offset_step = int(self.config.AGENT_0.OFFSET / self._step_time)  
        logging.debug(f'Sound onset step: {self._sound_onset_step} and offset step: {self._sound_offset_step}')
        is_same_sound = config.AGENT_0.SOUND_ID == self._current_sound
        if not is_same_sound:
            self._current_sound = self.config.AGENT_0.SOUND_ID
            self._load_single_source_sound()
            logging.debug(f'Current sound: {self._current_sound}')
        if self._audio_distractor_sensor_enabled:
            is_same_distractor_sound = self.config.AGENT_0.DISTRACTOR_SOUND_ID == self._current_distractor_sound
            if not is_same_distractor_sound:
                self._current_distractor_sound = self.config.AGENT_0.DISTRACTOR_SOUND_ID
                self._load_single_distractor_sound()
                logging.debug(f'Current distractor sound: {self._current_distractor_sound}')
        if self.config.AUDIO.HAS_NOISE_SOUND:
            self._current_noise_sound = np.random.choice(list(self._noise_sound_dict.keys()))
            logging.debug(f'Current noise sound: {self._current_noise_sound}')
        is_same_scene = config.SCENE == self._current_scene
        if not is_same_scene:
            self._current_scene = config.SCENE
            # logging.debug('Current scene: {}'.format(self.current_scene_name))

            self._sim.close()
            del self._sim
            self.sim_config = self.create_sim_config(self._sensor_suite)
            self._sim = habitat_sim.Simulator(self.sim_config)
            self.add_acoustic_config(self._audio_mic_type, self._audio_goal_sensor_uuid)
            audio_sensor = self._sim.get_agent(0)._sensors[self._audio_goal_sensor_uuid]
            audio_sensor.setAudioMaterialsJSON("data/mp3d_material_config.json")
            if self._audio_distractor_sensor_enabled:
                self.add_acoustic_config(self._audio_mic_type, self._audio_distractor_sensor_uuid)
                audio_sensor = self._sim.get_agent(0)._sensors[self._audio_distractor_sensor_uuid]
                audio_sensor.setAudioMaterialsJSON("data/mp3d_material_config.json")
            logging.debug(f'Loaded scene {self.current_scene_name}')

        self._update_agents_state()
        # reset some variables when the episode is reset
        self._episode_step_count = 0
        self._audio_index = 0
        audio_sensor = self._sim.get_agent(0)._sensors[self._audio_goal_sensor_uuid]
        # 1.5 is the offset for the height
        audio_sensor.setAudioSourceTransform(np.array(self.config.AGENT_0.GOAL_POSITION) + np.array([0, 1.5, 0]))
        self._filtered_source_signal = np.zeros(
            (
                self._num_samples_per_step * (self._max_episode_steps + 20),
                self._audio_channel_count
            ),
            dtype=np.float32
        )
        self._current_sound_sample_index = np.random.randint(self._num_samples_per_step)
        if self._audio_distractor_sensor_enabled:
            audio_sensor = self._sim.get_agent(0)._sensors[self._audio_distractor_sensor_uuid]
            audio_sensor.setAudioSourceTransform(np.array(self.config.AGENT_0.DISTRACTOR_POSITION) + np.array([0, 1.5, 0]))
            self._filtered_distractor_signal = np.zeros(
                (
                    self._num_samples_per_step * (self._max_episode_steps + 20),
                    self._audio_channel_count
                ),
                dtype=np.float32
            )
            self._current_distractor_sample_index = np.random.randint(self._num_samples_per_step)
        if self.config.AUDIO.HAS_NOISE_SOUND:
            self._current_noise_sample_index = np.random.randint(self._num_samples_per_step)
        self._has_computed_audiogoal = False

    def reset(self):
        logging.debug('Reset simulation')
        sim_obs = self._sim.reset()
        if self._update_agents_state():
            sim_obs = self._sim.get_sensor_observations()

        self._is_episode_active = True
        self._prev_sim_obs = sim_obs
        self._previous_step_collided = False
        # Encapsule data under Observations class
        observations = self._sensor_suite.get_observations(sim_obs)

        return observations

    def step(self, action):
        """
        All angle calculations in this function is w.r.t habitat coordinate frame, on X-Z plane
        where +Y is upward, -Z is forward and +X is rightward.
        Angle 0 corresponds to +X, angle 90 corresponds to +y and 290 corresponds to 270.

        :param action: action to be taken
        :return:
        Dict of observations
        """
        assert self._is_episode_active, (
            "episode is not active, environment not RESET or "
            "STOP action called previously"
        )
        self._episode_step_count += 1
        self._audio_index = self._episode_step_count * self._num_samples_per_step
        logging.debug(f'Episode step count: {self._episode_step_count}, sample index: {self._audio_index}')
        self._sim._sensors[self._audio_goal_sensor_uuid]._episode_step = self._episode_step_count
        self._current_sound_sample_index = int(self._current_sound_sample_index + 
                                               self._num_samples_per_step) % self.current_source_sound.shape[0]
        logging.debug(f'Current sound sample index: {self._current_sound_sample_index}')
        if self._audio_distractor_sensor_enabled:
            self._current_distractor_sample_index = int(self._current_distractor_sample_index + 
                                                        self._num_samples_per_step) % self.current_distractor_sound.shape[0]
            self._sim._sensors[self._audio_distractor_sensor_uuid]._episode_step = self._episode_step_count
            logging.debug(f'Current distractor sample index: {self._current_distractor_sample_index}')
        if self.config.AUDIO.HAS_NOISE_SOUND:
            self._current_noise_sample_index = int(self._current_noise_sample_index + 
                                                    self._num_samples_per_step) % self.current_noise_sound.shape[0]
            logging.debug(f'Current noise sample index: {self._current_noise_sample_index}')
        sim_obs = self._sim.step(action)
        self._prev_sim_obs = sim_obs
        observations = self._sensor_suite.get_observations(sim_obs)
        self._has_computed_audiogoal = False
        return observations

    def _normalize_audio(self, audio_data, target_power=0.001):
        # normalize the audio data to the target power
        if audio_data.ndim == 2:
            if audio_data.shape[0] < audio_data.shape[1]:
                audio_data = audio_data.T
            audio_data = np.mean(audio_data, axis=1)
        raw_energy = audio_data ** 2
        raw_power = np.mean(raw_energy)
        if raw_power == 0:
            return audio_data
        # clip the silent part of the audio data
        clipped_audio_data = audio_data[raw_energy > 0.01 * raw_power]
        if len(clipped_audio_data) == 0:
            return audio_data
        power = np.mean(clipped_audio_data ** 2)
        # normalize the audio data to the target power
        factor = np.sqrt(target_power / power)
        audio_data = np.clip(factor * audio_data, -1, 1)
        return audio_data
    
    def _load_single_source_sound(self):
        if self._current_sound not in self._source_sound_dict:
            audio_data, sr = librosa.load(os.path.join(self.source_sound_dir, self._current_sound),
                                          sr=self._sampling_rate)
            audio_data = self._normalize_audio(audio_data)
            # duplicate the audio data to the default length
            while audio_data.shape[0] < self._default_audio_length:
                audio_data = np.concatenate([audio_data, audio_data], axis=0)
            self._source_sound_dict[self._current_sound] = audio_data

    def _load_single_distractor_sound(self):
        if self._current_distractor_sound not in self._distractor_sound_dict:
            audio_data, sr = librosa.load(os.path.join(self.distractor_sound_dir, self._current_distractor_sound),
                                          sr=self._sampling_rate)
            audio_data = self._normalize_audio(audio_data)
            # duplicate the audio data to the default length
            while audio_data.shape[0] < self._default_audio_length:
                audio_data = np.concatenate([audio_data, audio_data], axis=0)
            self._distractor_sound_dict[self._current_distractor_sound] = audio_data

    def _load_noise_sounds(self):
        noise_sound_files = glob.glob(os.path.join(self.noise_sound_dir, '**/*.wav'), recursive=True)
        for noise_sound_file in noise_sound_files:
            filename = os.path.basename(noise_sound_file)
            if filename not in self._noise_sound_dict:
                audio_data, sr = librosa.load(noise_sound_file, sr=self._sampling_rate)
                audio_data = self._normalize_audio(audio_data)
                # duplicate the audio data to the default length
                while audio_data.shape[0] < self._default_audio_length:
                    audio_data = np.concatenate([audio_data, audio_data], axis=0)
                self._noise_sound_dict[filename] = audio_data

    def _compute_audiogoal(self):
        """
        Given sound_segment with length L1 and rir with length L2, the length of the convolved signal is L1 + L2 - 1.
        _filtered_source_signal and _filtered_distractor_signal are updated only when _episode_step_count
        falls within the goal's active interval [onset_step, offset_step). 
        The shape of the audiogoal is (self._num_samples_per_step, self._audio_channel_count)
        """
        if self._sound_onset_step <= self._episode_step_count < self._sound_offset_step:
            # update _filtered_source_signal with the convolved signal of the current step.
            goal_rir = np.transpose(np.array(self._prev_sim_obs[self._audio_goal_sensor_uuid])) # (rir_length, 2)
            sound_length = self.current_source_sound.shape[0]
            start_index = self._current_sound_sample_index
            end_index = start_index + self._num_samples_per_step
            if end_index > sound_length:
                end_index = start_index + self._num_samples_per_step - sound_length
                sound_segment = np.concatenate([self.current_source_sound[start_index:], self.current_source_sound[: end_index]])
            else:
                sound_segment = self.current_source_sound[start_index: end_index]
            audio_end_index = self._audio_index + self._num_samples_per_step + goal_rir.shape[0] - 1
            for channel in range(self._audio_channel_count):
                self._filtered_source_signal[self._audio_index: audio_end_index, channel] += fftconvolve(sound_segment, goal_rir[:, channel])
            logging.debug(f'Goal RIR: {goal_rir.shape}')
            logging.debug(f'Filtered source signal update index: {self._audio_index}:{audio_end_index}')
            # update _filtered_distractor_signal with the convolved signal of the current step.
            if self._audio_distractor_sensor_enabled :
                distractor_rir = np.transpose(np.array(self._prev_sim_obs[self._audio_distractor_sensor_uuid])) # (rir_length, 2)
                sound_length = self.current_distractor_sound.shape[0]
                start_index = self._current_distractor_sample_index
                end_index = start_index + self._num_samples_per_step
                if end_index > sound_length:
                    end_index = start_index + self._num_samples_per_step - sound_length
                    sound_segment = np.concatenate([self.current_distractor_sound[start_index:], self.current_distractor_sound[: end_index]])
                else:
                    sound_segment = self.current_distractor_sound[start_index: end_index]
                audio_end_index = self._audio_index + self._num_samples_per_step + distractor_rir.shape[0] - 1
                for channel in range(self._audio_channel_count):
                    self._filtered_distractor_signal[self._audio_index: audio_end_index, channel] += fftconvolve(sound_segment, distractor_rir[:, channel])
                logging.debug(f'Distractor RIR: {distractor_rir.shape}')
                logging.debug(f'Filtered distractor signal update index: {self._audio_index}:{audio_end_index}')
        audiogoal = self._filtered_source_signal[self._audio_index: self._audio_index + self._num_samples_per_step, :].copy()
        if self._audio_distractor_sensor_enabled:
            audiogoal += self._filtered_distractor_signal[self._audio_index: self._audio_index + self._num_samples_per_step, :].copy()
        if self.config.AUDIO.HAS_NOISE_SOUND:
            noise_segment = self.current_noise_sound[self._current_noise_sample_index: self._current_noise_sample_index + self._num_samples_per_step, np.newaxis]
            audiogoal += self._noise_factor * noise_segment
        logging.debug(f'Audiogoal: {audiogoal.shape}')
        self._current_audiogoal = audiogoal.astype(np.float32)

    def get_current_audiogoal_observation(self):
        # avoiding duplicate computations of the audiogoal, which cause computation error.
        if not self._has_computed_audiogoal:
            self._compute_audiogoal()
            self._has_computed_audiogoal = True
            logging.debug(f'Current audiogoal is computed')
        return self._current_audiogoal

    def geodesic_distance(self, position_a, position_bs, episode=None):
        if episode is None or episode._shortest_path_cache is None:
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_ends = np.array(
                [np.array(position_bs[0], dtype=np.float32)]
            )
        else:
            path = episode._shortest_path_cache

        path.requested_start = np.array(position_a, dtype=np.float32)

        self.pathfinder.find_path(path)

        if episode is not None:
            episode._shortest_path_cache = path

        return path.geodesic_distance

    @property
    def previous_step_collided(self):
        return self._previous_step_collided

    def seed(self, seed):
        self._sim.seed(seed)

    def get_observations_at(
            self,
            position: Optional[List[float]] = None,
            rotation: Optional[List[float]] = None,
            keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        current_state = self.get_agent_state()
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(
                position, rotation, reset_sensors=False
            )

        if success:
            sim_obs = self._sim.get_sensor_observations()

            self._prev_sim_obs = sim_obs

            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    reset_sensors=False,
                )
            return observations
        else:
            return None

    def get_straight_shortest_path_points(self, position_a, position_b):
        path = habitat_sim.ShortestPath()
        path.requested_start = position_a
        path.requested_end = position_b
        self.pathfinder.find_path(path)
        return path.points

    def make_greedy_follower(self, *args, **kwargs):
        return self._sim.make_greedy_follower(*args, **kwargs)
