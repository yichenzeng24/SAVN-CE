#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Modifications Copyright (c) 2026 Yichen Zeng, Wuhan University, Email: zengyichen@whu.edu.cn
# Description: Adapted for semantic audio-visual navigation in continuous environment (SAVN-CE).

import os
from typing import Any, List, Optional, Type

import attr
import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Sensor, SensorTypes, Simulator
from habitat.tasks.nav.nav import Measure, EmbodiedTask, Success
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
)

@attr.s(auto_attribs=True, kw_only=True)
class SAVNCE_Episode(NavigationEpisode):
    r"""Semantic Audio Goal Navigation Episode with distractor, continuous environment

    :param goals: List of goals
    :param distractor: Distractor object
    """
    goals: List["SAVNCE_AudioGoal"] = attr.ib(
        default=None,
        validator=not_none_validator,
        on_setattr=Episode._reset_shortest_path_cache_hook,
    )
    distractor: "SAVNCE_AudioDistractor" = attr.ib(
        default=None,
    )
    oracle_actions: List[int] = attr.ib(
        default=None,
    )

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals
        """
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


@attr.s(auto_attribs=True, kw_only=True)
class SAVNCE_AudioGoal(NavigationGoal):
    r"""Object goal provides information about an object that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        object_id: id that can be used to retrieve object from the semantic
        scene annotation
        object_name: name of the object
        object_category: object category name usually similar to scene semantic
        categories
        room_id: id of a room where object is located, can be used to retrieve
        room from the semantic scene annotation
        room_name: name of the room, where object is located
        view_points: navigable positions around the object with specified
        proximity of the object surface used for navigation metrics calculation.
        The object is visible from these positions.
    """

    object_id: str = attr.ib(default=None, validator=not_none_validator)
    object_name: Optional[str] = None
    object_category: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    view_points: Optional[List[List[float]]] = None
    onset: int = attr.ib(converter=int)
    offset: int = attr.ib(converter=int)
    duration: int = attr.ib(converter=int) 
    sound_id: str = attr.ib(converter=str)


@attr.s(auto_attribs=True, kw_only=True)
class SAVNCE_AudioDistractor:
    r"""Audio distractor is used to distract the agent, which can be specified by position and sound_id.

    Args:
        position: position of the distractor
        sound_id: id of the distractor sound
        onset: onset time of the distractor sound
        offset: offset time of the distractor sound
        duration: duration of the distractor sound
    """
    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    sound_id: str = attr.ib(converter=str)
    # onset: int = attr.ib(converter=int)
    # offset: int = attr.ib(converter=int)
    # duration: int = attr.ib(converter=int)


@registry.register_task(name="SAVNCE_Task")
class SAVNCE_Task(NavigationTask):
    r"""An Object Navigation Task class for a task specific methods.
        Used to explicitly state a type of the task in config.
    """

    def overwrite_sim_config(
            self, sim_config: Any, episode: Type[Episode]
    ) -> Any:
        return merge_sim_episode_config(sim_config, episode)


def merge_sim_episode_config(
        sim_config: Config, episode: Type[Episode]
) -> Any:
    sim_config.defrost()
    # here's where the scene update happens, extract the scene name out of the path
    sim_config.SCENE = episode.scene_id
    sim_config.freeze()
    if (
            episode.start_position is not None
            and episode.start_rotation is not None
    ):
        agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
        agent_cfg = getattr(sim_config, agent_name)
        agent_cfg.defrost()
        agent_cfg.START_POSITION = episode.start_position
        agent_cfg.START_ROTATION = episode.start_rotation
        agent_cfg.GOAL_POSITION = episode.goals[0].position
        agent_cfg.SOUND_ID = episode.goals[0].sound_id
        agent_cfg.ONSET = episode.goals[0].onset
        agent_cfg.OFFSET = episode.goals[0].offset
        agent_cfg.DURATION = episode.goals[0].duration
        if episode.distractor is not None:
            agent_cfg.DISTRACTOR_SOUND_ID = episode.distractor.sound_id
            agent_cfg.DISTRACTOR_POSITION = episode.distractor.position
        else:
            agent_cfg.DISTRACTOR_SOUND_ID = None
            agent_cfg.DISTRACTOR_POSITION = None
        agent_cfg.IS_SET_START_STATE = True
        agent_cfg.freeze()
    return sim_config



