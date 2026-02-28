#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""Implements evaluation of ``habitat.Agent`` inside ``habitat.Env``.
``habitat.Benchmark`` creates a ``habitat.Env`` which is specified through
the ``config_env`` parameter in constructor. The evaluation is task agnostic
and is implemented through metrics defined for ``habitat.EmbodiedTask``.
"""

from collections import defaultdict
from typing import Dict, Optional

from tqdm import tqdm
import savnce
from habitat import Config
from habitat.core.agent import Agent
# from habitat.core.env import Env
from savnce_baselines.common.environments import AudioNavRLEnv
from habitat.datasets import make_dataset
import logging

class Benchmark:
    r"""Benchmark for evaluating agents in environments.
    """

    def __init__(self, task_config: Optional[Config] = None, logger: logging.Logger = None) -> None:
        r"""..

        :param task_config: config to be used for creating the environment
        """
        dummy_config = Config()
        dummy_config.CONTINUOUS = True
        dummy_config.RL = Config()
        dummy_config.RL.SLACK_REWARD = -0.01
        dummy_config.RL.SUCCESS_REWARD = 10
        dummy_config.RL.WITH_TIME_PENALTY = True
        dummy_config.RL.DISTANCE_REWARD_SCALE = 1
        dummy_config.RL.WITH_DISTANCE_REWARD = True
        dummy_config.TASK_CONFIG = task_config
        dummy_config.freeze()
        self.logger = logger
        logger.info(f"dummy_config:\n{dummy_config}")

        dataset = make_dataset(id_dataset=task_config.DATASET.TYPE, config=task_config.DATASET)
        self._env = AudioNavRLEnv(config=dummy_config, dataset=dataset)

    def evaluate(
        self, agent: Agent, num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env.episodes)
                )
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        self.logger.info(f"Evaluating agent: {agent.__class__.__name__}")
        self.logger.info(f"Number of episodes: {num_episodes}")

        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0
        reward_episodes = 0
        step_episodes = 0
        success_count = 0
        for count_episodes in tqdm(range(num_episodes)):
            agent.reset()
            observations = self._env.reset()
            episode_reward = 0

            while not self._env.habitat_env.episode_over:
                action = agent.act(observations)
                observations, reward, done, info = self._env.step(**action)
                self.logger.debug("Reward: {}".format(reward))
                if done:
                    self.logger.debug('Episode reward: {}'.format(episode_reward))
                episode_reward += reward
                step_episodes += 1

            metrics = self._env.habitat_env.get_metrics()
            msg = f"episode: {count_episodes + 1}/{num_episodes}, "
            for m, v in metrics.items():
                agg_metrics[m] += v
                msg += f"{m}: {round(v, 3) if isinstance(v, float) else v}, "
            reward_episodes += episode_reward
            success_count += metrics['spl'] > 0
            self.logger.info(msg[:-2])
        avg_metrics = {k: v / num_episodes for k, v in agg_metrics.items()}
        self.logger.info("Average reward: {} in {} episodes".format(reward_episodes / num_episodes, num_episodes))
        self.logger.info("Average episode steps: {}".format(step_episodes / num_episodes))
        self.logger.info('Success rate: {}'.format(success_count / num_episodes))
        for k, v in avg_metrics.items():
            self.logger.info(f"Average {k}: {v:.3f}")
        return avg_metrics
