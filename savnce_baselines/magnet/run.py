#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import torch
from savnce_baselines.common.baseline_registry import baseline_registry
from savnce_baselines.magnet.config.default import get_config
from savnce_baselines.common.utils import find_best_ckpt_idx
# os.environ['HABITAT_SIM_LOG'] = 'warning'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "test"],
        # required=True,
        default='train',
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        # required=True,
        default='savnce_baselines/savnce/config/mp3d/rgbd_ddppo_clean_train.yaml',
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--model-dir",
        default='data/models/output',
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Evaluation interval of checkpoints",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action='store_true',
        help="Modify config options from command line"
    )
    parser.add_argument(
        "--prev-ckpt-ind",
        type=int,
        default=-1,
        help="Evaluation interval of checkpoints",
    )
    parser.add_argument(
        "--max-ckpt-ind",
        type=int,
        default=10000,
        help="Evaluation interval of checkpoints",
    )
    parser.add_argument(
        "--eval-best",
        default=False,
        action='store_true',
        help="Modify config options from command line"
    )
    args = parser.parse_args()

    # repo = git.Repo(search_parent_directories=True)
    # logging.info('Current git head hash code: {}'.format(repo.head.object.hexsha))
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir, exist_ok=True)

    if args.eval_best:
        best_ckpt_idx = find_best_ckpt_idx(os.path.join(args.model_dir, 'tb'), max_step=args.max_ckpt_ind)
        best_ckpt_path = os.path.join(args.model_dir, 'data', f'ckpt.{best_ckpt_idx}.pth')
        print(f'Evaluating the best checkpoint: {best_ckpt_path}')
        args.opts += ['EVAL_CKPT_PATH_DIR', best_ckpt_path]

    # run exp
    config = get_config(args.exp_config, args.opts, args.model_dir, args.run_type, args.overwrite)
    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)
    torch.set_num_threads(1)

    level = logging.DEBUG if config.DEBUG else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger().setLevel(level)

    if args.run_type == "train":
        trainer.train()
    elif args.run_type == "eval" or args.run_type == "test":
        trainer.eval(args.eval_interval, args.prev_ckpt_ind, config.USE_LAST_CKPT)


if __name__ == "__main__":
    main()
