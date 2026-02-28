#!/bin/bash
# Test script: evaluate best validation checkpoint on test split.
# Uncomment one block below, set task/exp_id/model_dir, and run.

export HABITAT_SIM_LOG='warning'
export PYTHONPATH="${PYTHONPATH:+:}$(pwd)"

# --- Usage notes ---
# Distracted Environment: replace **/rgbd_ddppo_clean.yaml with **/rgbd_ddppo_distractor.yaml,
#   and EVAL.SPLIT test with EVAL.SPLIT test_distractor.
# Specific checkpoint: replace --eval-best with EVAL_CKPT_PATH_DIR data/models/savnce/<exp_id>/data/ckpt.xxx.pth

# --- av_nav ---
# task=savnce
# exp_id=0225_av_nav
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0 \
# python \
#     savnce_baselines/av_nav/run.py \
#     --run-type test \
#     --exp-config savnce_baselines/av_nav/config/mp3d/rgbd_ddppo_clean.yaml \
#     --model-dir ${model_dir} \
#     --eval-best \
#     EVAL.SPLIT test \
#     TEST_EPISODE_COUNT 1000 \
#     RL.DDPPO.pretrained False

# --- smt_audio ---
# task=savnce
# exp_id=0225_smt_audio
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=1 \
# python \
#     savnce_baselines/savi/run.py \
#     --run-type test \
#     --exp-config savnce_baselines/benchmark/config/smt_audio/rgbd_ddppo_clean.yaml \
#     --model-dir ${model_dir} \
#     --eval-best \
#     EVAL.SPLIT test \
#     TEST_EPISODE_COUNT 1000 \
#     RL.DDPPO.pretrained False

# --- savi ---
# task=savnce
# exp_id=0225_savi
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0 \
# python \
#     savnce_baselines/savi/run.py \
#     --run-type test \
#     --exp-config savnce_baselines/savi/config/mp3d/rgbd_ddppo_clean.yaml \
#     --model-dir ${model_dir} \
#     --eval-best \
#     EVAL.SPLIT test \
#     TEST_EPISODE_COUNT 1000 \
#     RL.DDPPO.pretrained False

# --- oracle_accddoa ---
# task=savnce
# exp_id=0225_oracle_accddoa
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0 \
# python \
#     savnce_baselines/magnet/run.py \
#     --run-type test \
#     --exp-config savnce_baselines/magnet/config/mp3d/rgbd_ddppo_benchmark.yaml \
#     --model-dir ${model_dir} \
#     --eval-best \
#     EVAL.SPLIT test \
#     TEST_EPISODE_COUNT 1000 \
#     RL.DDPPO.pretrained False

# --- oracle_accddoa_everlasting ---
# task=savnce
# exp_id=0225_oracle_accddoa_everlasting
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0 \
# python \
#     savnce_baselines/magnet/run.py \
#     --run-type test \
#     --exp-config savnce_baselines/magnet/config/mp3d/rgbd_ddppo_benchmark.yaml \
#     --model-dir ${model_dir} \
#     --eval-best \
#     TASK_CONFIG.TASK.ORACLE_ACCDDOA_SENSOR.EVERLASTING True \
#     EVAL.SPLIT test \
#     TEST_EPISODE_COUNT 1000 \
#     RL.DDPPO.pretrained False

# --- magnet ---
# task=savnce
# exp_id=0225_magnet
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0 \
# python \
#     savnce_baselines/magnet/run.py \
#     --run-type test \
#     --exp-config savnce_baselines/magnet/config/mp3d/rgbd_ddppo_clean.yaml \
#     --model-dir ${model_dir} \
#     --eval-best \
#     EVAL.SPLIT test \
#     TEST_EPISODE_COUNT 1000 \
#     RL.DDPPO.pretrained False