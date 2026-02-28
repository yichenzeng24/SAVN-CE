#!/bin/bash
# Pretraining script for SAVN-CE baselines (av_nav, smt_audio, savi, oracle_accddoa, magnet).
# Uncomment one block below, set task/exp_id/model_dir, and run.

port=${1:-29500}

export HABITAT_SIM_LOG='warning'
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export MASTER_PORT=${port}

# --- Usage notes ---

# Environment: Examples use clean env; for Distracted Environment, replace
#   **/rgbd_ddppo_clean_pretraining.yaml  ->  **/rgbd_ddppo_distractor_pretraining.yaml

# Single GPU: Replace the torchrun block with:
#   CUDA_VISIBLE_DEVICES=0 python savnce_baselines/<method>/run.py ...

# Pretrained init (for fine-tuning), add the following flags to the training command:
# av_nav:
#     RL.PPO.use_pretrained True \
#     RL.PPO.pretrained_path data/pretrained_weights/av_nav_clean.pth
# smt_audio:
#     RL.PPO.SCENE_MEMORY_TRANSFORMER.use_pretrained True \
#     RL.PPO.SCENE_MEMORY_TRANSFORMER.pretrained_path data/pretrained_weights/smt_audio_clean.pth
# savi:
#     RL.PPO.SCENE_MEMORY_TRANSFORMER.use_pretrained True \
#     RL.PPO.SCENE_MEMORY_TRANSFORMER.pretrained_path data/pretrained_weights/savi_clean.pth
# magnet:
#     RL.PPO.SCENE_MEMORY_TRANSFORMER.use_pretrained True \
#     RL.PPO.SCENE_MEMORY_TRANSFORMER.pretrained_path data/pretrained_weights/magnet_clean.pth \
#     RL.PPO.GOAL_DESCRIPTOR.use_pretrained True \
#     RL.PPO.GOAL_DESCRIPTOR.pretrained_path data/pretrained_weights/magnet_clean.pth


# --- smt_audio ---
# task=savnce_pretraining
# exp_id=0225_smt_audio
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun \
#     --nproc_per_node 4 \
#     --master_port ${port} \
#     savnce_baselines/savi/run.py \
#     --run-type train \
#     --exp-config savnce_baselines/benchmark/config/smt_audio/rgbd_ddppo_clean_pretraining.yaml \
#     --model-dir ${model_dir}

# --- savi ---
# task=savnce_pretraining
# exp_id=0225_savi
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun \
#     --nproc_per_node 4 \
#     --master_port ${port} \
#     savnce_baselines/savi/run.py \
#     --run-type train \
#     --exp-config savnce_baselines/savi/config/mp3d/rgbd_ddppo_clean_pretraining.yaml \
#     --model-dir ${model_dir}

# --- oracle_accddoa ---
# task=savnce_pretraining
# exp_id=0225_oracle_accddoa
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun \
#     --nproc_per_node 4 \
#     --master_port ${port} \
#     savnce_baselines/magnet/run.py \
#     --run-type train \
#     --exp-config savnce_baselines/magnet/config/mp3d/rgbd_ddppo_benchmark_pretraining.yaml \
#     --model-dir ${model_dir}

# --- oracle_accddoa_everlasting ---
# task=savnce_pretraining
# exp_id=0225_oracle_accddoa_everlasting
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun \
#     --nproc_per_node 4 \
#     --master_port ${port} \
#     savnce_baselines/magnet/run.py \
#     --run-type train \
#     --exp-config savnce_baselines/magnet/config/mp3d/rgbd_ddppo_benchmark_pretraining.yaml \
#     --model-dir ${model_dir} \
#     TASK_CONFIG.TASK.ORACLE_ACCDDOA_SENSOR.EVERLASTING True

# --- magnet ---
# task=savnce_pretraining
# exp_id=0225_magnet
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun \
#     --nproc_per_node 4 \
#     --master_port ${port} \
#     savnce_baselines/magnet/run.py \
#     --run-type train \
#     --exp-config savnce_baselines/magnet/config/mp3d/rgbd_ddppo_clean_pretraining.yaml \
#     --model-dir ${model_dir}