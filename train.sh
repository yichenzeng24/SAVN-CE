#!/bin/bash
# Training script (after pretraining): train navigation agents with pretrained weights.
# Uncomment one block below, set task/exp_id/model_dir and pretrained_weights path, then run.

port=${1:-29500}

export HABITAT_SIM_LOG='warning'
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export MASTER_PORT=${port}

# --- av_nav ---
# task=savnce
# exp_id=0225_av_nav
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun \
#     --nproc_per_node 4 \
#     --master_port ${port} \
#     savnce_baselines/av_nav/run.py \
#     --run-type train \
#     --exp-config savnce_baselines/av_nav/config/mp3d/rgbd_ddppo_clean.yaml \
#     --model-dir ${model_dir}

# --- smt_audio ---
# task=savnce
# exp_id=0225_smt_audio
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun \
#     --nproc_per_node 4 \
#     --master_port ${port} \
#     savnce_baselines/savi/run.py \
#     --run-type train \
#     --exp-config savnce_baselines/benchmark/config/smt_audio/rgbd_ddppo_clean.yaml \
#     --model-dir ${model_dir} \
#     RL.DDPPO.pretrained True \
#     RL.DDPPO.pretrained_weights data/models/savnce_pretraining/0225_smt_audio/data/ckpt.xxx.pth

# --- savi ---
# task=savnce
# exp_id=0225_savi
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun \
#     --nproc_per_node 4 \
#     --master_port ${port} \
#     savnce_baselines/savi/run.py \
#     --run-type train \
#     --exp-config savnce_baselines/savi/config/mp3d/rgbd_ddppo_clean.yaml \
#     --model-dir ${model_dir} \
#     RL.DDPPO.pretrained True \
#     RL.DDPPO.pretrained_weights data/models/savnce_pretraining/0225_savi/data/ckpt.xxx.pth

# --- oracle_accddoa ---
# task=savnce
# exp_id=0225_oracle_accddoa
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun \
#     --nproc_per_node 4 \
#     --master_port ${port} \
#     savnce_baselines/magnet/run.py \
#     --run-type train \
#     --exp-config savnce_baselines/magnet/config/mp3d/rgbd_ddppo_benchmark.yaml \
#     --model-dir ${model_dir} \
#     RL.DDPPO.pretrained True \
#     RL.DDPPO.pretrained_weights data/models/savnce_pretraining/0225_oracle_accddoa/data/ckpt.xxx.pth

# --- oracle_accddoa_everlasting ---
# task=savnce
# exp_id=0225_oracle_accddoa_everlasting
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun \
#     --nproc_per_node 4 \
#     --master_port ${port} \
#     savnce_baselines/magnet/run.py \
#     --run-type train \
#     --exp-config savnce_baselines/magnet/config/mp3d/rgbd_ddppo_benchmark.yaml \
#     --model-dir ${model_dir} \
#     TASK_CONFIG.TASK.ORACLE_ACCDDOA_SENSOR.EVERLASTING True \
#     RL.DDPPO.pretrained True \
#     RL.DDPPO.pretrained_weights data/models/savnce_pretraining/0225_oracle_accddoa_everlasting/data/ckpt.xxx.pth

# --- magnet ---
# task=savnce
# exp_id=0225_magnet
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun \
#     --nproc_per_node 4 \
#     --master_port ${port} \
#     savnce_baselines/magnet/run.py \
#     --run-type train \
#     --exp-config savnce_baselines/magnet/config/mp3d/rgbd_ddppo_clean.yaml \
#     --model-dir ${model_dir} \
#     RL.DDPPO.pretrained True \
#     RL.DDPPO.pretrained_weights data/models/savnce_pretraining/0225_magnet/data/ckpt.xxx.pth