#!/bin/bash
# Evaluation script: run validation over checkpoints and get validation curves.
# Uncomment one block below, set task/exp_id/model_dir, and run.

export HABITAT_SIM_LOG='warning'
export PYTHONPATH="${PYTHONPATH:+:}$(pwd)"

# --- Eval pretraining checkpoints ---

task=savnce_pretraining
exp_id=0225_smt_audio
model_dir=data/models/${task}/${exp_id}

CUDA_VISIBLE_DEVICES=1 \
python \
    savnce_baselines/savi/run.py \
    --run-type eval \
    --exp-config savnce_baselines/benchmark/config/smt_audio/rgbd_ddppo_clean_pretraining.yaml \
    --model-dir ${model_dir} \
    --prev-ckpt-ind -1 \
    --eval-interval 2


# task=savnce_pretraining
# exp_id=0225_savi
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0 \
# python \
#     savnce_baselines/savi/run.py \
#     --run-type eval \
#     --exp-config savnce_baselines/benchmark/config/smt_audio/rgbd_ddppo_clean_pretraining.yaml \
#     --model-dir ${model_dir} \
#     --prev-ckpt-ind 0 \
#     --eval-interval 2


# task=savnce_pretraining
# exp_id=0225_oracle_accddoa
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0 \
# python \
#     savnce_baselines/magnet/run.py \
#     --run-type eval \
#     --exp-config savnce_baselines/magnet/config/mp3d/rgbd_ddppo_benchmark_pretraining.yaml \
#     --model-dir ${model_dir} \
#     --prev-ckpt-ind 0 \
#     --eval-interval 2


# task=savnce_pretraining
# exp_id=0225_oracle_accddoa_everlasting
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0 \
# python \
#     savnce_baselines/magnet/run.py \
#     --run-type eval \
#     --exp-config savnce_baselines/magnet/config/mp3d/rgbd_ddppo_benchmark_pretraining.yaml \
#     --model-dir ${model_dir} \
#     --prev-ckpt-ind 0 \
#     --eval-interval 2 \
#     TASK_CONFIG.TASK.ORACLE_ACCDDOA_SENSOR.EVERLASTING True


# task=savnce_pretraining
# exp_id=0225_magnet
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0 \
# python \
#     savnce_baselines/magnet/run.py \
#     --run-type eval \
#     --exp-config savnce_baselines/magnet/config/mp3d/rgbd_ddppo_clean_pretraining.yaml \
#     --model-dir ${model_dir} \
#     --prev-ckpt-ind 0 \
#     --eval-interval 2



# --- Eval training checkpoints ---

# task=savnce
# exp_id=0225_av_nav
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0 \
# python \
#     savnce_baselines/av_nav/run.py \
#     --run-type eval \
#     --exp-config savnce_baselines/av_nav/config/mp3d/rgbd_ddppo_clean.yaml \
#     --model-dir ${model_dir} \
#     --prev-ckpt-ind 0 \
#     --eval-interval 2


# task=savnce
# exp_id=0225_smt_audio
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=1 \
# python \
#     savnce_baselines/savi/run.py \
#     --run-type eval \
#     --exp-config savnce_baselines/benchmark/config/smt_audio/rgbd_ddppo_clean.yaml \
#     --model-dir ${model_dir} \
#     --prev-ckpt-ind 0 \
#     --eval-interval 2


# task=savnce
# exp_id=0225_savi
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0 \
# python \
#     savnce_baselines/savi/run.py \
#     --run-type eval \
#     --exp-config savnce_baselines/benchmark/config/smt_audio/rgbd_ddppo_clean.yaml \
#     --model-dir ${model_dir} \
#     --prev-ckpt-ind 0 \
#     --eval-interval 2


# task=savnce
# exp_id=0225_oracle_accddoa
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0 \
# python \
#     savnce_baselines/magnet/run.py \
#     --run-type eval \
#     --exp-config savnce_baselines/magnet/config/mp3d/rgbd_ddppo_benchmark.yaml \
#     --model-dir ${model_dir} \
#     --prev-ckpt-ind 0 \
#     --eval-interval 2


# task=savnce
# exp_id=0225_oracle_accddoa_everlasting
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0 \
# python \
#     savnce_baselines/magnet/run.py \
#     --run-type eval \
#     --exp-config savnce_baselines/magnet/config/mp3d/rgbd_ddppo_benchmark.yaml \
#     --model-dir ${model_dir} \
#     --prev-ckpt-ind 0 \
#     --eval-interval 2 \
#     TASK_CONFIG.TASK.ORACLE_ACCDDOA_SENSOR.EVERLASTING True


# task=savnce
# exp_id=0225_magnet
# model_dir=data/models/${task}/${exp_id}

# CUDA_VISIBLE_DEVICES=0 \
# python \
#     savnce_baselines/magnet/run.py \
#     --run-type eval \
#     --exp-config savnce_baselines/magnet/config/mp3d/rgbd_ddppo_clean.yaml \
#     --model-dir ${model_dir} \
#     --prev-ckpt-ind 0 \
#     --eval-interval 2