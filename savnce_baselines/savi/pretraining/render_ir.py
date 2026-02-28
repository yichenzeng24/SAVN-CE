# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modifications Copyright (c) 2026 Yichen Zeng, Wuhan University, Email: zengyichen@whu.edu.cn
# Description: Adapted for semantic audio-visual navigation in continuous environment (SAVN-CE).

import argparse
import os
import json
import shutil
import glob
import magnum as mn
import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_coeffs
import habitat_sim

import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyroomacoustics.experimental.rt60 import measure_rt60

sr = 16000

def make_configuration(scene_id, resolution=(512, 256), fov=20, visual_sensors=True):
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_id
    # backend_cfg.scene_dataset_config_file = "data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
    # backend_cfg.scene_dataset_config_file = "data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
    backend_cfg.enable_physics = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    if visual_sensors:
        # agent configuration
        rgb_sensor_cfg = habitat_sim.CameraSensorSpec()
        rgb_sensor_cfg.resolution = resolution
        rgb_sensor_cfg.far = np.iinfo(np.int32).max
        rgb_sensor_cfg.hfov = mn.Deg(fov)
        rgb_sensor_cfg.position = np.array([0, 1.5, 0])

        depth_sensor_cfg = habitat_sim.CameraSensorSpec()
        depth_sensor_cfg.uuid = 'depth_camera'
        depth_sensor_cfg.resolution = resolution
        depth_sensor_cfg.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_cfg.hfov = mn.Deg(fov)
        depth_sensor_cfg.position = np.array([0, 1.5, 0])

        # semantic_sensor_cfg = habitat_sim.CameraSensorSpec()
        # semantic_sensor_cfg.uuid = "semantic_camera"
        # semantic_sensor_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC
        # semantic_sensor_cfg.resolution = resolution
        # semantic_sensor_cfg.hfov = mn.Deg(fov)
        # semantic_sensor_cfg.position = np.array([0, 1.5, 0])

        agent_cfg.sensor_specifications = [rgb_sensor_cfg, depth_sensor_cfg]
    else:
        agent_cfg.sensor_specifications = []

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def add_acoustic_config(sim, args):
    # create the acoustic configs
    audio_sensor_spec = habitat_sim.AudioSensorSpec()
    audio_sensor_spec.uuid = "audio_sensor"
    audio_sensor_spec.enableMaterials = False
    audio_sensor_spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
    audio_sensor_spec.channelLayout.channelCount = 2
    audio_sensor_spec.position = [0.0, 1.5, 0.0]
    audio_sensor_spec.acousticsConfig.sampleRate = sr
    audio_sensor_spec.acousticsConfig.threadCount = 64
    audio_sensor_spec.acousticsConfig.indirect = True

    # add the audio sensor
    sim.add_sensor(audio_sensor_spec)
    if args.dataset in ['mp3d', 'gibson']:
        audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
        audio_sensor.setAudioMaterialsJSON('data/mp3d_material_config.json')


def get_res_angles_for(fov):
    if fov == 20:
        resolution = (384, 64)
        angles = [170, 150, 130, 110, 90, 70, 50, 30, 10, 350, 330, 310, 290, 270, 250, 230, 210, 190]
    elif fov == 30:
        resolution = (384, 128)
        angles = [0, 330, 300, 270, 240, 210, 180, 150, 120, 90, 60, 30]
    elif fov == 60:
        resolution = (256, 128)
        angles = [0, 300, 240, 180, 120, 60]
    elif fov == 90:
        resolution = (256, 256)
        angles = [0, 270, 180, 90]
    else:
        raise ValueError

    return resolution, angles

def acoustic_render(sim, receiver, source, rotation):
    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
    audio_sensor.setAudioSourceTransform(source)

    agent = sim.get_agent(0)
    new_state = sim.get_agent(0).get_state()
    new_state.position = receiver
    new_state.rotation = rotation
    new_state.sensor_states = {}
    agent.set_state(new_state, True)

    observation = sim.get_sensor_observations()

    return np.array(observation['audio_sensor'])


def normalize_depth(depth):
    min_depth = 0
    max_depth = 10
    depth = np.clip(depth, min_depth, max_depth)
    normalized_depth = (depth - min_depth) / (max_depth - min_depth)
    return normalized_depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mp3d')
    parser.add_argument('--dataset-type', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--output-dir', type=str, default="data/datasets/savi_pretraining_data")
    parser.add_argument('--num-per-scene', type=int, default=1000)
    parser.add_argument('--reset', default=False, action='store_true')
    parser.add_argument('--multithread', default=False, action='store_true')
    # parser.add_argument('--partition', type=str, default='learnlab,learnfair')
    # parser.add_argument('--slurm', default=False, action='store_true')
    args = parser.parse_args()

    if args.reset:
        shutil.rmtree(args.output_dir)

    if args.dataset == 'mp3d':
        from savnce.mp3d_utils import SCENE_SPLITS
        scenes = SCENE_SPLITS[args.dataset_type]
        # scenes.remove('2n8kARJN3HM')
        scene_ids = [f"data/scene_datasets/mp3d/{scene}/{scene}.glb" for scene in scenes]
    elif args.dataset == 'replica':
        # scene_ids = glob.glob(f"data/scene_datasets/replica/**/mesh.ply", recursive=True)
        scene_ids = glob.glob(f"data/scene_datasets/replica/**/habitat/mesh_semantic.ply", recursive=True)
    elif args.dataset == 'gibson':
        scene_ids = glob.glob(f"data/scene_datasets/gibson/*.glb")
    elif args.dataset == 'hm3d':
        scene_ids = glob.glob("data/scene_datasets/hm3d/**/*.basis.glb", recursive=True)
    else:
        raise ValueError
    print(f'{args.dataset} has {len(scene_ids)} environments')
    
    save = True
    if args.multithread:
        with ThreadPoolExecutor(max_workers=min(16, len(scene_ids))) as executor:
            futures = [executor.submit(run, args, scene_id, save) for scene_id in scene_ids]
            for future in tqdm(as_completed(futures), total=len(scene_ids), desc=f'Generating IR'):
                pass
    else:
        for scene_id in tqdm(scene_ids, total=len(scene_ids), desc=f'Generating IR'):
            run(args, scene_id, save)

def run(args, scene_id, save):
    scene = scene_id.split('/')[-1].split('.')[0]
    scene_obs_dir = os.path.join(args.output_dir, args.dataset, args.dataset_type, scene)
    os.makedirs(scene_obs_dir, exist_ok=True)

    metadata = {}
    cfg = make_configuration(scene_id, resolution=get_res_angles_for(fov=20)[0], visual_sensors=False)
    sim = habitat_sim.Simulator(cfg)
    add_acoustic_config(sim, args)
    for i in tqdm(range(args.num_per_scene), desc=scene):
        while True:
            agent_pos = sim.pathfinder.get_random_navigable_point()
            goal_pos = sim.pathfinder.get_random_navigable_point()
            delta = goal_pos - agent_pos
            distance = np.sqrt(delta[0] ** 2 + delta[2] ** 2)
            height_diff = np.abs(delta[1])
            try:
                if distance < 10 and distance > 2 and height_diff < 0.5:
                    goal_pos[1] += 1.5 # set goal height to 1.5m
                    heading = np.random.uniform(0, 2 * np.pi)
                    agent_rot = quat_from_angle_axis(heading, np.array([0, 1, 0]))
                    ir = acoustic_render(sim, agent_pos, goal_pos, agent_rot)
                    ir = ir.astype(np.float32) # (num_mic, num_samples)
                    if save:
                        sf.write(os.path.join(scene_obs_dir, f'{i:04d}_ir.wav'), ir.T, sr)
                    rt60 = measure_rt60(ir[0], sr, decay_db=30, plot=False)
                    agent_pos[1] += 1.5 # audio sensor is at 1.5m height relative to the agent
                    metadata[i] = (rt60, agent_pos.tolist(), quat_to_coeffs(agent_rot).tolist(), goal_pos.tolist(), heading)
                    break
            except Exception as e:
                print(e)
    sim.close()
    if save:
        with open(os.path.join(scene_obs_dir, 'metadata.json'), 'w') as fo:
            json.dump(metadata, fo)

if __name__ == '__main__':
    main()
