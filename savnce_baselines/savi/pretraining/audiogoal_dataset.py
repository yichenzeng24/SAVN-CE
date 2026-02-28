#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modifications Copyright (c) 2026 Yichen Zeng, Wuhan University, Email: zengyichen@whu.edu.cn
# Description: Adapted for semantic audio-visual navigation in continuous environment (SAVN-CE).

import glob
import os
from collections import Counter
import librosa
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy.signal import fftconvolve
from skimage.measure import block_reduce
from concurrent.futures import ProcessPoolExecutor
from savnce.mp3d_utils import CATEGORY_INDEX_MAPPING, SCENE_SPLITS
    

class AudioGoalDataset(Dataset):
    def __init__(self, scenes, split, has_distractor_sound=False):
        self.spectra = list()
        self.sed_labels = list()
        self.binaural_rir_dir = f"data/datasets/savi_pretraining_data/mp3d/{split}"
        self.cache_dir = f"data/cache/savi_pretraining_data/mp3d/{split}{'_distractor' if has_distractor_sound else ''}"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.source_sound_dir = f"data/sounds/semantic_splits/{split}{'_distractor' if has_distractor_sound else ''}"
        self.distractor_sound_dir = f"data/sounds/1s_all_distractor/{split}"
        self.rir_sampling_rate = 16000
        self.step_time = 0.25
        self.num_samples_per_step = int(self.step_time * self.rir_sampling_rate)
        self.mp3d_category2index_mapping = CATEGORY_INDEX_MAPPING
        self.mp3d_index2category_mapping = {v: k for k, v in self.mp3d_category2index_mapping.items()}
        self.source_sound_dict = dict()
        self.load_source_sounds()
        self.has_distractor_sound = has_distractor_sound
        if self.has_distractor_sound:
            self.distractor_sound_list = list()
            self.load_distractor_sounds()
        scenes = [path for path in scenes if os.path.isdir(os.path.join(self.binaural_rir_dir, path))]
        print(f'{len(scenes)} scenes found in {self.binaural_rir_dir}')
        with ProcessPoolExecutor(max_workers=16) as executor:
            futures = []
            for scene in scenes:
                cache_file = os.path.join(self.cache_dir, f'{scene}.h5')
                if os.path.exists(cache_file) and os.path.getsize(cache_file) <= 1024 * 1024 * 1: # 1MB
                    os.remove(cache_file)
                if not os.path.exists(cache_file):
                    futures.append(executor.submit(self.cache_dataset, cache_file, scene, position=0))
            if len(futures) > 0:
                for future in tqdm(futures, desc='Caching scenes', position=1):
                    future.result()
        self.load_caches()

    def load_caches(self):
        for cache_file in tqdm(glob.glob(os.path.join(self.cache_dir, '*.h5')), desc='Loading caches'):
            with h5py.File(cache_file, 'r') as f:
                self.spectra.extend(f['spectra'])
                self.sed_labels.extend(f['sed_labels'])
        self.dataset_len = len(self.spectra)
        print(f'Dataset has {self.dataset_len} steps in total')

    def load_source_sounds(self):
        source_sound_files = glob.glob(os.path.join(self.source_sound_dir, '*.wav'))
        for sound_file in source_sound_files:
            category = os.path.basename(sound_file).split('.')[0]
            audio_data, sr = librosa.load(sound_file, sr=self.rir_sampling_rate)
            audio_data = normalize_audio(audio_data)
            self.source_sound_dict[category] = audio_data

    def load_distractor_sounds(self):
        distractor_sound_files = glob.glob(os.path.join(self.distractor_sound_dir, '*.wav'))
        for sound_file in distractor_sound_files:
            audio_data, sr = librosa.load(sound_file, sr=self.rir_sampling_rate)
            audio_data = normalize_audio(audio_data)
            self.distractor_sound_list.append(audio_data)

    def compute_audiogoal(self, source_rir, source_sound, distractor_rir=None, distractor_sound=None, num_samples_per_step=4000):
        """
        Given sound_segment with length L1 and rir with length L2, the length of the convolved signal is L1 + L2 - 1.
        _filtered_source_signal and _filtered_distractor_signal are not updated if _episode_step_count 
        is not between the goal's onset and offset step.
        Args:
            rir: (num_channels, rir_length)
            sound: (sound_length,)
            num_samples_per_step: int, number of samples per step
        Returns:
            audiogoal: (steps, num_samples_per_step, num_channels)
        """
        if source_rir.shape[1] < source_rir.shape[0]:
            source_rir = source_rir.T # (num_channels, rir_length)
        num_channels = source_rir.shape[0]
        steps = len(source_sound) // num_samples_per_step
        filtered_source_signal = []
        for channel in range(num_channels):
            filtered_source_signal.append(fftconvolve(source_sound, source_rir[channel]))
        filtered_source_signal = np.stack(filtered_source_signal, axis=-1) # (audio_length, num_channels)
        if distractor_rir is not None and distractor_sound is not None:
            if distractor_rir.shape[1] < distractor_rir.shape[0]:
                distractor_rir = distractor_rir.T # (num_channels, rir_length)
            assert distractor_rir.shape[0] == source_rir.shape[0]
            while len(distractor_sound) < len(source_sound):
                distractor_sound = np.concatenate([distractor_sound, distractor_sound], axis=0)
            distractor_sound = distractor_sound[:len(source_sound)]
            filtered_distractor_signal = []
            for channel in range(num_channels):
                filtered_distractor_signal.append(fftconvolve(distractor_sound, distractor_rir[channel]))
            filtered_distractor_signal = np.stack(filtered_distractor_signal, axis=-1) # (audio_length, num_channels)
            audiogoal = filtered_source_signal[:steps * num_samples_per_step, :] + \
                        filtered_distractor_signal[:steps * num_samples_per_step, :] # (audio_length, num_channels)
        else:
            audiogoal = filtered_source_signal[:steps * num_samples_per_step, :] # (audio_length, num_channels)
        audiogoal = audiogoal.reshape(steps, num_samples_per_step, num_channels).astype(np.float32)
        return audiogoal

    def cache_dataset(self, cache_file, scene, position=0):
        scene_dir = os.path.join(self.binaural_rir_dir, scene)
        rir_files = glob.glob(os.path.join(scene_dir, '*.wav'))
        num_rirs = len(rir_files)
        sed_labels = np.random.randint(0, len(self.mp3d_category2index_mapping), size=num_rirs)
        with h5py.File(cache_file, 'w') as f:
            dataset_spectra = []
            dataset_sed_labels = []
            for item in tqdm(range(num_rirs), desc=scene, position=position):
                source_rir = librosa.load(rir_files[item], sr=self.rir_sampling_rate, mono=False)[0]
                category = self.mp3d_index2category_mapping[sed_labels[item]]
                source_sound = self.source_sound_dict[category]
                if self.has_distractor_sound:
                    distractor_rir = librosa.load(np.random.choice(rir_files), sr=self.rir_sampling_rate, mono=False)[0]
                    distractor_sound = self.distractor_sound_list[np.random.randint(0, len(self.distractor_sound_list))]
                else:
                    distractor_rir = None
                    distractor_sound = None
                audiogoal = self.compute_audiogoal(source_rir, source_sound, distractor_rir, distractor_sound, self.num_samples_per_step)
                steps = audiogoal.shape[0]
                dataset_spectra.extend([compute_spectrogram(audiogoal[i]) for i in range(steps)])
                dataset_sed_labels.extend([sed_labels[item]] * steps)
            f.create_dataset('spectra', data=np.array(dataset_spectra), dtype=np.float32)
            f.create_dataset('sed_labels', data=np.array(dataset_sed_labels), dtype=np.int32)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        spectra = torch.from_numpy(self.spectra[index])
        sed_labels = torch.tensor(self.sed_labels[index])
        return spectra, sed_labels
    
    
def compute_spectrogram(audiogoal):
    def compute_stft(signal):
        n_fft = 512
        hop_length = 160
        win_length = 400
        stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
        stft = block_reduce(stft, block_size=(4, 1), func=np.mean)
        return stft

    channel1_magnitude = np.log1p(compute_stft(audiogoal[:, 0]))
    channel2_magnitude = np.log1p(compute_stft(audiogoal[:, 1]))
    spectrogram = np.stack([channel1_magnitude, channel2_magnitude])

    return spectrogram

def normalize_audio(audio_data, target_power=0.001):
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

def test_h5py(file_name, selections=['spectra']):
    if isinstance(selections, str):
        selections = [selections]
    with h5py.File(file_name, 'r') as f:
        print(f.keys())
        for selection in selections:
            data = f[selection]
            if selection == 'spectra':
                print(f'spectra shape:{data.shape}')
                print(f'spectra dtype:{data.dtype}') 
            elif selection == 'sed_labels':
                print(f'sed_labels shape:{data.shape}')
                print(f'sed_labels dtype:{data.dtype}') 
                counter = Counter(data[:].tolist())
                mapping = {v: k for k, v in CATEGORY_INDEX_MAPPING.items()}
                print(f'there are {len(counter)} categories')
                print(', '.join(f'{mapping[key]}: {value}' for key, value in counter.items()))
            else:
                print(f'{selection} not found')

if __name__ == '__main__':

    split = 'val'
    scenes = SCENE_SPLITS[split]
    has_distractor_sound = False
    dataset = AudioGoalDataset(scenes=scenes, split=split, has_distractor_sound=has_distractor_sound)

    # file_name = "data/cache/savi_pretraining_data/mp3d/train/sT4fr6TAbpF.h5"
    # test_h5py(file_name, selections=['spectra', 'sed_labels', 'doa_labels'])   

    # file_name = "data/cache/savi_pretraining_data/mp3d/val/2azQ1b91cZZ.h5"
    # test_h5py(file_name, selections=['spectra', 'sed_labels', 'doa_labels'])   

    # file_name = "data/cache/savi_pretraining_data/mp3d/test_distractor/pa4otMbVnkk.h5"
    # test_h5py(file_name, selections=['spectra', 'sed_labels'])  

