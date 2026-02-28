# Installation Guide for SAVN-CE

This document describes how to install SAVN-CE and prepare the required data for training and evaluation.

## Prerequisites

- Linux (tested on Ubuntu 22.04)
- CUDA-capable GPU (recommended for training and inference)
- Conda or Miniconda

## Set up the conda environment

```bash
conda create -n savnce python=3.9 cmake=3.14.0 -y
conda activate savnce

# Install system dependencies
sudo apt update
sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxft-dev libxext-dev libxi-dev libgl1-mesa-dev libglu1-mesa-dev libxcb-xinerama0 libx11-xcb-dev libglu1-mesa

# Install Habitat-Sim
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout RLRAudioPropagationUpdate
python setup.py install --headless --audio
cd ..
```

To support multiple audio sensors, edit `habitat_sim/simulator.py` (around `line 765`) and replace:  
`audio_sensor = self._agent._sensors["audio_sensor"]`
with
`audio_sensor = self._agent._sensors[self._spec.uuid]`.  
You can do this either in the habitat-sim source tree before running `python setup.py install` (file: `habitat-sim/src_python/habitat_sim/simulator.py`), or after install in the conda env (e.g. `$CONDA_PREFIX/lib/python3.9/site-packages/habitat_sim/simulator.py`).

```
Note: You do not need to install Habitat-Lab separately; this repo includes a modified version based on v0.2.2.
```

```bash
# Install Python dependencies
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# Clone and install SAVN-CE
git clone https://github.com/yichenzeng24/SAVN-CE.git
cd SAVN-CE
pip install -e .
```

## Prepare Data

Create the data directory (from the SAVN-CE repo root):

```bash
mkdir -p data && cd data
```

**1.** Download the [savnce-dataset](https://drive.google.com/drive/folders/1tz92HS9JsWmZuSnFjK513eu-Q0NYnTHT?usp=drive_link) and place it in the `datasets` folder.  
**2.** Follow the [Habitat-Sim dataset instructions](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md) to download the Matterport3D scene datasets into the `scene_datasets` folder.  
**3.** Download the sound assets from [SoundSpaces](https://github.com/facebookresearch/sound-spaces/blob/main/soundspaces/README.md), e.g.,
```bash
wget http://dl.fbaipublicfiles.com/SoundSpaces/sounds.tar.xz && tar xvf sounds.tar.xz
```
and place them in the `sounds` folder.  
**4.** (Optional) Download the [pretrained checkpoints](https://drive.google.com/drive/folders/1tz92HS9JsWmZuSnFjK513eu-Q0NYnTHT?usp=drive_link) and place them in the `pretrained_ckpts` folder.

The `data` folder should have the following structure:

```
data/
├── datasets/                          # Episodes for each split
│   └── savnce-dataset/
│       └── mp3d/v1/
│           └── [split]/
│               ├── [split].json.gz
│               └── content/
│                   └── [scene].json.gz
├── pretrained_ckpts/                   # Pretrained model weights
│   ├── av_nav/
│   ├── benchmark/
│   ├── magnet/
│   │   ├── oracle_accddoa_everlasting.pth
│   │   ├── oracle_accddoa.pth
│   │   ├── magnet_clean.pth
│   │   └── magnet_distractor.pth
│   ├── savi/
│   └── smt_with_audio/
├── scene_datasets/
│   └── mp3d/
│       └── [scene]/                    # Matterport3D scenes
│           ├── [scene]_semantic.ply
│           ├── [scene].glb
│           ├── [scene].house
│           └── [scene].navmesh
└── sounds/
    ├── semantic_splits/                # Goal sounds
    └── 1s_all_distractor/              # Distractor sounds
```

### (Optional) Enable Material-based Audio Simulation

By default, material-based audio simulation is disabled to reduce computational overhead and avoid known compatibility issues in SoundSpaces (see Issues [#111](https://github.com/facebookresearch/sound-spaces/issues/111) and [#145](https://github.com/facebookresearch/sound-spaces/issues/145)).  
If you would like to enable acoustic material modeling, download the MP3D material configuration file from [rlr-audio-propagation](https://github.com/facebookresearch/rlr-audio-propagation), e.g.,
```bash
cd data && wget https://raw.githubusercontent.com/facebookresearch/rlr-audio-propagation/main/RLRAudioPropagationPkg/data/mp3d_material_config.json
```
Then, in `savnce/sims/savnce_simulator.py` (around line 452), set `audio_sensor_spec.enableMaterials` to `True`.
The `mp3d_material_config.json` file assigns acoustic coefficients to different materials for audio rendering.
