# Semantic Audio-Visual Navigation (SAVi) Model

## Details

This folder contains the code for the `SAVi` baseline, a semantic audio-visual navigation model introduced in the [Semantic Audio-Visual Navigation](https://arxiv.org/pdf/2012.11583.pdf) paper.

## Usage

1. Render RIRs to prepare the training and evaluation data for the label predictor.

```bash
python \
  savnce_baselines/savi/pretraining/render_ir.py \
  --dataset-type train \
  --output-dir data/datasets/savi_pretraining_data \
  --num-per-scene 1000 \
  --multithread
```

2. Pretrain the label predictor (or use the pretrained model weights provided with this repository):

```bash
python \
  savnce_baselines/savi/pretraining/audiogoal_trainer.py \
  --run-type train \
  --model-dir data/models/savi_pretraining
```

For the `Distracted Environment` setting, add the `--distractor` flag.

3. Train the SAVi model with DDPPO using the best label predictor checkpoint from step 2 (the location predictor is trained online, following the original implementation). For training and evaluation commands, see the scripts in the repository root: `pretraining.sh`, `eval.sh`, `train.sh`, and `test.sh`.

## Citation

If you use this model in your research, please cite the following paper:

```
@inproceedings{chen21semantic,
  title     = {Semantic Audio-Visual Navigation},
  author    = {Changan Chen and Ziad Al-Halah and Kristen Grauman},
  booktitle = {CVPR},
  year      = {2021}
}
```