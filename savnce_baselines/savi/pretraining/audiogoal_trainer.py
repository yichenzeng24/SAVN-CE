#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modifications Copyright (c) 2026 Yichen Zeng, Wuhan University, Email: zengyichen@whu.edu.cn
# Description: Adapted for semantic audio-visual navigation in continuous environment (SAVN-CE).

import os
import time
import copy
from tqdm import tqdm
import torch.nn as nn
import torch
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from savnce_baselines.savi.pretraining.audiogoal_predictor import AudioGoalPredictor
from savnce_baselines.savi.pretraining.audiogoal_dataset import AudioGoalDataset
from savnce.mp3d_utils import SCENE_SPLITS
from savnce.utils import get_logger

class AudioGoalPredictorTrainer:
    def __init__(self, model_dir, has_distractor_sound):
        self.model_dir = model_dir
        self.has_distractor_sound = has_distractor_sound
        self.device = (torch.device("cuda", 0))
        self.logger = get_logger(name=__name__, filename=os.path.join(self.model_dir, 'train.log'))
        self.batch_size = 1024
        self.num_worker = 8
        self.lr = 1e-3
        self.weight_decay = None
        self.num_epoch = 50
        self.audiogoal_predictor = AudioGoalPredictor().to(device=self.device)
        summary(self.audiogoal_predictor.predictor, (2, 65, 26), device='cuda')

    def run(self, splits, writer=None):

        self.logger.info(self.audiogoal_predictor)
        datasets = dict()
        dataloaders = dict()
        dataset_sizes = dict()
        for split in splits:
            scenes = SCENE_SPLITS[split]
            datasets[split] = AudioGoalDataset(
                scenes=scenes,
                split=split,
                has_distractor_sound=self.has_distractor_sound
            )
            dataloaders[split] = DataLoader(
                dataset=datasets[split],
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=self.num_worker,
                sampler=None,
            )

            dataset_sizes[split] = len(datasets[split])
            self.logger.info('{} has {} samples'.format(split.upper(), dataset_sizes[split]))

        classifier_criterion = nn.CrossEntropyLoss().to(device=self.device)
        model = self.audiogoal_predictor
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)

        # training params
        since = time.time()
        best_acc = 0
        best_model_wts = None
        num_epoch = self.num_epoch if 'train' in splits else 1
        for epoch in range(num_epoch):
            self.logger.info('-' * 50)
            self.logger.info('Epoch {}/{}'.format(epoch, num_epoch))

            # Each epoch has a training and validation phase
            for split in splits:
                if split == 'train':
                    self.audiogoal_predictor.train()  # Set model to training mode
                else:
                    self.audiogoal_predictor.eval()  # Set model to evaluate mode

                running_total_loss = 0.0
                running_classifier_loss = 0.0
                running_classifier_corrects = 0
                # Iterating over data once is one epoch
                for i, data in enumerate(tqdm(dataloaders[split])):
                    # get the inputs
                    spectra, sed_labels = data
                    spectra = spectra.to(device=self.device, dtype=torch.float)
                    sed_labels = sed_labels.to(device=self.device, dtype=torch.long)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    inputs = spectra
                    predicts = model(inputs)
  
                    classifier_loss = classifier_criterion(predicts, sed_labels)
                    loss = classifier_loss

                    # backward + optimize only if in training phase
                    if split == 'train':
                        loss.backward()
                        optimizer.step()

                    running_total_loss += loss.item() * spectra.size(0)
                    running_classifier_loss += classifier_loss.item() * spectra.size(0)

                    # hard accuracy
                    running_classifier_corrects += torch.sum(
                        torch.argmax(predicts, dim=1) == sed_labels
                    ).item()

                epoch_total_loss = running_total_loss / dataset_sizes[split]
                epoch_classifier_loss = running_classifier_loss / dataset_sizes[split]
                epoch_classifier_acc = running_classifier_corrects / dataset_sizes[split]
                if writer is not None:
                    writer.add_scalar(f'Loss/{split}_total', epoch_total_loss, epoch)
                    writer.add_scalar(f'Loss/{split}_classifier', epoch_classifier_loss, epoch)
                    writer.add_scalar(f'Accuracy/{split}_classifier', epoch_classifier_acc, epoch)
                self.logger.info(f'{split.upper()} Total loss: {epoch_total_loss:.4f}, '
                             f'label loss: {epoch_classifier_loss:.4f}, label acc: {epoch_classifier_acc:.4f}')

                # deep copy the model
                target_acc = epoch_classifier_acc

                if split == 'val' and target_acc > best_acc:
                    best_acc = target_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    self.save_checkpoint(f"ckpt.{epoch}.pth")

        time_elapsed = time.time() - since
        self.logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        if 'test' not in splits:
            self.save_checkpoint(f"best_val.pth", checkpoint={"audiogoal_predictor": best_model_wts})
            self.logger.info('Best val acc: {:4f}'.format(best_acc))

        # if best_model_wts is not None:
        #     model.load_state_dict(best_model_wts)

    def save_checkpoint(self, ckpt_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = {
                "audiogoal_predictor": self.audiogoal_predictor.state_dict(),
            }
        torch.save(
            checkpoint, os.path.join(self.model_dir, ckpt_path)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        # required=True,
        default='train',
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--model-dir",
        default='data/models/savi_pretraining',
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--distractor",
        default=False,
        action='store_true',
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action='store_true',
        help="Modify config options from command line"
    )

    args = parser.parse_args()

    if args.distractor:
        args.model_dir = args.model_dir + '_distractor'
    log_dir = os.path.join(args.model_dir, 'tb')
    audiogoal_predictor_trainer = AudioGoalPredictorTrainer(args.model_dir, args.distractor)

    if args.run_type == 'train':
        writer = SummaryWriter(log_dir=log_dir)
        audiogoal_predictor_trainer.run(['train', 'val'], writer)
    else:
        ckpt = torch.load(os.path.join(args.model_dir, 'best_val.pth'), weights_only=False)
        audiogoal_predictor_trainer.audiogoal_predictor.load_state_dict(ckpt['audiogoal_predictor'])
        audiogoal_predictor_trainer.run(['test'])


if __name__ == '__main__':
    main()

