import os
import os.path as osp
import random
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import Adagrad
from tqdm import tqdm

from model.MSHNet import MSHNet
from model.loss import AverageMeter, SLSIoULoss
from utils.data import IRSTD_Dataset
from utils.metric import PD_FA, ROCMetric, mIoU


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', '1', 'y'):
        return True
    if value in ('no', 'false', 'f', '0', 'n'):
        return False
    raise ValueError('Boolean value expected.')


def parse_args(default_mode=None):
    parser = ArgumentParser(description='MSHNet training and testing')

    parser.add_argument('--dataset-dir', type=str, default='datasets/IRSTD-1K')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--warm-epoch', type=int, default=5)

    parser.add_argument('--base-size', type=int, default=256)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--multi-gpus', type=str2bool, default=False)
    parser.add_argument('--if-checkpoint', type=str2bool, default=False)
    parser.add_argument('--resume-path', type=str, default='')
    parser.add_argument('--save-dir', type=str, default='repro_runs')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-split-file', type=str, default=None)
    parser.add_argument('--test-split-file', type=str, default=None)
    parser.add_argument(
        '--allow-test-selection',
        action='store_true',
        help=(
            'Legacy reproduction only: evaluate the official test split every '
            'epoch and select weight.pkl by test mIoU. Never use for RC/AAAI results.'
        ),
    )

    parser.add_argument('--mode', type=str, default=default_mode or 'train', choices=['train', 'test'])
    parser.add_argument('--weight-path', type=str, default='')

    return parser.parse_args()


def validate_spatial_args(args):
    for name in ('base_size', 'crop_size'):
        value = int(getattr(args, name))
        if value <= 0 or value % 16 != 0:
            raise ValueError('--{} must be a positive multiple of 16'.format(name.replace('_', '-')))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer(object):
    def __init__(self, args):
        validate_spatial_args(args)
        self.args = args
        self.start_epoch = 0
        self.mode = args.mode
        self.device = self._select_device(args.device)

        self.train_loader = None
        if args.mode == 'train':
            trainset = IRSTD_Dataset(args, mode='train')
            self.train_loader = Data.DataLoader(
                trainset,
                args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=args.num_workers,
                pin_memory=self.device.type == 'cuda',
            )

        self.val_loader = None
        if args.mode == 'test' or args.allow_test_selection:
            if args.mode == 'train':
                print(
                    'WARNING: --allow-test-selection reads the official test split '
                    'during training. This run is legacy/non-claim-bearing.'
                )
            testset = IRSTD_Dataset(args, mode='test')
            self.val_loader = Data.DataLoader(
                testset,
                1,
                drop_last=False,
                num_workers=args.num_workers,
                pin_memory=self.device.type == 'cuda',
            )

        model = MSHNet(3)
        if args.multi_gpus and self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            print('use ' + str(torch.cuda.device_count()) + ' gpus')
            model = nn.DataParallel(model)
        model.to(self.device)
        self.model = model

        self.optimizer = Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)

        self.down = nn.MaxPool2d(2, 2)
        self.loss_fun = SLSIoULoss()
        self.PD_FA = PD_FA(1, 10, args.base_size)
        self.mIoU = mIoU(1)
        self.ROC = ROCMetric(1, 10)
        self.best_iou = 0
        self.warm_epoch = args.warm_epoch

        if args.mode == 'train':
            if args.if_checkpoint:
                self._resume_checkpoint(args.resume_path or args.weight_path)
            else:
                self.save_folder = osp.join(
                    args.save_dir,
                    'MSHNet-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))),
                )
                os.makedirs(self.save_folder, exist_ok=True)

        if args.mode == 'test':
            if not args.weight_path:
                raise ValueError('--weight-path is required in test mode')
            self._load_weight(args.weight_path)
            self.warm_epoch = -1

    @staticmethod
    def _select_device(device_arg):
        if device_arg == 'cpu':
            return torch.device('cpu')
        if device_arg == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA was requested, but torch.cuda.is_available() is False')
            return torch.device('cuda')
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _normalise_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'net' in checkpoint:
                state_dict = checkpoint['net']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        checkpoint_is_parallel = any(key.startswith('module.') for key in state_dict)
        model_is_parallel = isinstance(self.model, nn.DataParallel)
        if checkpoint_is_parallel and not model_is_parallel:
            state_dict = {
                key.replace('module.', '', 1): value
                for key, value in state_dict.items()
            }
        elif model_is_parallel and not checkpoint_is_parallel:
            state_dict = {'module.' + key: value for key, value in state_dict.items()}
        return state_dict

    def _load_weight(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = self._normalise_state_dict(checkpoint)
        self.model.load_state_dict(state_dict)

    def _resume_checkpoint(self, path):
        if not path:
            raise ValueError('--resume-path or --weight-path is required when --if-checkpoint true')

        checkpoint = torch.load(path, map_location=self.device)
        if not self.args.allow_test_selection:
            if not isinstance(checkpoint, dict):
                raise ValueError(
                    'Safe fixed-last resume requires a metadata checkpoint; '
                    'raw/legacy weights have unknown selection provenance.'
                )
            if checkpoint.get('checkpoint_selection') != 'fixed_last_no_test_or_target_validation':
                raise ValueError(
                    'Safe fixed-last resume refuses a checkpoint with unknown or '
                    'test-selected provenance. Use a fixed-last checkpoint or run '
                    'the explicitly legacy --allow-test-selection mode.'
                )
            if checkpoint.get('official_test_accessed_during_training') is not False:
                raise ValueError(
                    'Safe fixed-last resume requires an explicit no-test-access marker.'
                )
        self.model.load_state_dict(self._normalise_state_dict(checkpoint))
        if isinstance(checkpoint, dict) and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if isinstance(checkpoint, dict):
            self.start_epoch = checkpoint.get('epoch', -1) + 1
            self.best_iou = checkpoint.get('iou', 0)

        self.save_folder = osp.dirname(path) or self.args.save_dir
        os.makedirs(self.save_folder, exist_ok=True)

    def train(self, epoch):
        self.model.train()
        tbar = tqdm(self.train_loader)
        losses = AverageMeter()
        tag = epoch > self.warm_epoch

        for _, (data, mask) in enumerate(tbar):
            data = data.to(self.device, non_blocking=True)
            labels = mask.to(self.device, non_blocking=True)

            masks, pred = self.model(data, tag)
            loss = self.loss_fun(pred, labels, self.warm_epoch, epoch)

            scaled_labels = labels
            for j in range(len(masks)):
                if j > 0:
                    scaled_labels = self.down(scaled_labels)
                loss = loss + self.loss_fun(masks[j], scaled_labels, self.warm_epoch, epoch)

            loss = loss / (len(masks) + 1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, loss %.4f' % (epoch, losses.avg))

    def test(self, epoch):
        if self.val_loader is None:
            raise RuntimeError(
                'No test loader is available. Training defaults to fixed-last '
                'without official-test access; use --mode test for final evaluation.'
            )
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        self.ROC.reset()
        tbar = tqdm(self.val_loader)
        tag = epoch > self.warm_epoch

        with torch.no_grad():
            for _, (data, mask) in enumerate(tbar):
                data = data.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)

                _, pred = self.model(data, tag)

                self.mIoU.update(pred, mask)
                self.PD_FA.update(pred, mask)
                self.ROC.update(pred, mask)
                _, mean_IoU = self.mIoU.get()

                tbar.set_description('Epoch %d, IoU %.4f' % (epoch, mean_IoU))

            FA, PD = self.PD_FA.get(len(self.val_loader))
            _, mean_IoU = self.mIoU.get()

            if self.mode == 'train':
                if mean_IoU > self.best_iou:
                    self.best_iou = mean_IoU
                    torch.save(self.model.state_dict(), osp.join(self.save_folder, 'weight.pkl'))
                    with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
                        f.write(
                            '{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n'.format(
                                time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                                epoch,
                                self.best_iou,
                                PD[0],
                                FA[0] * 1000000,
                            )
                        )

                all_states = {
                    'net': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'iou': self.best_iou,
                    'checkpoint_selection': 'official_test_selected_best_legacy_only',
                    'official_test_accessed_during_training': True,
                }
                torch.save(all_states, osp.join(self.save_folder, 'checkpoint.pkl'))
            elif self.mode == 'test':
                print('mIoU: ' + str(mean_IoU))
                print('Pd: ' + str(PD[0]))
                print('Fa: ' + str(FA[0] * 1000000))

    def save_fixed_last(self, epoch):
        """Save the current epoch without constructing or reading a test loader."""

        if self.mode != 'train':
            raise RuntimeError('save_fixed_last is available only in train mode')
        state_dict = self.model.state_dict()
        all_states = {
            'net': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'checkpoint_selection': 'fixed_last_no_test_or_target_validation',
            'official_test_accessed_during_training': False,
            'protocol_scope': 'legacy_single_domain_fixed_last_not_rc_protocol',
            'train_split_file': str(self.train_loader.dataset.list_dir),
            'seed': int(self.args.seed),
        }
        torch.save(state_dict, osp.join(self.save_folder, 'weight-last.pkl'))
        torch.save(all_states, osp.join(self.save_folder, 'checkpoint.pkl'))


def main(default_mode=None):
    args = parse_args(default_mode)
    seed_everything(args.seed)

    trainer = Trainer(args)

    if trainer.mode == 'train':
        for epoch in range(trainer.start_epoch, args.epochs):
            trainer.train(epoch)
            if args.allow_test_selection:
                trainer.test(epoch)
            else:
                trainer.save_fixed_last(epoch)
    else:
        trainer.test(1)


if __name__ == '__main__':
    main()
