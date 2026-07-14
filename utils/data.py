import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms

import os
from PIL import Image, ImageOps, ImageFilter
import os.path as osp
import sys
import random
import shutil
from glob import glob


class IRSTD_Dataset(Data.Dataset):
    def __init__(self, args, mode='train'):
        
        dataset_dir = args.dataset_dir

        self.list_dir = self._find_split_file(
            dataset_dir,
            mode,
            getattr(args, 'split_file', None),
        )
        self.imgs_dir = osp.join(dataset_dir, 'images')
        self.label_dir = osp.join(dataset_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines() if line.strip()]

        self.mode = mode
        self.crop_size = args.crop_size
        self.base_size = args.base_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __getitem__(self, i):
        name = osp.splitext(self.names[i])[0]
        img_path = self._resolve_image_path(self.imgs_dir, name)
        label_path = self._resolve_mask_path(self.label_dir, name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        
        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)

    @staticmethod
    def _find_split_file(dataset_dir, mode, split_file=None):
        if split_file:
            path = osp.expanduser(split_file)
            if not osp.isabs(path):
                path = osp.join(dataset_dir, path)
            path = osp.realpath(path)
            if not osp.isfile(path):
                raise FileNotFoundError('Explicit split file does not exist: {}'.format(path))
            return path

        if mode == 'train':
            candidates = [
                osp.join(dataset_dir, 'trainval.txt'),
                osp.join(dataset_dir, 'train.txt'),
            ]
            pattern = osp.join(dataset_dir, 'img_idx', 'train*.txt')
        elif mode == 'val':
            candidates = [osp.join(dataset_dir, 'test.txt')]
            pattern = osp.join(dataset_dir, 'img_idx', 'test*.txt')
        else:
            raise ValueError("Unkown self.mode")

        for path in candidates:
            if osp.exists(path):
                return path

        matches = sorted({osp.realpath(path) for path in glob(pattern) if osp.isfile(path)})
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                'Multiple split files match mode "{}" under {}: {}. '
                'Pass an explicit split_file.'.format(mode, dataset_dir, ', '.join(matches))
            )

        raise FileNotFoundError(
            'Cannot find split file for mode "{}" under {}. Expected one of {} or {}'.format(
                mode, dataset_dir, ', '.join(candidates), pattern
            )
        )

    @staticmethod
    def _resolve_image_path(root, name):
        for extension in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
            path = osp.join(root, name + extension)
            if osp.isfile(path):
                return path

        raise FileNotFoundError('Cannot find image/mask for "{}" under {}'.format(name, root))

    @staticmethod
    def _resolve_mask_path(root, name):
        # NUAA-SIRST uses ``<image_id>_pixels0.png`` while IRSTD-1K and
        # NUDT-SIRST use the image id directly.  Explicit candidates avoid
        # accidentally opening the NUAA XML annotation as an image.
        for suffix in ('', '_pixels0'):
            try:
                return IRSTD_Dataset._resolve_image_path(root, name + suffix)
            except FileNotFoundError:
                continue

        raise FileNotFoundError('Cannot find mask for "{}" under {}'.format(name, root))

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img, mask


    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        return img, mask
