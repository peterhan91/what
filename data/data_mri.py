import glob
import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as tvtransforms
from data.fastmri import subsample, transforms

class ImageDataset(Dataset):
    def __init__(self, config, transforms_=None, mode='train'):
        self.opt = config
        self.transform = transforms_
        self.files = sorted(glob.glob(os.path.join(self.opt.data_dir, '%s' % 'singlecoil_'+mode) + '/*.png'))

    def __getitem__(self, index):
        GT_size = self.opt.GT_size

        img = np.asarray(Image.open(self.files[index % len(self.files)]))
        img = img.astype(np.float32) / 255.
        if np.ndim(img) < 3:
            img = np.expand_dims(img, -1)
        H, W, _ = img.shape
        rnd_h = random.randint(0, max(0, H - GT_size))
        rnd_w = random.randint(0, max(0, W - GT_size))
        img = img[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]

        img_lr, img, mean, std = self.transform(img)
        return {'image': img_lr, 'label': img, 'mean': mean, 'std': std}

    def __len__(self):
        return len(self.files)

class DataTransform:
    def __init__(self, seed, factor=4.0, rcenter=0.08, hamming=True):
        self.seed = seed 
        self.hamming = hamming
        if hamming: 
            self.mask_func = subsample.HammingMaskFunc(
                                    accelerations=[factor])
        else:
            self.mask_func = subsample.RandomMaskFunc(
                                    center_fractions=[rcenter],
                                    accelerations=[factor])
    def __call__(self, target):
        img = target
        if target.shape[2] != 2:
            img = np.concatenate((target, np.zeros_like(target)), axis=2)
        assert img.shape[-1] == 2
        img = transforms.to_tensor(img)
        kspace = transforms.fft2(img) 
        center_kspace, _ = transforms.apply_mask(kspace, self.mask_func, 
                                                hamming=self.hamming, seed=self.seed)
        img_LF = transforms.complex_abs(transforms.ifft2(center_kspace))
        img_LF = img_LF.unsqueeze(0)
        _, mean, std = transforms.normalize_instance(img_LF, eps=1e-11)
        target = transforms.to_tensor(np.transpose(target, (2, 0, 1)))  # target shape [1, H, W]
        # target = transforms.normalize(target, mean, std, eps=1e-11)
        
        return img_LF, target, mean, std