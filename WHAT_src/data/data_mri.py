import glob
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, '%s' % 'singlecoil_'+mode) + '/*.png'))

    def __getitem__(self, index):
        img = Image.open(self.files_A[index % len(self.files_A)])
        img_lr = self.transform(img)
        return {'image': img_lr, 'label': img}

    def __len__(self):
        return len(self.files)