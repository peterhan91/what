from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
from data.data_mri import ImageDataset, DataTransform


def get_dataloader(config):
    data_dir = config.data_dir
    batch_size = config.batch_size

    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.0,), (1.0,))])

    if config.data_name == 'mnist':
        train_dataset = dset.MNIST(root=data_dir, train=True, transform=trans, download=True)
        val_dataset = dset.MNIST(root=data_dir, train=False, transform=trans, download=True)
    elif config.data_name == 'fashion_mnist':
        train_dataset = dset.FashionMNIST(root=data_dir, train=True, transform=trans, download=True)
        val_dataset = dset.FashionMNIST(root=data_dir, train=False, transform=trans, download=True)
    elif config.data_name == 'mri':
        train_dataset = ImageDataset(config, transforms_=DataTransform(seed=config.seed), mode='train')
        val_dataset = ImageDataset(config, transforms_=DataTransform(seed=config.seed), mode='val')


    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              num_workers=config.num_work, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                             num_workers=config.num_work, shuffle=False)

    print('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(val_loader)))

    data_loader = {'train': train_loader, 'val': val_loader}

    return data_loader
