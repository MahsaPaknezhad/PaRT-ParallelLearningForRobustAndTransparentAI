import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def train_loader(path, train_batch_size, num_workers=24, pin_memory=False, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_transform.transforms.append(Cutout(16))

    train_dataset = datasets.ImageFolder(path, train_transform)
    '''train_dataset.samples = train_dataset.samples[:5000]
    train_dataset.imgs = train_dataset.imgs[:5000]
    train_dataset.targets = train_dataset.targets[:5000]'''

    return torch.utils.data.DataLoader(train_dataset,
        batch_size=train_batch_size, shuffle=True, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)


def val_loader(path, val_batch_size, num_workers=24, pin_memory=False, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=IMAGENET_MEAN, std=IMAGENET_STD)

    val_dataset = \
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Resize((256,256)),
                                 transforms.CenterCrop((224,224)),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))
    '''val_dataset.samples = val_dataset.samples[:4000]
    val_dataset.imgs = val_dataset.imgs[:4000]
    val_dataset.targets = val_dataset.targets[:4000]'''

    return torch.utils.data.DataLoader(val_dataset,
        batch_size=val_batch_size, shuffle=False, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)


def train_loader_cropped(path, train_batch_size, num_workers=24, pin_memory=False):
    normalize = transforms.Normalize(
        mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_transform.transforms.append(Cutout(16))

    train_dataset = datasets.ImageFolder(path, train_transform)
    '''train_dataset.samples = train_dataset.samples[:5000]
    train_dataset.imgs = train_dataset.imgs[:5000]
    train_dataset.targets = train_dataset.targets[:5000]'''

    return torch.utils.data.DataLoader(train_dataset,
        batch_size=train_batch_size, shuffle=True, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)


def val_loader_cropped(path, val_batch_size, num_workers=24, pin_memory=False):
    normalize = transforms.Normalize(
        mean=IMAGENET_MEAN, std=IMAGENET_STD)

    val_dataset = \
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Resize((224,224)),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))

    '''val_dataset.samples = val_dataset.samples[:4000]
    val_dataset.imgs = val_dataset.imgs[:4000]
    val_dataset.targets = val_dataset.targets[:4000]'''

    return torch.utils.data.DataLoader(val_dataset,
        batch_size=val_batch_size, shuffle=False, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)

