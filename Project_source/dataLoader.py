import torch
import pandas as pd
import os
from skimage import io, transform, color
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils

import matplotlib.pyplot as plt


class MaskedDataset(Dataset):
    def __init__(self, file_info_dataframe, root_dir, transform=None):
        """
           Args:
               file_info_dataframe (dataframe): Path to the csv file with annotations.
               root_dir (string): Directory with all the images.
               transform (callable, optional): Optional transform to be applied
                   on a sample.
       """
        self.masked_data = file_info_dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.masked_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.toList()

        image_dir = os.path.join(self.root_dir, self.masked_data.iloc[idx, 1])

        image = io.imread(image_dir)

        if len(image.shape) == 2:       # Converting from BW to RGB
            image = color.gray2rgb(image)

        if image.shape[2] == 4:
            print(image.shape)
            print(image_dir)
            print()

        sample = {'image': image, 'label': self.masked_data.iloc[idx, 2]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = int(output_size)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # h, w = image.shape[:2]
        # if isinstance(self.output_size, int):
        #     if h > w:
        #         new_h, new_w = self.output_size * h / w, self.output_size
        #     else:
        #         new_h, new_w = self.output_size, self.output_size * w / h
        # else:
        #     new_h, new_w = self.output_size
        #
        # new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (self.output_size, self.output_size))

        return {'image': img, 'label': label}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(np.array([label]))}


if __name__ == "__main__":
    csv_file = "Dataset/dataset.csv"
    root_dir = "Dataset/"

    data_loader = MaskedDataset(pd.read_csv(csv_file), root_dir)

    scale = Rescale(256)
    crop = RandomCrop(128)
    composed = transforms.Compose([Rescale(256),
                                   RandomCrop(224)])

    # Apply each of the above transforms on sample.
    fig = plt.figure()
    sample = data_loader[65]
    plt.imshow(sample['image'])
    plt.show()

    fig = plt.figure()
    for i, tsfrm in enumerate([scale, composed]):
        transformed_sample = tsfrm(sample)

        ax = plt.subplot(1, 3, i + 1)
        plt.imshow(transformed_sample['image'])
        plt.show()
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)

    plt.show()
