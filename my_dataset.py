import sys

from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """
    Custom dataset class for image classification.
    
    This class handles loading images and their corresponding labels,
    and applies transformations to the images if specified.
    """

    def __init__(self, images_path: list, images_class: list, transform=None):
        """
        Initialize the dataset.
        
        Args:
            images_path: List of paths to images
            images_class: List of class labels for each image
            transform: Transformations to apply to images
        """
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        return len(self.images_path)

    def __getitem__(self, item):
        """
        Get a sample from the dataset.
        
        Args:
            item: Index of the sample
            
        Returns:
            img: Transformed image
            label: Class label
        """
        img = Image.open(self.images_path[item])
        # RGB is for color images, L is for grayscale images
        if img.mode != 'RGB':
            img = img.convert('RGB')

        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for creating batches.
        
        Each element in batch is a result from __getitem__, which is a tuple.
        The first element is the image, the second element is the label.
        The tuple contains 8 elements, 8 (img, label) pairs.
        So we use zip to unpack it into images and labels.
        
        Args:
            batch: Batch of samples
            
        Returns:
            images: Batch of images
            labels: Batch of labels
        """
        # The official implementation of default_collate can be referenced at
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
