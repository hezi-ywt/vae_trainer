import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


class ResolutionBucketDataset(Dataset):
    def __init__(self, root_dir, bucket_sizes, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            bucket_sizes (list of tuple): The list of (width, height) tuples defining the buckets.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.bucket_sizes = bucket_sizes
        self.transform = transform
        self.image_paths = []
        self.image_buckets = defaultdict(list)

        self._load_images()

    def _load_images(self):
        for filename in os.listdir(self.root_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                filepath = os.path.join(self.root_dir, filename)
                try:
                    with Image.open(filepath) as img:
                        width, height = img.size
                        bucket = self._get_bucket(width, height)
                        if bucket:
                            self.image_paths.append(filepath)
                            self.image_buckets[bucket].append(filepath)
                except Exception as e:
                    print(f"Error loading image {filename}: {e}")

    def _get_bucket(self, width, height):
        for bucket in self.bucket_sizes:
            if width <= bucket[0] and height <= bucket[1]:
                return bucket
        return None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        sample = {'image': image, 'path': img_path}

        if self.transform:
            sample['image'] = self.transform(image)

        return sample
