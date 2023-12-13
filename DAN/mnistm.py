import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import random_split
import os
from PIL import Image

class MNISTMDataset(Dataset):
    def __init__(self, root_dir, train, transform=None):
        self.train = train
        self.root_dir = root_dir
        self.transform = transform

        self.labels = []
        #name = "train" if self.train else "test"
        #with open(os.path.join(root_dir, 'mnist_m', f'mnist_m_{name}_labels.txt'), 'r') as file:
        #    for line in file:
        #        self.labels.append(int(line.strip()[-1]))
                
        name = "train" if self.train else "test"
        labels_file = os.path.join(self.root_dir, "mnist_m", f"mnist_m_{name}_labels.txt")
        img_dir = os.path.join(self.root_dir, "mnist_m", f"mnist_m_{name}")
        with open(labels_file) as f:
            content = [line.rstrip().split(" ") for line in f]
        self.img_paths = [os.path.join(img_dir, x[0]) for x in content]
        #check_length(self, {"train": 59001, "test": 9001}[name])
        self.labels = [int(x[1]) for x in content]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #name = "train" if self.train else "test"
        #img_dir = os.path.join(self.root_dir, "mnist_m", f"mnist_m_{name}")
        img_name = self.img_paths[idx]
        image = Image.open(img_name)

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label