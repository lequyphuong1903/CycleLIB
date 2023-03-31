from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class A2B(Dataset):
    def __init__(self, root_A, root_B, transform=None):
        self.root_A = root_A
        self.root_B = root_B
        self.transform = transform

        self.photo_B = os.listdir(root_B)
        self.photo_A = os.listdir(root_A)
        self.length_dataset = max(len(self.photo_A), len(self.photo_B)) # 1000, 1500
        self.A_len = len(self.photo_A)
        self.B_len = len(self.photo_B)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        B_img = self.photo_B[index % self.B_len]
        A_img = self.photo_A[index % self.A_len]

        A_path = os.path.join(self.root_A, A_img)
        B_path = os.path.join(self.root_B, B_img)

        B_img = np.array(Image.open(B_path).convert("RGB"))
        A_img = np.array(Image.open(A_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=A_img, image0=B_img)
            A_img = augmentations["image"]
            B_img = augmentations["image0"]

        return A_img, B_img