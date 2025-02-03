import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class ContentDataset(Dataset):
    """
    Loads content images from a directory for style transfer training.
    Exects all images to be in .jpg or.png in root_dir."""
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir #setting root directory to given path
        self.fnames = [
            f for f in os.listdir(root_dir) if f.lower().endswith('.jpg') or f.lower().endswith('.png')
        ] #list of image filenames in root_dir matching the given extensions
        self.transform = transform

    def __len__(self):
        return len(self.fnames) #return number of images in the dataset
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.fnames[idx]) #get image path
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) #read image in BGR format (default)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert from BGR to RGB

        if self.transform:
            img = self.transform(img)

        return img