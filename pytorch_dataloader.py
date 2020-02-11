import torch
import os
import random
import numpy as np
from torch.utils import data
from torchvision import datasets
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision.utils import save_image
import torchvision

img_lists = []

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
        

def get_loader(image_dir):

    transform = []
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    
    dataset = ImageFolderWithPaths(image_dir, transform)
    print(dataset)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1)
    return data_loader


dataset_loader = get_loader('./test/') # NOTE! need to be subfolders inside the designated folder

if not os.path.exists('./path_record'):
	os.makedirs('./path_record')

for i, data in enumerate(dataset_loader, 0):
	print(os.path.basename(data[2][0]))
	save_image(data[0], './path_record/{}_'.format(i+1) + os.path.basename(data[2][0]) + '.png')
