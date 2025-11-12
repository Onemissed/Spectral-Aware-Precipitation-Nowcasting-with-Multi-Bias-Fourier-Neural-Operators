import os
import numpy as np
import random
from torch.utils.data import Dataset
# from scipy.misc import imsave, imread
# from matplotlib.pyplot import imread
from imageio import imread

class DataGenerator(Dataset):
    def __init__(self, lists, root_path):
        self.lists = lists
        self.root_path = root_path

    def __len__(self):
        return len(self.lists)

    def __getitem__(self, index):
        self.folds = self.root_path + 'sample_' + str(self.lists[index]) + '/'
        files = ["img_{}.png".format(x) for x in range(1, 16)]
        imgs = []
        for file in files:
            imgs.append(imread(self.folds + file)[:, :, np.newaxis])
        imgs = np.stack(imgs, 0)
        return imgs