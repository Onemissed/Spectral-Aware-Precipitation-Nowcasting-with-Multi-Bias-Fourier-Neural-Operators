import os
import numpy as np
from torch.utils.data import Dataset

class DataGenerator(Dataset):
    def __init__(self, lists):
        self.lists = lists
        # self.root_path = root_path

    def __len__(self):
        return len(self.lists)

    def __getitem__(self, index):
        if 'random' in str(self.lists[index]):
            sequence = np.load(os.path.join('/your/path/to/sevir_lr/data/vil_split/random/', str(self.lists[index]) + '.npy'))
        else:
            sequence = np.load(os.path.join('/your/path/to/sevir_lr/data/vil_split/storm', str(self.lists[index]) + '.npy'))

        return sequence