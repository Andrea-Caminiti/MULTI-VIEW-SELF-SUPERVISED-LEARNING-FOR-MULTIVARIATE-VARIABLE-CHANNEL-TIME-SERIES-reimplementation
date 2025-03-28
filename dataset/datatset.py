import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataset.util import load_fif_file, extract_channels, extract_labels, window
import os
import gc
from tqdm import tqdm

class EEGDataset(Dataset):
    def __init__(self, data, transform=None):
        
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, labels = self.data[idx]  # Shape: (channels, labels)
        if self.transform:
            for i in range(len(sample)):
                sample[i] = self.transform(sample[i])
            
        sample = (sample-np.mean(sample))/np.std(sample)

        return torch.tensor(np.array(sample), dtype=torch.float32), torch.tensor(labels, dtype=torch.int16)

class EEGAugmentations:
    def __init__(self, noise_std=0.05, scaling_factor=0.1, mask_prob=0.2, time_warp_ratio=0.1):
       
        self.noise_std = noise_std
        self.scaling_factor = scaling_factor
        self.mask_prob = mask_prob
        self.time_warp_ratio = time_warp_ratio

    def add_noise(self, x):
        return x + np.random.normal(0, self.noise_std, x.shape)

    def scale(self, x):
        factor = 1 + np.random.uniform(-self.scaling_factor, self.scaling_factor)
        return x * factor

    def mask(self, x):
        mask = np.random.rand(*x.shape) > self.mask_prob
        return x * mask

    def time_warp(self, x):
        num_warp = int(len(x) * self.time_warp_ratio)
        warp_points = np.random.randint(0, len(x), num_warp)
        x[warp_points] = x[warp_points] * np.random.uniform(0.8, 1.2)
        return x

    def __call__(self, x):
        if np.random.rand() < 0.25:
            x = self.add_noise(x)
        if np.random.rand() < 0.25:
            x = self.scale(x)
        if np.random.rand() < 0.25:
            x = self.mask(x)
        if np.random.rand() < 0.25:
            x = self.time_warp(x)
        return x

def createDataset(folderPath, aug = EEGAugmentations()):
    
    thinkers = {}
    for i, f in tqdm(enumerate(os.listdir(folderPath)), desc='Loading Dataset...', leave=False):
        path = os.path.join(folderPath, f)
        raw = load_fif_file(path)
        raw = extract_channels(raw)
        labels, start = extract_labels(raw)
        subject = f[:f.index('_')]  
        if 'sleep-cassette' in folderPath:
            chs = [np.split(ch, len(ch)//3000) for ch in raw.get_data()] # The data in sleep cassette is made up of 30s windows
            data = [np.array(chs), labels]
        else:
            chs = [window(ch, start) for ch in raw.get_data()]
            data = [chs, labels]
        thinkers[subject] = data
        del data, labels, raw
    
    d_set = EEGDataset(list(thinkers.values()), aug)
    
    return d_set
    
            
         
if __name__ == '__main__':
    createDataset('data/Dataset/sleep-cassette')