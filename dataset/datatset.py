import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataset.util import load_fif_file, extract_channels, extract_labels, window
import os
import gc
from tqdm import tqdm

class EEGDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data (np.ndarray): NumPy array containing the eeg datat.
            transform (callable, optional): Optional transform to apply to the data.
        """
        
        self.data = data
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, labels = self.data[idx]  # Shape: channels, labels
        if self.transform:
            for i in range(len(sample)):
                sample[i] = self.transform(sample[i])

        return torch.tensor(np.array(sample), dtype=torch.float32), torch.tensor(labels, dtype=torch.int16)

class EEGAugmentations:
    def __init__(self, noise_std=0.05, scaling_factor=0.1, mask_prob=0.2, time_warp_ratio=0.1):
        """
        Args:
            noise_std (float): Standard deviation of Gaussian noise.
            scaling_factor (float): Max scaling factor.
            mask_prob (float): Probability of masking a segment.
            time_warp_ratio (float): Ratio of samples to warp.
        """
        self.noise_std = noise_std
        self.scaling_factor = scaling_factor
        self.mask_prob = mask_prob
        self.time_warp_ratio = time_warp_ratio

    def add_noise(self, x):
        """Adds Gaussian noise"""
        return x + np.random.normal(0, self.noise_std, x.shape)

    def scale(self, x):
        """Scales signal by a random factor"""
        factor = 1 + np.random.uniform(-self.scaling_factor, self.scaling_factor)
        return x * factor

    def mask(self, x):
        """Randomly masks segments of the signal"""
        mask = np.random.rand(*x.shape) > self.mask_prob
        return x * mask

    def time_warp(self, x):
        """Applies random time warping"""
        num_warp = int(len(x) * self.time_warp_ratio)
        warp_points = np.random.randint(0, len(x), num_warp)
        x[warp_points] = x[warp_points] * np.random.uniform(0.8, 1.2)
        return x

    def __call__(self, x):
        """Applies a random combination of augmentations"""
        if np.random.rand() < 0.5:
            x = self.add_noise(x)
        if np.random.rand() < 0.5:
            x = self.scale(x)
        if np.random.rand() < 0.5:
            x = self.mask(x)
        if np.random.rand() < 0.5:
            x = self.time_warp(x)
        return x

def createDataloaders(eegDataset, dim):
    num = len(eegDataset) // dim
    for i in range(num):
        yield DataLoader(eegDataset[i*dim:(i+1)*dim], batch_size=32)
        


def createDataset(folderPath):
    
    thinkers = {}
    for i, f in tqdm(enumerate(os.listdir(folderPath)), desc='Loading Dataset...', leave=False):
        path = os.path.join(folderPath, f)
        raw = load_fif_file(path)
        raw = extract_channels(raw)
        labels, start = extract_labels(raw)
        subject = f[:f.index('_')]  
        if 'sleep-cassette' in folderPath:
            chs = [np.split(ch, len(ch)//3000) for ch in raw.get_data()] # The data in sleep cassette is made up of 30s windows
            data = [chs, labels]
        else:
            chs = [window(ch, start) for ch in raw.get_data()]
            data = [chs, labels]
        thinkers[subject] = data
        del data, labels, raw
        gc.collect()
    
    aug = EEGAugmentations()
    d_set = EEGDataset(list(thinkers.values()), aug)
    
    return d_set
    
            
         
if __name__ == '__main__':
    createDataset('data/Dataset/sleep-cassette')