import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataset.util import load_fif_file, extract_channels, extract_labels
import os
import gc

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
        sample, labels = self.data[idx]  # Shape: [(num_channels, timepoints), labels]
        
        if self.transform:
            sample = self.transform(sample)

        return torch.tensor(sample, dtype=torch.float32), torch.tensor(labels, dtype=torch.int16)

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
        
def window(eeg_data, window_size=3000):
    _, num_samples = eeg_data.shape
    num_windows = num_samples // window_size
    
    windows = np.array(np.split(eeg_data[:, :num_windows * window_size], num_windows, axis=1))
    
    return windows 

def createDataset(folderPath):
    
    thinkers = {}
    for i, f in enumerate(os.listdir(folderPath)):
        path = os.path.join(folderPath, f)
        raw = load_fif_file(path)
        raw = extract_channels(raw)
        labels = extract_labels(raw)
        if 'sleep-cassette' in folderPath:
            data = [list(zip(np.split(ch, len(ch)//3000), labels)) for ch in raw.get_data()] # The data in sleep cassette is made up of 30s windows
            subject = f[:5]  
        else:
            data = [list(zip(ch, labels)) for ch in window(raw.get_data())]
            subject = f[:f.index('_')]  
        if subject not in thinkers:
            thinkers[subject] = [data]
        else:
            thinkers[subject].append(data)
        del data, labels, raw
        gc.collect()
    
    aug = EEGAugmentations()
    d_set = EEGDataset(list(thinkers.values()), aug)
    
    return d_set
    
            
         
if __name__ == '__main__':
    createDataset('data/Dataset/sleep-cassette')