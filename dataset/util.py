import mne
import numpy as np
import os
import scipy.io

mne.set_log_level(False)

def load_fif_file(filepath):
    raw = mne.io.read_raw_fif(filepath, preload=True)
    return raw


def extract_channels(raw):
    target_channels = ["F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1", "EEG Fpz-Cz", "EEG Pz-Oz"]
    available_channels = raw.ch_names
    selected_channels = [ch for ch in target_channels if ch in available_channels]
    
    if not selected_channels:
        raise ValueError("None of the target channels found in the file")

    raw.pick(selected_channels)
    raw.resample(100)  # Resample to 100 Hz if needed
    return raw

def extract_labels(raw):
    events_map = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2, 'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4}
    try:
        events = mne.events_from_annotations(raw, event_id=events_map, chunk_duration=30)[0][:, -1]
    except ValueError:
        events_map = {'nonrem1': 0, 'nonrem2': 1, 'nonrem3': 2, 'rem': 3, 'wake': 4}
        events = mne.events_from_annotations(raw, event_id=events_map, chunk_duration=30)[0]

    start = events[0][0]
    events = events[:, -1]
    return events, start

def window(eeg_data, start, window_size=3000):
    num_samples = len(eeg_data[start:])
    num_windows = num_samples // window_size
    end = num_samples % window_size
    data = eeg_data[start:-end] if end != 0 else eeg_data[start:]
    windows = np.array(np.split(data, num_windows))
    
    return windows 

if __name__ == '__main__':
    path = 'data/Dataset/You snooze you win/training_fif/tr03-0005_001_30s_raw.fif'
    raw = load_fif_file(path)
    eeg_data = extract_channels(raw)
    events = extract_labels(raw)
    print(events)