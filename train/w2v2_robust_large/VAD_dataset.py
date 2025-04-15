# VAD_dataset.py
import json
import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np
import random
import librosa
import torch
import random
import numpy as np


class VADDataset(Dataset):
    def __init__(self, manifest_path, feature_extractor, max_duration_s=30.0, stride_duration_s=15.0):
        """
        Args:
            manifest_path (str): Path to manifest.json file
            feature_extractor: Wav2Vec2 feature extractor
            max_duration_s (float): Window size in seconds
            stride_duration_s (float): Stride size in seconds for sliding window
        """
        self.feature_extractor = feature_extractor
        self.max_duration_s = max_duration_s
        self.stride_duration_s = stride_duration_s
        
        # Read manifest file
        self.entries = []
        with open(manifest_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())

                # Calculate number of windows for this file
                duration = entry['duration']
                n_windows = self._calculate_num_windows(duration)

                # Store entry with number of windows
                if n_windows > 0:
                    entry['n_windows'] = n_windows
                    self.entries.append(entry)
        
        # Create mappings for audio file index and the window index of that respective audio
        self.index_mapping = []
        for entry_idx, entry in enumerate(self.entries):
            for window_idx in range(entry['n_windows']):
                self.index_mapping.append((entry_idx, window_idx))

    def _calculate_num_windows(self, duration):
        """Calculate number of windows for a given duration"""
        samples = int(duration * 16000)  # assuming 16kHz sampling rate
        window_samples = int(self.max_duration_s * 16000)
        stride_samples = int(self.stride_duration_s * 16000)
        
        if samples < window_samples:
            return 1
        
        return (samples - window_samples) // stride_samples + 1

    def _read_rttm(self, rttm_path, audio_duration, sr=16000):
        """Read RTTM file and convert to frame-level labels"""
        num_frames = int(audio_duration * sr)
        labels = np.zeros(num_frames)
        
        with open(rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                start_time = float(parts[3])
                duration = float(parts[4])
                
                start_frame = int(start_time * sr)
                end_frame = int((start_time + duration) * sr)
                
                if end_frame > num_frames:
                    end_frame = num_frames
                labels[start_frame:end_frame] = 1
                
        return labels

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        '''
        Returns labels and features for a 60s chunk of audio
        '''

        # Get entry and window indices
        entry_idx, window_idx = self.index_mapping[idx]
        entry = self.entries[entry_idx]
        
        # Load audio
        audio_path = entry['audio_filepath']
        speech_array, sr = librosa.load(audio_path, sr=16000)
        speech_array = torch.from_numpy(speech_array)

        # Calculate window positions
        window_samples = int(self.max_duration_s * sr)  # Fixed size for all windows
        stride_samples = int(self.stride_duration_s * sr)
        start_idx = window_idx * stride_samples
        end_idx = start_idx + window_samples
        
        audio_window = torch.zeros(window_samples)  # Fixed size tensor
        label_window = torch.zeros(window_samples)  # Fixed size tensor
        
        # Get frame-level labels
        labels = self._read_rttm(entry['rttm_filepath'], entry['duration'], sr)
        labels = torch.from_numpy(labels)
        
        # Handle audio and label windowing
        if start_idx < len(speech_array):
            # Calculate how much of the window we can fill
            actual_samples = min(window_samples, len(speech_array) - start_idx)
            
            # Fill the fixed windows with available data
            audio_window[:actual_samples] = speech_array[start_idx:start_idx + actual_samples]
            label_window[:actual_samples] = labels[start_idx:start_idx + actual_samples]
        
        # Process audio through feature extractor
        inputs = self.feature_extractor(
            audio_window.numpy(),
            sampling_rate=sr,
            return_tensors="pt",
            padding="longest",
        )

        return {
            "input_values": inputs["input_values"][0],
            "labels": label_window,
            "audio_path": audio_path,
            "num_speakers": entry["num_speakers"],
            "window_idx": window_idx,
            "start_time": start_idx / sr,
            "end_time": end_idx / sr
        }
