import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as Transforms
import torch.nn.functional as F
import torchaudio
import re
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from tqdm import tqdm
import random

from lib.data.spatial_aug import random_so3, rotate_foa_acn

SAMPLE_RATE = 16000

# Augmentations

def polarity_flip(x: torch.Tensor, p: float = 0.5):
    if torch.rand(()) < p:
        return -x
    return x

def gain_jitter(x: torch.Tensor, lo_db: float = -4.0, hi_db: float = 4.0):
    g = torch.empty(1).uniform_(lo_db, hi_db).item()
    return x * (10.0 ** (g / 20.0))

def add_mixture_noise(x: torch.Tensor, snr_db_range=(25.0, 45.0), p: float = 0.3):
    if torch.rand(()) > p:
        return x
    snr = float(torch.empty(1, device=x.device).uniform_(*snr_db_range))
    rms = _rms_joint(x)
    noise = torch.randn_like(x)
    if x.dim() == 2 and x.size(0) == 4:
        noise[1:, :] = 0.0 # only W channel gets noise (to be ambisonic safe)

    noise = noise * (rms / (_rms_joint(noise) + 1e-12)) * (10.0 ** (-snr / 20.0))
    return x + noise

class SoundAugmenter:
    def __init__(self, sr=SAMPLE_RATE, enable_flip=False, enable_jitter=False, enable_noise=False,
                 enable_rotation=False, add_noise_prob=0.3, noise_snr_db=(30.0, 50.0)):
        self.sr = sr
        self.enable_flip = enable_flip
        self.enable_jitter = enable_jitter
        self.enable_noise = enable_noise
        self.enable_rotation = enable_rotation
        self.add_noise_prob = add_noise_prob
        self.noise_snr_db = noise_snr_db

    def augment_sounds(self, stems: list[torch.Tensor]) -> list[torch.Tensor]:
        out = []
        for x in stems:
            T = x.size(-1)
            dtype = x.dtype
            device = x.device

            if self.enable_jitter:
                x = gain_jitter(x)
            
            if self.enable_flip:
                x = polarity_flip(x)
            
            if x.dtype != dtype:
                x = x.to(dtype)

            if x.device != device:
                x = x.to(device)

            out.append(x.contiguous())
        return out
    
    def augment_mixture(self, mixture: torch.Tensor) -> torch.Tensor:
        if not self.enable_noise:
            return mixture
        return add_mixture_noise(mixture, snr_db_range=self.noise_snr_db, p=self.add_noise_prob)

    def augment_field(self, mixture: torch.Tensor, sources: torch.Tensor):
        """Apply the same random SO(3) to mixture and sources (FOA only).

        mixture: [4, T] in ACN order [W, Y, Z, X].
        sources: [S, 4, T] in the same order.
        Returns the rotated pair, or the inputs unchanged if not FOA / disabled.
        """
        if not self.enable_rotation or mixture.dim() < 2 or mixture.size(0) != 4:
            return mixture, sources
        R = random_so3(1, device=mixture.device, dtype=mixture.dtype).squeeze(0)
        mixture = rotate_foa_acn(mixture, R)
        sources = rotate_foa_acn(sources, R)
        return mixture, sources


def seed_worker(worker_id):
    """
    Seeds each dataloader worker differently.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _rms_joint(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    RMS over all channels jointly.
    x: [C, T]
    returns scalar tensor
    """
    return torch.sqrt((x ** 2).mean() + eps)

def _scale_to_snr(ref: torch.Tensor, src: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Scale src so that RMS(ref) / RMS(src_scaled) = 10^(snr_db/20).
    One scalar gain applied to all channels of `src`.
    """
    r_ref = _rms_joint(ref)
    r_src = _rms_joint(src)
    ratio = 10.0 ** (snr_db / 20.0)
    gain = r_ref / (r_src * ratio + 1e-12)
    return src * gain

class DCase2019Dataset(Dataset):
    def __init__(self, audio_dir, chunk_size=2*SAMPLE_RATE, n_overlapping_sounds=2, split=list(range(1, 5)), seed=42, mode='train', num_val_samples=None, out_channels=1, output_dataframe=False, snr_range=(-2.0, 2.0),
                 augmentations=None):
        """
        Args:
            audio_dir (str): Directory with all audio files.
            chunk_size (int): Size of audio chunks in samples (e.g., 48000 = 1 second for 48kHz audio).
            n_overlapping_sounds (int): Maximum number of overlapping audios in chunks. 
            #create_ov_sounds (bool): If True, overlapping sounds are created during __getitem__ from isolated sounds.
            split (list): List of integers to filter audio files by split number.
            seed (int): Seed
            mode (str): train, val, test
            num_val_samples (int): Number of validation samples to use. If None, use all available combinations.
            out_channels (int): Number of output channels (1 for mono, 4 for FOA, etc.).
            output_dataframe (bool): If True, return a DataFrame with labels for each sample.
            snr_range (tuple): Range of SNR values to use for scaling.
            augmentations (callable): Function to apply data augmentations.
        """
        self.number_regex = re.compile(r'\d+')
        self.seed = seed
        self.mode = mode
        self.num_val_samples = num_val_samples
        self.audio_dir = Path(audio_dir)
        self.chunk_size = chunk_size
        self.n_overlapping_sounds = n_overlapping_sounds
        self.sample_rate = SAMPLE_RATE
        self.out_channels = out_channels
        self.output_dataframe = output_dataframe
        self.snr_low, self.snr_high = snr_range
        self.augmentations = augmentations
        ### Collect all files from which audio will be choosen.
        
        # Select split of dataset
        all_wav_files_in_dir = [os.path.join(self.audio_dir, f) for f in os.listdir(self.audio_dir) if f.endswith('.wav')]
        if split:
            splits_tuple = tuple(["split" + str(s) for s in split])
            self.audio_files = [f_path for f_path in all_wav_files_in_dir if os.path.basename(f_path).startswith(splits_tuple)]
        else:
            self.audio_files = all_wav_files_in_dir

        # Create or use existing overlapping sounds
        #if create_ov_sounds:
            #pass
            #self.audio_files = [f for f in self.audio_files if os.path.split(f)[1].find(f'_ov1_') != -1]

        self.audiolabels = self.find_non_overlapping_sounds(self.audio_files)
        self.used_combos = set()
        self.combinations_list = list(self.groupwise_combinations(self.audiolabels, r=self.n_overlapping_sounds, used_combos=self.used_combos))

        if (self.mode == 'val' or self.mode == 'test') and self.num_val_samples is not None:
            if self.num_val_samples > len(self.combinations_list):
                print(f"Warning: Requested {self.num_val_samples} validation samples, but only "
                      f"{len(self.combinations_list)} are available. Using all available samples.")
            else:
                # Shuffle the list in a reproducible way using the seed
                print(f"Creating fixed validation set with {self.num_val_samples} samples.")
                random.Random(self.seed).shuffle(self.combinations_list)
                # Take only the first N samples
                self.combinations_list = self.combinations_list[:self.num_val_samples]
        

    def find_non_overlapping_sounds(self, filelist, overlap_count=0):
        sound_df_list = []
        for file in filelist:
            # Read Label file and sort them by starting time
            label_file = os.path.splitext(file)[0] + '.csv'
            df = pd.read_csv(label_file)
            df = df.sort_values(by='start_time').reset_index(drop=True)
            current_start = df.iloc[0]['start_time']
            current_end = df.iloc[0]['end_time']
    
            df['prev_end'] = df['end_time'].shift(1)  # End time of the previous sound
            df['next_start'] = df['start_time'].shift(-1)  # Start time of the next sound
            df['prev_end'] = df['prev_end'].fillna(-float('inf'))
            df['next_start'] = df['next_start'].fillna(float('inf'))
            
            if overlap_count == 0:
                sounds = df[(df['start_time'] >= df['prev_end']) & (df['end_time'] <= df['next_start'])]
                sounds = sounds[['sound_event_recording', 'start_time', 'end_time', 'ele', 'azi', 'dist']]
            else:
                df['new_group'] = (df['start_time'] > df['end_time'].shift()).cumsum()
                df = df[df.groupby('new_group')['new_group'].transform('count') <= overlap_count]
                # Group by 'new_group' and aggregate to get min start_time and max end_time
                sounds = df.groupby('new_group').agg(
                    {'sound_event_recording': 'sum','start_time': 'min', 'end_time': 'max',
                     'ele': lambda x: np.nan, 'azi': lambda x: np.nan, 'dist': lambda x: np.nan}).reset_index(drop=True)
            
            sounds.insert(0, "filename", os.path.splitext(file)[0])
            sounds.insert(1, "room", self.get_room_of_file(os.path.splitext(file)[0]))
            sound_df_list.append(sounds)
        final_sound_df = pd.concat(sound_df_list, axis=0).reset_index(drop=True)
        return final_sound_df

    def get_room_of_file(self, filename):
            # find roomnumber
            found = self.number_regex.findall(os.path.split(filename)[1])
            ir = found[1]
            return ir

    def get_audio_from_idx(self, idx, offset=None):
        entry_series = self.audiolabels.iloc[idx] # Get the row as a Series
        sound_filepath = entry_series["filename"] + ".wav"
        
        csv_start_time, csv_end_time = entry_series['start_time'], entry_series['end_time']
        # Calculate sample positions based on CSV times
        sound_event_start_sample = int(np.floor(csv_start_time * self.sample_rate))
        sound_event_end_sample = int(np.ceil(csv_end_time * self.sample_rate))
        sound_event_duration_samples = sound_event_end_sample - sound_event_start_sample
        
        frames_to_load = 0
        start_offset_in_file = 0
        
        if sound_event_duration_samples <= 0:
            waveform = torch.zeros((1 if self.out_channels == 0 else self.out_channels, self.chunk_size)) # Default silent waveform
        elif sound_event_duration_samples >= self.chunk_size:
            max_start_offset_within_event = sound_event_duration_samples - self.chunk_size
            random_offset_within_event = np.random.randint(0, max_start_offset_within_event + 1)
            start_offset_in_file = sound_event_start_sample + random_offset_within_event
            frames_to_load = self.chunk_size
        else:
            start_offset_in_file = sound_event_start_sample
            frames_to_load = sound_event_duration_samples
        
        if frames_to_load > 0:
            waveform, _ = torchaudio.load(sound_filepath, frame_offset=start_offset_in_file, num_frames=frames_to_load)
        else: 
            waveform = torch.zeros((1 if self.out_channels == 0 else self.out_channels, 0))
        
        
        if self.out_channels == 1 and waveform.shape[0] > 1:
            waveform = waveform[0].unsqueeze(0)  # Take first channel for mono
        elif self.out_channels > 0 and waveform.shape[0] == self.out_channels:
            pass # Channels match
        elif waveform.shape[0] != self.out_channels and self.out_channels > 0:
            raise ValueError(f"Mismatch in waveform channels ({waveform.shape[0]}) and desired out_channels ({self.out_channels}) for {sound_filepath}")
        
        
        final_waveform = torch.zeros((waveform.shape[0] if waveform.ndim > 1 else 1, self.chunk_size), dtype=waveform.dtype)
        
        current_frames = waveform.shape[1]
        placement_offset = 0
        if current_frames > 0:
            if current_frames < self.chunk_size:
                placement_offset = np.random.randint(0, self.chunk_size - current_frames + 1)
            else: # current_frames == self.chunk_size (or > if something went wrong, but should be clipped by load)
                placement_offset = 0
                if current_frames > self.chunk_size: # Should not happen with correct loading
                    waveform = waveform[:, :self.chunk_size]
                    current_frames = self.chunk_size
        
            final_waveform[:, placement_offset : placement_offset + current_frames] = waveform
        
        label_df = pd.DataFrame()
        if self.output_dataframe:
            # Update label_df times relative to the start of the chunk
            label_df = entry_series.to_frame().T.copy(deep=True) # Convert Series to DataFrame and copy
            label_df['start_time'] = placement_offset / self.sample_rate
            label_df['end_time'] = (placement_offset + current_frames) / self.sample_rate
            # Ensure labels are within chunk boundaries
            label_df[['start_time', 'end_time']] = label_df[['start_time', 'end_time']].clip(
                lower=0.0, upper=float(self.chunk_size)/self.sample_rate
            )
        
        return final_waveform, self.sample_rate, label_df


    def groupwise_combinations(self, df, group_col='room', r=2, used_combos=None):
        used = set(used_combos) if used_combos else set()
        room_to_indices = defaultdict(list)
        
        for idx, room in zip(df.index, df[group_col]):
            room_to_indices[room].append(idx)
    
        for room, indices in room_to_indices.items():
            for combo in combinations(indices, r):
                if combo not in used:
                    used.add(combo)
                    yield room, combo
        
        
    def __len__(self):
        return len(self.combinations_list) if self.n_overlapping_sounds > 1 else len(self.audiolabels)
    
    def __getitem__(self, idx):
        
        # Add n other audios randomly from the same room
        if self.n_overlapping_sounds > 1:
            try:
                room, combo = self.combinations_list[idx]
            except IndexError:
                raise IndexError(f"Combination index {idx} is out of range.")
            
            # Temporary Mixer (Maybe use this instead? https://github.com/ertug/MixCycle/blob/main/src/sc09mix.py)
            waveform, sample_rate, label_df = self.get_audio_from_idx(combo[0])

            if self.out_channels == 1:
                waveform = waveform[0].unsqueeze(0)

            # Augment first waveform if augmenter is provided
            if self.augmentations and self.mode == 'train':
                augmented_stems = self.augmentations.augment_sounds([waveform])
                waveform = augmented_stems[0]
            
            ref_waveform = waveform.clone()

            s_true = [waveform]
            mixture = waveform.clone()
            for n in range(1, len(combo)):
                add_waveform, _, add_label_df = self.get_audio_from_idx(combo[n])
                if self.out_channels == 1:
                    add_waveform = add_waveform[0].unsqueeze(0)

                # Augment additional waveforms if augmenter is provided
                if self.augmentations and self.mode == 'train':
                    augmented_stems = self.augmentations.augment_sounds([add_waveform])
                    add_waveform = augmented_stems[0]

                # Add together audio waves and label DataFrames
                snr_db = random.uniform(self.snr_low, self.snr_high)
                add_waveform = _scale_to_snr(ref_waveform, add_waveform, snr_db)
                mixture = mixture + add_waveform
                s_true.append(add_waveform)

                if self.output_dataframe:
                    label_df = pd.concat([label_df, add_label_df])
            s_true = torch.stack(s_true, dim=0)

            # Loudness scaling with peak cap to improve numerical conditioning
            target_rms = 0.05  # comfortable amplitude scale for TasNet
            peak_cap = 0.99
            rms = mixture.pow(2).mean().sqrt()
            if rms > 0:
                gain = target_rms / rms
                peak = mixture.abs().amax()
                if peak > 0:
                    gain = min(gain, peak_cap / peak)
                mixture = mixture * gain
                s_true = s_true * gain

            if self.output_dataframe:
                return mixture, s_true, label_df.to_dict(orient='records')
            
            if self.augmentations and self.mode == 'train':
                mixture = self.augmentations.augment_mixture(mixture)
                mixture, s_true = self.augmentations.augment_field(mixture, s_true)

            return mixture, s_true
        else:
            waveform, sample_rate, label_df = self.get_audio_from_idx(idx)
            if self.out_channels == 1:
                waveform = waveform[0].unsqueeze(0)
            if self.output_dataframe:
                return waveform, label_df.to_dict(orient='records')
            return waveform

class PrecomputedValDataset(Dataset):
    def __init__(self, preprocessed_dir):
        data = torch.load(preprocessed_dir)
        self.mixtures = data["mixtures"]
        self.sources = data["sources"]

    def __len__(self):
        return self.mixtures.size(0)

    def __getitem__(self, idx):
        return self.mixtures[idx], self.sources[idx]
    

class ChannelSliceDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, in_channels, out_channels):
        self.base_dataset = base_dataset
        self.in_channels = in_channels
        self.out_channels = out_channels

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        mixture, sources = self.base_dataset[idx]
        mixture = mixture[: self.in_channels, :]
        sources = sources[:, : self.out_channels, :]
        return mixture, sources
