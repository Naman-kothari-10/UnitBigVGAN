# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import math
import os
import random
import torch
import torch.utils.data
import numpy as np
import librosa
from librosa.filters import mel as librosa_mel_fn
import pathlib
from tqdm import tqdm
from typing import List, Tuple, Optional
from env import AttrDict

MAX_WAV_VALUE = 32767.0  # NOTE: 32768.0 -1 to prevent int16 overflow (results in popping sound in corner cases)


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes):
    return dynamic_range_decompression_torch(magnitudes)


mel_basis_cache = {}
hann_window_cache = {}


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int = None,
    center: bool = False,
) -> torch.Tensor:
    """
    Calculate the mel spectrogram of an input signal.
    This function uses slaney norm for the librosa mel filterbank (using librosa.filters.mel) and uses Hann window for STFT (using torch.stft).

    Args:
        y (torch.Tensor): Input signal.
        n_fft (int): FFT size.
        num_mels (int): Number of mel bins.
        sampling_rate (int): Sampling rate of the input signal.
        hop_size (int): Hop size for STFT.
        win_size (int): Window size for STFT.
        fmin (int): Minimum frequency for mel filterbank.
        fmax (int): Maximum frequency for mel filterbank. If None, defaults to half the sampling rate (fmax = sr / 2.0) inside librosa_mel_fn
        center (bool): Whether to pad the input to center the frames. Default is False.

    Returns:
        torch.Tensor: Mel spectrogram.
    """
    if torch.min(y) < -1.0:
        print(f"[WARNING] Min value of input waveform signal is {torch.min(y)}")
    if torch.max(y) > 1.0:
        print(f"[WARNING] Max value of input waveform signal is {torch.max(y)}")

    device = y.device
    key = f"{n_fft}_{num_mels}_{sampling_rate}_{hop_size}_{win_size}_{fmin}_{fmax}_{device}"

    if key not in mel_basis_cache:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)
        hann_window_cache[key] = torch.hann_window(win_size).to(device)

    mel_basis = mel_basis_cache[key]
    hann_window = hann_window_cache[key]

    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(
        y.unsqueeze(1), (padding, padding), mode="reflect"
    ).squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = spectral_normalize_torch(mel_spec)

    return mel_spec


def get_mel_spectrogram(wav, h):
    """
    Generate mel spectrogram from a waveform using given hyperparameters.

    Args:
        wav (torch.Tensor): Input waveform.
        h: Hyperparameters object with attributes n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax.

    Returns:
        torch.Tensor: Mel spectrogram.
    """
    return mel_spectrogram(
        wav,
        h.n_fft,
        h.num_mels,
        h.sampling_rate,
        h.hop_size,
        h.win_size,
        h.fmin,
        h.fmax,
    )


def get_dataset_filelist(a):
    training_files = []
    validation_files = []
    list_unseen_validation_files = []

    with open(a.input_training_file, "r", encoding="utf-8") as fi:
        training_files = [
            # os.path.join(a.input_wavs_dir, x.split("|")[0] + ".wav")
            x.split("|")[0] + ".wav"
            for x in fi.read().split("\n")
            if len(x) > 0
        ]
        print(f"first training file: {training_files[0]}")

    with open(a.input_validation_file, "r", encoding="utf-8") as fi:
        validation_files = [
            # os.path.join(a.input_wavs_dir, x.split("|")[0] + ".wav")
            x.split("|")[0] + ".wav"
            for x in fi.read().split("\n")
            if len(x) > 0
        ]
        print(f"first validation file: {validation_files[0]}")

    for i in range(len(a.list_input_unseen_validation_file)):
        with open(a.list_input_unseen_validation_file[i], "r", encoding="utf-8") as fi:
            unseen_validation_files = [
                # os.path.join(a.list_input_unseen_wavs_dir[i], x.split("|")[0] + ".wav")
                x.split("|")[0] + ".wav"
                for x in fi.read().split("\n")
                if len(x) > 0
            ]
            print(
                f"first unseen {i}th validation fileset: {unseen_validation_files[0]}"
            )
            list_unseen_validation_files.append(unseen_validation_files)

    return training_files, validation_files, list_unseen_validation_files



class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, audio_files, unit_file_path, utt2lang_filepath, lang_ids_json, spk_emb_scp_filepath, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax, unit_hop_size=320, num_kmeans_units=1000,
                 split=True,utt2f0_filepath=None, shuffle=False, fmax_loss=None, is_multispkr=False, is_f0=False, is_multilingual=False, **kwargs):

        super().__init__()
        self.audio_files = audio_files # list with all the audio paths
        
        self.utt_ids = [
            os.path.basename(x).replace(".wav", "")
            for x in self.audio_files
        ]

        self.unit_file_path = unit_file_path
        self.utt2lang_filepath = utt2lang_filepath #espnet format
        # self.utt2f0_filepath = utt2f0_filepath # espnet format
        self.lang_ids_json = lang_ids_json       
        self.split = split
        
        # Audio Params
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        
        # Unit Params
        self.unit_hop_size = unit_hop_size
        self.pad_id = num_kmeans_units
        self.frames_per_segment = math.ceil(self.segment_size / self.unit_hop_size)
         
        # Conditionals
        self.is_multispkr = is_multispkr
        self.is_f0 = is_f0
        self.is_multilingual = is_multilingual

        # 1. Load Speaker Map
        self.spk_map = {}
        if self.is_multispkr and spk_emb_scp_filepath:
             with open(spk_emb_scp_filepath, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    utt_id = parts[0]
                    if len(parts) >= 2:
                        self.spk_map[utt_id] = parts[1]
        import json
        if is_multilingual and lang_ids_json and utt2lang_filepath:
            with open(lang_ids_json, "r") as f:
                self.lang_ids_map = json.load(f)
            
            self.utt2lang = {}
            with open(utt2lang_filepath, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.utt2lang[parts[0]] = self.lang_ids_map[parts[1]] # we need to make sure that utt ids are consistent across all files :-)
        

        # 2. Load Units (Strict 1:1 alignment with audio_files expected) # modified to load units into a dict for faster access
        self.unit_map = {}
        with open(self.unit_file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                utt_id = parts[0]
                units = np.array([int(k) for k in parts[1:]], dtype=np.int32)
                self.unit_map[utt_id] = units
                
        missing = [u for u in self.utt_ids if u not in self.unit_map]
        if len(missing) > 0:
            raise ValueError(f"Missing unit entries for {len(missing)} utterances")
        
    #     self.utt2f0_map = {}
    #     if self.is_f0 and self.utt2f0_filepath:
    #         with open(self.utt2f0_filepath, "r", encoding="utf-8") as f:
    #             for line in f:
    #                 parts = line.strip().split()
    #                 utt_id = parts[0]
    #                 self.utt2f0_map[utt_id] = parts[1]  # path to f0 npy file

    # def _get_f0(self, utt_id):
    #     """Retrieves F0 array using the pre-loaded map."""
    #     if utt_id in self.utt2f0_map:
    #         try:
    #             return np.load(self.utt2f0_map[utt_id])
    #         except Exception as e:
    #             print(f"Error loading F0 for {utt_id}: {e}")
    #             return None
    #     return None

    def __getitem__(self, index):
        try:
            filename = self.audio_files[index]
            # unit_seq = self.unit_arr[index]
            utt_id = self.utt_ids[index]
            # print(filename,utt_id)
            unit_seq = self.unit_map[utt_id]
            # Load Audio
            audio, source_sr = librosa.load(filename, sr=None, mono=True)

            # Load Conditioningz
            if self.is_multispkr:
                utt_id = pathlib.Path(filename).stem # 4503599627705253_chunk_1_enhanced
                if utt_id in self.spk_map:
                    spk_emb = np.load(self.spk_map[utt_id])
                    spk_emb = torch.from_numpy(spk_emb).float().squeeze()
                    
            lang_id = None        
            if self.is_multilingual:
                # utt_id = pathlib.Path(filename).stem
                lang_id = self.utt2lang[utt_id]
            # f0_seq = None
            # if self.is_f0:
            #     f0_seq = self._get_f0(filename)
            #     if f0_seq is not None:
            #          # Basic safety resize
            #          if len(f0_seq) > len(unit_seq): f0_seq = f0_seq[:len(unit_seq)]
            #          elif len(f0_seq) < len(unit_seq): 
            #             f0_seq = np.pad(f0_seq, (0, len(unit_seq)-len(f0_seq)), mode='edge')
            #     else:
            #         f0_seq = np.zeros_like(unit_seq, dtype=np.float32)

            # VALIDATION (Full Sequence) 
            if not self.split:
                unit_seq = torch.LongTensor(unit_seq)
                # unit_seq = torch.repeat_interleave(unit_seq, repeats=2, dim=0)# not required, we adjusted the output_padding and upsample rates for aqc units
                if source_sr != self.sampling_rate:
                    audio = librosa.resample(audio, orig_sr=source_sr, target_sr=self.sampling_rate)
                
                audio = librosa.util.normalize(audio) * 0.95
                
                # Align audio length to units
                target_samples = unit_seq.size(0) * self.unit_hop_size
                if audio.shape[0] < target_samples:
                    audio = np.pad(audio, (0, target_samples - audio.shape[0]), mode='constant')
                else:
                    audio = audio[:target_samples]
                
                audio = torch.FloatTensor(audio).unsqueeze(0) # [1, T]

                mel = mel_spectrogram(audio, self.n_fft, self.num_mels, self.sampling_rate,
                                      self.hop_size, self.win_size, self.fmin, self.fmax_loss, center=False)
                # print(f"{filename=}")
                ret = {"code": unit_seq, "audio": audio.squeeze(0), "mel": mel.squeeze(0),"filenames":filename}
                if self.is_multispkr: ret["spkr"] = spk_emb
                if self.is_multilingual: ret["lang_ids"] = torch.LongTensor([lang_id]).squeeze()
                # if self.is_f0: ret["f0"] = torch.FloatTensor(f0_seq)
                return ret

            #TRAINING (Random Crop)
            else:
                # Chunk Units
                if len(unit_seq) >= self.frames_per_segment:
                    max_start = len(unit_seq) - self.frames_per_segment
                    start_idx = random.randint(0, max_start)
                    unit_chunk = unit_seq[start_idx : start_idx + self.frames_per_segment]
                    # if self.is_f0: f0_chunk = f0_seq[start_idx : start_idx + self.frames_per_segment]
                else:
                    pad_amount = self.frames_per_segment - len(unit_seq)
                    unit_chunk = np.pad(unit_seq, (0, pad_amount), mode='constant', constant_values=self.pad_id)
                    start_idx = 0
                    # if self.is_f0: f0_chunk = np.pad(f0_seq, (0, pad_amount), mode='constant')

                unit_chunk = torch.LongTensor(unit_chunk)
                # unit_chunk = torch.repeat_interleave(unit_chunk, repeats=2, dim=0)


                # Chunk Audio
                resample_ratio = source_sr / self.sampling_rate
                audio_hop_source = int(self.unit_hop_size * resample_ratio)
                target_seg_source = int(self.segment_size * resample_ratio)
                
                audio_start = start_idx * audio_hop_source
                audio_end = audio_start + target_seg_source

                if audio.shape[0] >= audio_end:
                    audio_chunk = audio[audio_start:audio_end]
                else:
                    audio_chunk = np.pad(audio[audio_start:], (0, max(0, audio_end - audio.shape[0])), mode='constant')

                if source_sr != self.sampling_rate and audio_chunk.size > 0:
                    audio_chunk = librosa.resample(audio_chunk, orig_sr=source_sr, target_sr=self.sampling_rate)
                
                # Strict Length
                if audio_chunk.shape[0] > self.segment_size:
                    audio_chunk = audio_chunk[:self.segment_size]
                elif audio_chunk.shape[0] < self.segment_size:
                    audio_chunk = np.pad(audio_chunk, (0, self.segment_size - audio_chunk.shape[0]), mode='constant')
                
                audio_chunk = librosa.util.normalize(audio_chunk) * 0.95
                audio_chunk = torch.FloatTensor(audio_chunk).unsqueeze(0)

                mel = mel_spectrogram(audio_chunk, self.n_fft, self.num_mels, self.sampling_rate,
                                      self.hop_size, self.win_size, self.fmin, self.fmax_loss, center=False)

                ret = {"code": unit_chunk, "audio": audio_chunk.squeeze(0), "mel": mel.squeeze(0), "filenames":filename}
                if self.is_multispkr: ret["spkr"] = spk_emb
                if self.is_multilingual: ret["lang_ids"] = torch.LongTensor([lang_id]).squeeze()
                # if self.is_f0: ret["f0"] = torch.FloatTensor(f0_chunk)
                return ret

        except Exception as e:
            print(f"Error loading {self.audio_files[index]}: {e}")
            return self.__getitem__(random.randint(0, len(self.audio_files) - 1))

    def __len__(self):
        return len(self.audio_files)