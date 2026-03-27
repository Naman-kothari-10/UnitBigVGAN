import os
import numpy as np
import soundfile as sf
import pyworld as pw
import argparse
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
import resampy

# ==========================================
# CONFIGURATION
# ==========================================
TARGET_LANGUAGES = ["Hindi", "Bengali", "Telugu", "Tamil"] 
TARGET_SPLITS = ["dev", "train", "test"] 

# ==========================================
# CORE SIGNAL PROCESSING (Optimized)
# ==========================================

def quantize_f0_log(f0, n_bins=256, min_f0=50.0, max_f0=800.0):
    f0_indices = np.zeros_like(f0, dtype=np.int64)
    uv_mask = (f0 <= 0)
    voiced_mask = ~uv_mask
    
    if np.any(voiced_mask):
        voiced_f0 = np.clip(f0[voiced_mask], min_f0, max_f0)
        log_f0 = np.log(voiced_f0)
        log_min = np.log(min_f0)
        log_max = np.log(max_f0)
        bins = np.linspace(log_min, log_max, 255)
        indices = np.digitize(log_f0, bins)
        indices = indices + 1
        f0_indices[voiced_mask] = indices
    
    f0_indices = np.clip(f0_indices, 0, n_bins - 1)
    return f0_indices

def extract_and_align_pitch(wav_path, unit_hop_size=320, sr=16000):
    try:
        # OPTIMIZATION: Use SoundFile instead of Librosa
        # Soundfile is 10x-50x faster for standard WAVs
        y, orig_sr = sf.read(wav_path)
        
        # Handle Multi-channel (Stereo -> Mono)
        if y.ndim > 1:
            y = y.mean(axis=1)
            
        # Resample only if necessary
        if orig_sr != sr:
            # Low-quality filter is fine for Pitch Extraction and much faster
            y = resampy.resample(y, orig_sr, sr, filter='kaiser_fast')
            
        y = y.astype(np.float64)
        
        # Exact Target Frames
        expected_frames = len(y) // unit_hop_size
        if expected_frames == 0: return None

        # Extract Pitch (HARVEST)
        frame_period = (unit_hop_size / sr) * 1000
        _f0, t = pw.harvest(y, sr, frame_period=frame_period, f0_floor=50.0, f0_ceil=800.0)
        f0 = pw.stonemask(y, _f0, t, sr)
        
        # Align
        curr_frames = len(f0)
        if curr_frames > expected_frames:
            f0 = f0[:expected_frames]
        elif curr_frames < expected_frames:
            pad_amount = expected_frames - curr_frames
            f0 = np.pad(f0, (0, pad_amount), mode='constant', constant_values=0)

        # Quantize
        return quantize_f0_log(f0)

    except Exception as e:
        return None

# ==========================================
# WORKER
# ==========================================

def process_file_wrapper(task):
    utt_id, wav_path_raw, scp_dir, output_dir, unit_hop_size, sr = task
    
    save_path = os.path.join(output_dir, f"{utt_id}.npy")
    if os.path.exists(save_path):
        return
        
    wav_path = wav_path_raw
    if not os.path.exists(wav_path):
        alt_path = os.path.join(scp_dir, wav_path)
        if os.path.exists(alt_path):
            wav_path = alt_path
        else:
            return 

    f0_tokens = extract_and_align_pitch(wav_path, unit_hop_size, sr)
    
    if f0_tokens is not None:
        np.save(save_path, f0_tokens)

# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Path to 'dump/raw'")
    parser.add_argument("--unit_hop_size", type=int, default=320)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_jobs", type=int, default=64, help="Try 32-64")
    args = parser.parse_args()

    base_path = Path(args.base_dir)
    tasks = []

    print(">>> Scanning Directory Structure & Collecting Tasks...")

    for lang in TARGET_LANGUAGES:
        lang_path = base_path / lang
        if not lang_path.exists(): continue
            
        for split in TARGET_SPLITS:
            split_path = lang_path / split
            wav_scp = split_path / "wav.scp"
            
            if wav_scp.exists():
                output_dir = split_path / "f0"
                os.makedirs(output_dir, exist_ok=True)
                
                with open(wav_scp, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                scp_dir = str(split_path) 
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 2: continue
                    
                    utt_id = parts[0]
                    wav_path_raw = parts[1]
                    tasks.append((utt_id, wav_path_raw, scp_dir, str(output_dir), args.unit_hop_size, args.sr))

    print(f">>> Found {len(tasks)} audio files to process.")
    print(f">>> Starting Multiprocessing with {args.n_jobs} cores (Optimized I/O).")
    
    # Chunksize helps reduce communication overhead on slow networks
    with Pool(processes=args.n_jobs) as pool:
        for _ in tqdm(pool.imap_unordered(process_file_wrapper, tasks, chunksize=10), total=len(tasks), unit="file"):
            pass
            
    print(">>> Processing Complete.")

if __name__ == "__main__":
    main()