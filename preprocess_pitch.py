import os
import numpy as np
import librosa
import pyworld as pw
import argparse
from tqdm import tqdm
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
TARGET_LANGUAGES = ["Hindi", "Bengali", "Telugu", "Tamil"] 
TARGET_SPLITS = ["dev", "train", "test"] 

# ==========================================
# CORE SIGNAL PROCESSING
# ==========================================

def quantize_f0_log(f0, n_bins=256, min_f0=50.0, max_f0=800.0):
    """
    Production-Ready Log Quantization.
    - 0: Unvoiced
    - 1..255: Voiced (Log Scale)
    """
    # Initialize with 0 (Unvoiced)
    f0_indices = np.zeros_like(f0, dtype=np.int64)
    
    # Identify Voiced Frames (Pitch > 0)
    # We ignore 0s (Unvoiced) during log calculation to be clean
    uv_mask = (f0 <= 0)
    voiced_mask = ~uv_mask
    
    if np.any(voiced_mask):
        # 1. Clamp voiced values to min/max range for safety
        voiced_f0 = np.clip(f0[voiced_mask], min_f0, max_f0)
        
        # 2. Log Scale
        log_f0 = np.log(voiced_f0)
        log_min = np.log(min_f0)
        log_max = np.log(max_f0)
        
        # 3. Create Bins
        # We want to map voiced frames to indices 1..255 (255 buckets)
        # using 255 boundaries gives us 256 regions (0..255) from digitize.
        bins = np.linspace(log_min, log_max, 255)
        
        # 4. Digitize
        # Indices will be 0..255
        # 0 means < min_f0
        # 255 means >= max_f0
        indices = np.digitize(log_f0, bins)
        
        # 5. Shift to 1..256 range? 
        # Actually, let's keep it simple: 
        # Map 0..254 -> 1..255. Clamp the top.
        # We add 1 because index 0 is reserved for unvoiced.
        indices = indices + 1
        
        # 6. Assign to the output array
        f0_indices[voiced_mask] = indices
    
    # 7. Final Safety Clip (Ensure we stay in [0, 255])
    f0_indices = np.clip(f0_indices, 0, n_bins - 1)
    
    return f0_indices

def extract_and_align_pitch(wav_path, unit_hop_size=320, sr=16000):
    try:
        # 1. Load Audio
        y, _ = librosa.load(wav_path, sr=sr)
        y = y.astype(np.float64)
        
        # 2. Calculate Exact Target Frames
        expected_frames = len(y) // unit_hop_size
        
        if expected_frames == 0:
            return None

        # 3. Extract Pitch using HARVEST
        frame_period = (unit_hop_size / sr) * 1000
        
        # Harvest is robust. f0_floor/ceil prevents octave errors.
        _f0, t = pw.harvest(y, sr, frame_period=frame_period, f0_floor=50.0, f0_ceil=800.0)
        f0 = pw.stonemask(y, _f0, t, sr)
        
        # 4. ALIGNMENT (Trim/Pad Strategy)
        curr_frames = len(f0)
        
        if curr_frames > expected_frames:
            # Trim
            f0 = f0[:expected_frames]
        elif curr_frames < expected_frames:
            # Pad with 0 (Unvoiced)
            pad_amount = expected_frames - curr_frames
            f0 = np.pad(f0, (0, pad_amount), mode='constant', constant_values=0)

        # 5. Quantize
        f0_tokens = quantize_f0_log(f0)
        
        return f0_tokens

    except Exception as e:
        print(f"[Error] Processing {wav_path}: {e}")
        return None

# ==========================================
# DIRECTORY PROCESSING
# ==========================================

def process_scp_file(scp_path, output_dir, unit_hop_size, sr):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(scp_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Processing {len(lines)} files from {scp_path}...")
    
    for line in tqdm(lines):
        parts = line.strip().split()
        if len(parts) < 2: continue
            
        utt_id = parts[0]
        wav_path = parts[1]
        
        # Save as utt_id.npy in the f0 folder
        save_path = os.path.join(output_dir, f"{utt_id}.npy")
        
        if os.path.exists(save_path): continue

        # Resolve path
        if not os.path.exists(wav_path):
            alt_path = os.path.join(os.path.dirname(scp_path), wav_path)
            if os.path.exists(alt_path):
                wav_path = alt_path
            else:
                continue

        f0_tokens = extract_and_align_pitch(wav_path, unit_hop_size, sr)
        
        if f0_tokens is not None:
            np.save(save_path, f0_tokens)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Path to 'dump/raw' or similar")
    parser.add_argument("--unit_hop_size", type=int, default=320)
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    base_path = Path(args.base_dir)

    for lang in TARGET_LANGUAGES:
        lang_path = base_path / lang
        if not lang_path.exists(): continue
            
        print(f"\n>>> Language: {lang}")
        for split in TARGET_SPLITS:
            split_path = lang_path / split
            wav_scp = split_path / "wav.scp"
            
            if wav_scp.exists():
                # Creates .../Language/train/f0/
                output_dir = split_path / "f0"
                process_scp_file(wav_scp, output_dir, args.unit_hop_size, args.sr)

if __name__ == "__main__":
    main()