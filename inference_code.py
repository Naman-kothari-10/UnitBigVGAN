# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse
import json
import torch
import librosa
import numpy as np
from utils import load_checkpoint
from meldataset import get_mel_spectrogram
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE

from CodeBigvgan import CodeGenerator



h = None
device = None
torch.backends.cudnn.benchmark = False

def inference(a,h): # h-> config
    generator = CodeGenerator(h, use_cuda_kernel=a.use_cuda_kernel).to(device)
    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g["generator"])
    
    is_multispkr = h.get("is_multispkr", False)
    is_multilingual = h.get("is_multilingual", False)
    lang_ids_json = a.lang_ids_json
    utt2lang_filepath = a.utt2lang_filepath
    
    with open(a.unit_filepath, "r") as f:
        lines = f.readlines()
        unit_arr = [
            np.array([int(k) for k in lines[i].split()[1:]], dtype=np.int32)
            for i in range(len(lines))
        ]
        filelist = [line.split()[0]
            for line in lines
        ]
        
        
    if is_multispkr and a.ecapa_tdnn_spk_emb_scp_file:
        spk_map = {}
        with open(a.ecapa_tdnn_spk_emb_scp_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                utt_id = parts[0] # Bengali_1407374883553505_chunk_6_enhanced -> 1407374883553505_chunk_6_enhanced
                if len(parts) >= 2:
                    spk_map[utt_id] = parts[1]
    
    if is_multilingual and lang_ids_json and utt2lang_filepath:
            with open(lang_ids_json, "r") as f:
                lang_ids_map = json.load(f)

            utt2lang_map = {}
            with open(utt2lang_filepath, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        utt2lang_map[parts[0]] = lang_ids_map[parts[1]] # we need to make sure that utt ids are consistent across all files :-)
        
    os.makedirs(a.output_dir, exist_ok=True)
    
    generator.eval()
    # generator.remove_weight_norm()
    with torch.no_grad():
        
        for i, (filename,unit_seq) in enumerate(zip(filelist, unit_arr)):
            
            wav, sr = librosa.load(
                os.path.join(a.input_wavs_dir, filename + ".wav") , sr=h.sampling_rate, mono=True
            )
            wav = torch.FloatTensor(wav).to(device)
            
            unit_seq = torch.LongTensor(unit_seq).unsqueeze(0).to(device)
            lang_id = torch.LongTensor([utt2lang_map[filename]]).to(device) if is_multilingual else None
            ret = {"code": unit_seq, "filenames": filename}
            if is_multilingual:
                ret["lang_ids"] = lang_id
            if is_multispkr:
                spk_emb = np.load(spk_map[filename]) # file name is same as utt_id
                # print(f"Spk emb shape: {spk_emb.shape}")
                spk_emb = torch.from_numpy(spk_emb).float().to(device)
                ret["spkr"] = spk_emb
            
            y_g_hat = generator(**ret)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype("int16")
            
            output_file = os.path.join(a.output_dir, filename + ".wav")
            write(output_file, h.sampling_rate, audio)
            print(output_file)
            
def main():
    print("Initializing Inference Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_wavs_dir", default="test_files")
    parser.add_argument("--unit_filepath",required=True)
    parser.add_argument("--config_filepath", required=True)
    parser.add_argument("--output_dir", default="generated_files")
    parser.add_argument("--use_cuda_kernel", action="store_true", default=False)
    parser.add_argument("--checkpoint_file", required=True)
    parser.add_argument("--ecapa_tdnn_spk_emb_scp_file",default="None", required=False)
    parser.add_argument("--lang_ids_json", default = None)
    parser.add_argument("--utt2lang_filepath", default = None)

    a = parser.parse_args()

    config_file = a.config_filepath
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    inference(a, h)


if __name__ == "__main__":
    main()
            
            
            
            