#!/bin/bash
#SBATCH -J v2_500
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu 16
#SBATCH --gres=gpu:A100-SXM4:4
#SBATCH --partition=dibdp
#SBATCH --time=7-00:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=<email>
#SBATCH --export=ALL,JOB_DESCRIPTION="This job is to train longform 2B automatic speech recognition bilingual model combining english longform data and tamil longform data with punctuation and 0.5 threshold of data conversion ideally to get better performance in the transcription generated along with english words code switched or code mixed also with the punctuation to make the output much better to read and can be used for translation purposes",EXPECTED_OUTCOME="This job is to train longform 2B automatic speech recognition bilingual model combining english longform data and tamil longform data with punctuation and 0.5 threshold of data conversion ideally to get better performance in the transcription generated along with english words code switched or code mixed also with the punctuation to make the output much better to read and can be used for translation purposes"

source /nlsasfs/home/dibd/dibd-speech/iitm/laish/miniconda3/etc/profile.d/conda.sh
conda activate /nlsasfs/home/dibd/dibd-speech/iitm/laish/miniconda3/envs/bigvgan

python /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/train.py --config /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/configs_codebigvgan/config_v2_500.json \
    --checkpoint_path /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/checkpoints/v2/500 \
    --input_wavs_dir /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr \
    --input_training_file /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/bigvgan_format/train_files.txt \
    --input_validation_file /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/bigvgan_format/val_files.txt \
    --list_input_unseen_validation_file /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/bigvgan_format/test_files.txt \
    --list_input_unseen_wavs_dir /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr \
    --training_unit_file /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/units_aqc/kmeans_500/combined_multilingual_train/units_uc \
    --validation_unit_file /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/units_aqc/kmeans_500/combined_multilingual_dev/units_uc \
    --unseen_validation_unit_file /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/units_aqc/kmeans_500/combined_multilingual_test/units_uc \
    --ecapa_tdnn_spk_emb_scp_file /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/ecapa_tdnn_spk_emb_bsz1/train_dev/ecapa_tdnn_emb.scp \
    --checkpoint_interval 20000 \
    --validation_interval 100000 \
    --skip_seen true