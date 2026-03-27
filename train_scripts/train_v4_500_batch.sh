#!/bin/bash
#SBATCH -J v4_1k
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu 16
#SBATCH --gres=gpu:A100-SXM4:4
#SBATCH --partition=dibdp
#SBATCH --time=7-00:00:00
#SBATCH --error=/nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/logs/v4/1k/job.%J.err
#SBATCH --output=/nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/logs/v4/1k/job.%J.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=namankothari.10.nk@gmail.com
#SBATCH --export=ALL,JOB_DESCRIPTION="This job is to train longform 2B automatic speech recognition bilingual model combining english longform data and tamil longform data with punctuation and 0.5 threshold of data conversion ideally to get better performance in the transcription generated along with english words code switched or code mixed also with the punctuation to make the output much better to read and can be used for translation purposes",EXPECTED_OUTCOME="This job is to train longform 2B automatic speech recognition bilingual model combining english longform data and tamil longform data with punctuation and 0.5 threshold of data conversion ideally to get better performance in the transcription generated along with english words code switched or code mixed also with the punctuation to make the output much better to read and can be used for translation purposes"


source /nlsasfs/home/dibd/dibd-speech/iitm/laish/miniconda3/etc/profile.d/conda.sh
conda activate /nlsasfs/home/dibd/dibd-speech/iitm/laish/miniconda3/envs/bigvgan

python /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/train.py --config /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/configs_codebigvgan/config_v4_500.json \
    --checkpoint_path /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/checkpoints/v4/500 \
    --input_wavs_dir /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr \
    --input_training_file /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/bigvgan_format/train_files.txt \
    --input_validation_file /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/bigvgan_format/val_files.txt \
    --list_input_unseen_validation_file /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/bigvgan_format/test_files.txt \
    --list_input_unseen_wavs_dir /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr \
    --training_unit_file /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/units_aqc/kmeans_500/combined_multilingual_train/units_uc \
    --validation_unit_file /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/units_aqc/kmeans_500/combined_multilingual_dev/units_uc \
    --unseen_validation_unit_file /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/units_aqc/kmeans_500/combined_multilingual_test/units_uc \
    --training_utt2lang_path /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/espnet_format/combined/train/utt2lang \
    --validation_utt2lang_path /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/espnet_format/combined/dev/utt2lang \
    --unseen_validation_utt2lang_path /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/espnet_format/combined/test/utt2lang \
    --lang_ids_json_path /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/lang_ids_map.json \
    --checkpoint_interval 20000 \
    --validation_interval 50000 \
    --skip_seen true