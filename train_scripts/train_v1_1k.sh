#!/bin/bash

python /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/train.py --config /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/configs_codebigvgan/config_v1_1k.json \
    --checkpoint_path /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/checkpoints/v1/1k \
    --input_wavs_dir /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr \
    --input_training_file /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/bigvgan_format/train_files.txt \
    --input_validation_file /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/bigvgan_format/val_files.txt \
    --list_input_unseen_validation_file /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/bigvgan_format/test_files.txt \
    --list_input_unseen_wavs_dir /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr \
    --training_unit_file /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/units_aqc/kmeans_1000/combined_multilingual_train/units_uc \
    --validation_unit_file /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/units_aqc/kmeans_1000/combined_multilingual_dev/units_uc \
    --unseen_validation_unit_file /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/units_aqc/kmeans_1000/combined_multilingual_test/units_uc \
    --checkpoint_interval 10000 \
    --validation_interval 50000
