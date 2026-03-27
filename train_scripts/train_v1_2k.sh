#!/bin/bash

python train.py --config /speech/naman/exps/bigvgan/CodeBigvgan/config_v1_2k.json \
    --checkpoint_path /speech/naman/exps/bigvgan/CodeBigvgan/checkpoints/v1/2k \
    --input_wavs_dir /speech/naman/data/indicvoices-r \
    --input_training_file /speech/naman/data/indicvoices-r/bigvgan_data/training.txt \
    --input_validation_file /speech/naman/data/indicvoices-r/bigvgan_data/validation.txt \
    --list_input_unseen_validation_file /speech/naman/data/indicvoices-r/bigvgan_data/testing.txt \
    --list_input_unseen_wavs_dir /speech/naman/data/indicvoices-r \
    --training_unit_file /speech/naman/data/indicvoices-r/units/combined_2k/train_units/units_uc \
    --validation_unit_file /speech/naman/data/indicvoices-r/units/combined_2k/valid_units/units_uc \
    --unseen_validation_unit_file /speech/naman/data/indicvoices-r/units/combined_2k/test_units/units_uc \
    --checkpoint_interval 10000 \
