python /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/train.py --config /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/config_mel_all_24.yaml \
    --checkpoint_path /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/checkpoints/all_24_mel/ \
    --input_wavs_dir /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/all_24_symlink \
    --input_training_file /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/all_24_tts_data_bigvgan_format/train_files_symlink.txt \
    --input_validation_file /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/all_24_tts_data_bigvgan_format/test_files_symlink.txt \
    --list_input_unseen_validation_file //nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/all_24_tts_data_bigvgan_format/test_files_symlink.txt \
    --list_input_unseen_wavs_dir /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/all_24_symlink \
    --checkpoint_interval 5000 \
    --validation_interval 100000
