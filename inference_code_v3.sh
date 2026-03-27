#!/bin/bash
set -e

# Define languages as "LanguageName/code" pairs
LANGS="Bengali/ben Telugu/tel Hindi/hin Tamil/tam" 
# LANGS="Gujarati/guj Marathi/mar Kannada/kan Malayalam/mal Kashmiri/kas Bodo/bod Dogri/dog Santali/san Maithili/mai Manipuri/man"
for lang_pair in $LANGS; do
    # Split the pair into Name and Code
    LANG_NAME=${lang_pair%/*}
    LANG_CODE=${lang_pair#*/}
    
    echo "Running inference for $LANG_NAME ($LANG_CODE)..."

    INPUT_WAVS_DIR="/nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/${LANG_NAME}/test/wavs_16khz"
    OUTPUT_DIR="/nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/inference_outputs/v3/500_mel/${LANG_NAME}"
    # OUTPUT_DIR="/nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/inference_outputs_cross_lingual_spk_mixing/hin_tel__ben_tam/v3/2k_mel/${LANG_NAME}"
    CHKPT_PATH="/nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/checkpoints/v3/500_mel/g_00400000"
    UNIT_FILEPATH="/nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/units_aqc/kmeans_500/${LANG_NAME}/test/units_uc"
    CONFIG_FILEPATH="/nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/configs_codebigvgan/config_v3_500.json"
    SPK_EMB_FILEPATH="/nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/ecapa_tdnn_spk_emb_bsz1/test/ecapa_tdnn_emb.scp"
    # SPK_EMB_FILEPATH="/nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/ecapa_tdnn_spk_emb_bsz1/test/cross_lingual_shuffled/ecapa_tdnn_emb_cross_lingual_hin_ben__tam_tel.scp"
    # SPK_EMB_FILEPATH="/nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/ecapa_tdnn_spk_emb_bsz1/test/cross_lingual_shuffled/ecapa_tdnn_emb_cross_lingual_hin_tel__ben_tam.scp"
    LANG_IDS_JSON="/nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/lang_ids_map.json"
    UTT2LANG="/nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/espnet_format/${LANG_NAME}/test/utt2lang"

    python /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/inference_code.py \
      --input_wavs_dir "$INPUT_WAVS_DIR" \
      --output_dir "$OUTPUT_DIR" \
      --checkpoint_file "$CHKPT_PATH" \
      --unit_filepath "$UNIT_FILEPATH" \
      --config_filepath "$CONFIG_FILEPATH" \
      --ecapa_tdnn_spk_emb_scp_file "$SPK_EMB_FILEPATH" \
      --lang_ids_json "$LANG_IDS_JSON" \
      --utt2lang_filepath "$UTT2LANG" 
done