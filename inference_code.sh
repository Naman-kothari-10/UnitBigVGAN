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
    OUTPUT_DIR="/nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/inference_outputs/v3/5k/cross_lingual_indic/${LANG_NAME}"
    CHKPT_PATH="/nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/checkpoints/v3/5k/g_00400000"
    UNIT_FILEPATH="/nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/units_aqc/kmeans_5000/${LANG_NAME}/test/units_uc"
    CONFIG_FILEPATH="/nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/configs_codebigvgan/config_v3_5k.json"
    SPK_EMB_FILEPATH="/nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/data/ivr/ecapa_tdnn_spk_emb_bsz1/test/ecapa_tdnn_emb.scp"

    python /nlsasfs/home/dibd/dibd-speech/iitm/laish/naman/exps/CodeBigvgan/inference_code.py \
      --input_wavs_dir "$INPUT_WAVS_DIR" \
      --output_dir "$OUTPUT_DIR" \
      --checkpoint_file "$CHKPT_PATH" \
      --unit_filepath "$UNIT_FILEPATH" \
      --config_filepath "$CONFIG_FILEPATH" \
      --ecapa_tdnn_spk_emb_scp_file "$SPK_EMB_FILEPATH"
done