#!/bin/bash

RESULTS_FILE="DER_results_all_speaker.txt" # Name of the file where we'll store the summary of FA results
DIARIZER_LOG="diarizer_temp.log" # Temporary log file to capture the diarizer output
CKPT="/nlp/scr/askhan1/train_ahmed_hf/checkpoints-noisy/ckpt-1.pt"
MANIFEST_FILE="/nlp/scr/askhan1/clipped_umd/UMD_clipped_test_manifest_all_speaker_noisy.json"

# output directories
VAD_OUTPUT="vad_output_frames"
WHISPER_OUTPUT="whisper_output_frames"
DIAR_OUTPUT_DIR="diarization_output"

# VAD frames
python generate_w2v2_speech_labels/run_vad.py \
    --manifest_file "${MANIFEST_FILE}" \
    --checkpoint_path "${CKPT}" \
    --vad_manifest_path "null.json" \
    --frames_output_path "${VAD_OUTPUT}"

# whisper frames
python generate_whisper_speech_labels/whisper_transcribe.py \
    --manifest_file "${MANIFEST_FILE}" \
    --output_dir "${WHISPER_OUTPUT}"


# Combine outputs
# Clear any old results file
rm -f "${RESULTS_FILE}"
# $(seq 0.10 0.20 1.0)
for alpha in $(seq 0.20 0.20 1.0)
do
  for offset in $(seq 0.10 0.05 0.80)
  do
    for onset in $(seq 0.30 0.05 0.90)
    do
        if (( $(echo "$onset > $offset" | bc -l) )); then
            echo "--------------------------------------------------"
            echo "Running with alpha=${alpha}, onset=${onset}, offset=${offset}"
            echo "--------------------------------------------------"

            rm -f ./vad_outs.json
            rm -rf ./w2v_res_w_asr.json
            rm -rf pymp-* tmp* torchelastic*

            # 1) run the VAD parameter tuning script to produce your VAD segments JSON
            python run_diarization/tune_vad_params.py \
                --manifest_file="${MANIFEST_FILE}" \
                --frame_dir="${VAD_OUTPUT}" \
                --asr_dir="${WHISPER_OUTPUT}" \
                --alpha="${alpha}" \
                --onset="${onset}" \
                --offset="${offset}" \
                --out_dir='./vad_outs.json'

            # 2) run the diarizer, pointing to the newly created VAD segments JSON
            #    We capture stdout/stderr to a temp log file so we can parse it.
            python3 NeMo/examples/speaker_tasks/diarization/clustering_diarizer/offline_diar_infer.py \
                diarizer.manifest_filepath="${MANIFEST_FILE}" \
                diarizer.out_dir="${DIAR_OUTPUT_DIR}" \
                diarizer.vad.model_path=null \
                diarizer.vad.external_vad_manifest='./vad_outs.json' \
                diarizer.speaker_embeddings.parameters.save_embeddings=False \
                diarizer.speaker_embeddings.model_path='titanet_large' \
                diarizer.clustering.parameters.oracle_num_speakers=True \
                2>&1 | tee "${DIARIZER_LOG}"

            # 3) Parse the diarizer's output log for the line(s) that start with "| FA:"
            FA_LINE=$(grep '| FA' "${DIARIZER_LOG}")

            # 4) Append the results to a summary file, including the parameters used
            if [ -n "$FA_LINE" ]; then
                echo "alpha=${alpha}, onset=${onset}, offset=${offset}, ${FA_LINE}" >> "${RESULTS_FILE}"
            else
                # If no line found, optionally record that
                echo "alpha=${alpha}, onset=${onset}, offset=${offset}, NO_FA_LINE_FOUND" >> "${RESULTS_FILE}"
            fi
        fi
    done
  done
done

echo "All experiments completed. Results stored in ${RESULTS_FILE}."
