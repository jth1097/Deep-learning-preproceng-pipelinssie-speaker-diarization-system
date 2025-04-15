#!/bin/bash

# Name of the file where we'll store the summary of FA results
RESULTS_FILE="test_results.txt"
# Temporary log file to capture the diarizer output
DIARIZER_LOG="diarizer_temp.log"
MANIFEST_FILE="/nlp/scr/askhan1/github_files/edm25-nemo-classroom-diarization/manifests/classbank_sample_manifest.json"
VAD_OUTPUT="/nlp/scr/askhan1/github_files/edm25-nemo-classroom-diarization/generate_w2v2_speech_labels/vad_output_frames"
WHISPER_OUTPUT="/nlp/scr/askhan1/github_files/edm25-nemo-classroom-diarization/generate_whisper_speech_labels/output"
DIAR_OUTPUT_DIR="test"
# Clear any old results file
rm -f "${RESULTS_FILE}"

# Loop over multiple parameter sets if you want best performing params

#   for offset in 0.2 0.25 0.3 0.35 0.45 0.5 
#   do
#     for onset in 0.7 0.65 0.6 0.55 0.50 0.45 0.4 0.35 0.3 0


for alpha in 0.0 0.25 0.5 0.75 1.0
do
  for offset in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2
  do
    for onset in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2
    do
        if (( $(echo "$onset > $offset" | bc -l) )); then
            echo "--------------------------------------------------"
            echo "Running with alpha=${alpha}, onset=${onset}, offset=${offset}"
            echo "--------------------------------------------------"

            rm -f ./vad_outs.json
            rm -rf ./w2v_res_w_asr.json
            rm -rf pymp-* tmp* torchelastic*

            # 1) Run the VAD parameter tuning script to produce your VAD segments JSON
            python tune_vad_params.py \
                --manifest_file="${MANIFEST_FILE}" \
                --frame_dir="${VAD_OUTPUT}" \
                --asr_dir="${WHISPER_OUTPUT}" \
                --alpha="${alpha}" \
                --onset="${onset}" \
                --offset="${offset}" \
                --out_dir='./vad_outs.json'

            # 2) Run the diarizer, pointing to the newly created VAD segments JSON
            #    We capture stdout/stderr to a temp log file so we can parse it.
            python3 ../NeMo/examples/speaker_tasks/diarization/clustering_diarizer/offline_diar_infer.py \
                diarizer.manifest_filepath="${MANIFEST_FILE}" \
                diarizer.out_dir="${DIAR_OUTPUT_DIR}" \
                diarizer.vad.model_path=null \
                diarizer.vad.external_vad_manifest='./vad_outs.json' \
                diarizer.speaker_embeddings.parameters.save_embeddings=False \
                diarizer.speaker_embeddings.model_path='titanet_large' \
                diarizer.clustering.parameters.oracle_num_speakers=True \
                2>&1 | tee "${DIARIZER_LOG}"

            # 3) Parse the diarizer's output log for the line(s) that start with "| FA:"
            #    (Adjust this pattern if your lines differ, e.g. "FA (gt=0..." or other prefixes.)
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
