#!/bin/bash

python run_vad.py \
    --manifest_file "/nlp/scr/askhan1/github_files/edm25-nemo-classroom-diarization/manifests/CLASSBANK_test_manifest_2s_noisy.json" \
    --checkpoint_path "/nlp/scr/askhan1/train_umd_cb_both_robust-large/checkpoints-noisy/ckpt-1.pt" \
    --vad_manifest_path "null.json" \
    --frames_output_path "./vad_output_frames"
