import os
import json
import shutil

manifest_path = "/nlp/scr/askhan1/github_files/edm25-nemo-classroom-diarization/manifests/CLASSBANK_test_manifest_all_speaker_noisy.json"
audio_out_dir = "/nlp/scr/askhan1/github_files/edm25-nemo-classroom-diarization/classbank_audio_data/audio"
rttm_out_dir = "/nlp/scr/askhan1/github_files/edm25-nemo-classroom-diarization/classbank_audio_data/rttm"
updated_manifest_path = "/nlp/scr/askhan1/github_files/edm25-nemo-classroom-diarization/manifests/CLASSBANK_test_manifest_2s_noisy_relpaths.json"

# Create output directories if they don't exist
os.makedirs(audio_out_dir, exist_ok=True)
os.makedirs(rttm_out_dir, exist_ok=True)

updated_entries = []

with open(manifest_path, "r") as f:
    for line in f:
        entry = json.loads(line)
        audio_path = entry["audio_filepath"]
        rttm_path = entry["rttm_filepath"]

        audio_basename = os.path.basename(audio_path)
        audio_relpath = f"../classbank_audio_data/audio/{audio_basename}"

        if os.path.exists(audio_path):
            shutil.copy(audio_path, os.path.join(audio_out_dir, audio_basename))
        else:
            print(f"Missing audio file: {audio_path}")

        entry["audio_filepath"] = audio_relpath

        if rttm_path:
            rttm_basename = os.path.basename(rttm_path)
            rttm_relpath = f"../classbank_audio_data/rttm/{rttm_basename}"

            if os.path.exists(rttm_path):
                shutil.copy(rttm_path, os.path.join(rttm_out_dir, rttm_basename))
                entry["rttm_filepath"] = rttm_relpath
            else:
                print(f"Missing RTTM file: {rttm_path}")
                entry["rttm_filepath"] = None

        updated_entries.append(entry)

# Save updated manifest
with open(updated_manifest_path, "w") as f:
    for entry in updated_entries:
        f.write(json.dumps(entry) + "\n")
