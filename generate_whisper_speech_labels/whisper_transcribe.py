import os
import json
import whisper
import librosa
import numpy as np
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Process a manifest file with Whisper model and save output.")
parser.add_argument("--manifest_file", type=str, required=True, help="Path to the manifest file.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output .npy files.")
args = parser.parse_args()

manifest_file = args.manifest_file
output_dir = args.output_dir

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

MODEL = "large-v2"
# Load Whisper model once
model = whisper.load_model(MODEL, device="cuda")
with open(manifest_file, 'r') as f:
    for line in f:
        # Parse each JSON line
        entry = json.loads(line.strip())
        audio_path = entry["audio_filepath"]  
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        npy_path = os.path.join(output_dir, base_name + ".npy")

        # uncomment if you want to skip over existing files
        if os.path.exists(npy_path):
            print(f"Skipping {audio_path}, output already exists.")
            continue
        
        print("Processing:", audio_path)

        # Transcribe with word-level timestamps
        transcript = model.transcribe(audio_path, word_timestamps=True)
        
        # Load audio at 16 kHz
        sr = 16000
        audio, _ = librosa.load(audio_path, sr=sr)
        num_samples = len(audio)
        
        # Initialize a binary labels array
        labels = np.zeros(num_samples, dtype=np.int8)
        
        # Mark speech frames based on Whisper's word timestamps
        for segment in transcript.get("segments", []):
            for word_info in segment.get("words", []):
                start_sec = word_info["start"]
                end_sec   = word_info["end"]
                
                start_sample = int(start_sec * sr)
                end_sample   = int(end_sec * sr)
                
                # Clip boundaries if needed
                start_sample = max(start_sample, 0)
                end_sample = min(end_sample, num_samples)
                
                labels[start_sample:end_sample] = 1
        
        # Save the label array
        np.save(npy_path, labels)
        print(f"Saved speech labels to {npy_path}")
