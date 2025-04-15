import torch
import librosa
from transformers import Wav2Vec2FeatureExtractor
import numpy as np
from model import * 
from dataclasses import dataclass
from typing import List
from transformers import Wav2Vec2Config
import os
import torchaudio
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import torch.nn.functional as F
from tqdm import tqdm
from speechbrain.inference import EncoderClassifier  # Updated import
import json
from scipy.ndimage import median_filter
from typing import List, Dict, Any
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import os

def save_frame_probabilities(probs, output_path):
    print(f"Saving frames in {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for prob in probs:
            f.write(f"{prob:.4f}\n")

def save_segments(segments, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for seg in segments:
            f.write(f"{seg.start:.4f} {seg.end-seg.start:.4f} speech\n")


@dataclass
class SpeechSegment:
    start: float
    end: float
# vad_inference.py
from collections import OrderedDict


def load_vad_model(
    checkpoint_path: str,
    local_rank: str = "cuda" if torch.cuda.is_available() else "cpu",
    layer: int = 14
):
    """
    Load the VAD model from a checkpoint using setup_model for inference.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        fairseq_path (str): Path to the Fairseq pre-trained model.
        local_rank (str): Device to load the model ('cuda' or 'cpu').
        layer (int): Layer of the Fairseq model to use for representation.

    Returns:
        model (torch.nn.Module): The loaded VAD model ready for inference.
    """
    # Initialize the model with weights_only=True internally (ensure Fairseq version supports it)
    model = Wav2VecWithClassifier()

    # Load the VAD-specific state dictionary from the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=local_rank, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Function to remove 'module.' prefix
    def remove_module_prefix(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_key = k[7:]  # Remove 'module.' prefix
            else:
                new_key = k
            new_state_dict[new_key] = v
        return new_state_dict

    # Remove 'module.' prefix if present
    state_dict = remove_module_prefix(state_dict)

    # Load the modified state dictionary into the model
    try:
        model.load_state_dict(state_dict, strict=True)
        # print("State dictionary loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        raise e

    # Move model to device and set it to evaluation mode
    model.to(local_rank)
    model.eval()
    print("VAD Model loaded and ready for inference.")

    return model


def setup_model(local_rank, fairseq_path, layer=14):
    model = Wav2VecWithClassifier(
        checkpoint_path=fairseq_path,
        layer=layer,
        hidden_dim=1024,
        num_labels=2
    )

    model.to(local_rank)
    return model

def merge_segments(segments: List[SpeechSegment], min_silence_duration: float) -> List[SpeechSegment]:
    """Merge segments that are separated by silence shorter than min_silence_duration."""
    if not segments:
        return segments
    
    merged = []
    current_segment = segments[0]
    
    for next_segment in segments[1:]:
        gap_duration = next_segment.start - current_segment.end
        if gap_duration <= min_silence_duration:
            # Merge segments
            current_segment = SpeechSegment(
                start=current_segment.start,
                end=next_segment.end
            )
        else:
            merged.append(current_segment)
            current_segment = next_segment
    merged.append(current_segment)
    return merged

def get_speech_segments(
    audio_path: str,
    model,
    feature_extractor,
    frames_output_path,
    window_length_in_sec: float = 2,    # VAD context window duration in sec
    shift_length_in_sec: float = 0.25,       # Step size for advancing along the audio
    smoothing: str = None,              # Smoothing method ("median" or "mean"); False for no smoothing
    overlap: float = 0.5,                   # Overlap ratio for smoothing filter window size (between 0 and 1)
    onset: float = 0.55,                     # Onset threshold (speech probability threshold to mark speech start)
    offset: float = 0.45,                    # Offset threshold (speech probability threshold to mark speech end)
    pad_onset: float = 0,                 # Padding to add before each speech segment
    pad_offset: float = 0,                  # Padding to add after each speech segment
    min_duration_on: float = 0,             # Minimum duration of a valid speech segment before deletion (unused here)
    min_duration_off: float = 0.15,          # Minimum speech segment duration (segments shorter than this will be dropped)
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_probs=True
    ) -> List[SpeechSegment]:
    """Process audio file and return list of speech segments using the tuned parameters."""
    # Load and resample audio
    # print("Testing new model trained using both w2v and linear model")
    audio, sr = librosa.load(audio_path, sr=16000)
    # For demonstration limit to first 10 seconds; remove if you want to process full file.
    # audio = audio[: 5 * sr]
    audio_duration = len(audio) / sr

    # Calculate window and shift sizes in samples
    window_samples = int(window_length_in_sec * sr)
    shift_samples = int(shift_length_in_sec * sr)
    
    # Determine frames per window from model output
    with torch.no_grad():
        # Convert numpy array to tensor first
        sample_audio = torch.randn(1, 32000).to(device)
        sample_output = model(sample_audio)
        frames_per_window = sample_output["logits"].shape[1]
    
    # Duration of each frame (in seconds)
    frame_duration = window_length_in_sec / frames_per_window
    # print(f"Frame Duration: {frame_duration}")
    total_frames = int(np.ceil(audio_duration / frame_duration))
    
    # Accumulate frame-level speech probabilities and count overlaps
    speech_prob_accumulator = np.zeros(total_frames)
    overlap_counter = np.zeros(total_frames)
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, len(audio), shift_samples), desc="VAD Processing"):
            end_idx = min(start_idx + window_samples, len(audio))
            if end_idx - start_idx < window_samples:
                audio_segment = np.pad(
                    audio[start_idx:end_idx],
                    (0, window_samples - (end_idx - start_idx))
                )
            else:
                audio_segment = audio[start_idx:end_idx]
            
            inputs = torch.from_numpy(audio_segment).float().reshape(1, 32000).to(device)
            outputs = model(inputs)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            # Get probabilities for the speech class (assumed to be index 1)
            speech_probs = probs[:, :, 1].cpu().numpy()[0]
            # print(speech_probs)
            # exit(0)
            start_frame = int(start_idx / (sr * frame_duration))
            frames_to_add = min(len(speech_probs), total_frames - start_frame)
            speech_prob_accumulator[start_frame:start_frame + frames_to_add] += speech_probs[:frames_to_add]
            overlap_counter[start_frame:start_frame + frames_to_add] += 1
    
    # Compute the average speech probability per frame
    speech_prob_accumulator = np.divide(
        speech_prob_accumulator,
        overlap_counter,
        out=np.zeros_like(speech_prob_accumulator),
        where=overlap_counter != 0
    )
    
    # print(speech_prob_accumulator)
    # print(len(speech_prob_accumulator), total_frames)

    if save_probs:
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        frame_path = os.path.join(frames_output_path, f"{audio_name}.frame")
        save_frame_probabilities(speech_prob_accumulator, frame_path)

    # Use separate thresholds for  and offset to determine speech regions.
    predictions = np.zeros_like(speech_prob_accumulator, dtype=bool)
    predictions[speech_prob_accumulator >= onset] = True  # Mark as speech if above onset threshold
    
    # print(predictions)
    # Refine boundaries: once in speech, wait until falling below offset threshold to mark end
    segments = []
    in_speech = False
    start_frame_idx = 0
    
    for i, prob in enumerate(speech_prob_accumulator):
        if prob >= onset and not in_speech:
            start_frame_idx = i
            in_speech = True
        elif in_speech and prob < offset:
            end_frame_idx = i
            duration = (end_frame_idx - start_frame_idx) * frame_duration
            if duration >= min_duration_off:
                # Apply padding while clamping the boundaries to the audio duration
                seg_start = max(0, start_frame_idx * frame_duration - pad_onset)
                seg_end = min(audio_duration, end_frame_idx * frame_duration + pad_offset)
                segments.append(SpeechSegment(start=seg_start, end=seg_end))
            in_speech = False
    if in_speech:  # Handle case if audio ends while still in speech region
        end_frame_idx = len(speech_prob_accumulator)
        duration = (end_frame_idx - start_frame_idx) * frame_duration
        if duration >= min_duration_off:
            seg_start = max(0, start_frame_idx * frame_duration - pad_onset)
            seg_end = min(audio_duration, end_frame_idx * frame_duration + pad_offset)
            segments.append(SpeechSegment(start=seg_start, end=seg_end))
    
    # Optional: Merge segments if gaps between them are very short.
    segments = merge_segments(segments, min_silence_duration=pad_offset if pad_offset > 0 else 0.5)
    
    # Finally, filter out any segments that are too short.
    segments = [seg for seg in segments if (seg.end - seg.start) >= min_duration_off]
    # print(segments)
    return segments

def write_vad_manifest(speech_segments: List[SpeechSegment], audio_path: str, manifest_path: str):
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    # print("Writing at ", manifest_path)
    with open(manifest_path, 'a') as fp:
        for seg in speech_segments:
            offset = float(seg.start)
            duration = float(seg.end) - offset
            vad_data = {
                'audio_filepath': audio_path,
                'offset': round(offset, 2),
                'duration': round(duration, 2),
                'label': 'UNK',
                'uniq_id': audio_name
            }
            json.dump(vad_data, fp)
            fp.write('\n')

def vad(
    audio_path: str,
    model,
    frames_output_path: str = "./frames", 
    vad_manifest_path: str = './umd_nemo_vad_outputs.json',
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):

    """Run VAD with the provided parameters."""
    # feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    
    # Get speech segments with new parameters.
    speech_segments = get_speech_segments(
        audio_path=audio_path,
        model=model,
        frames_output_path=frames_output_path,
        feature_extractor=None,
        device=device
    )

    # print("Detected segments:")
    # for seg in speech_segments:
    #     print(f"Start: {seg.start:.2f} sec, End: {seg.end:.2f} sec")
        
    
    # write_vad_manifest(speech_segments, audio_path, vad_manifest_path)
