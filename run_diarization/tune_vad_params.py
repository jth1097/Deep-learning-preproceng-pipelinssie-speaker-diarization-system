import os
import numpy as np
import librosa
from dataclasses import dataclass
from typing import List
import json
import argparse
from tqdm import tqdm

@dataclass
class SpeechSegment:
    start: float
    end: float

def read_frame_probabilities(frame_file: str) -> np.ndarray:
    """
    Read a .frame file where each line represents the speech probability of a single frame.
    Returns a 1D numpy array of speech probabilities.
    """
    with open(frame_file, 'r') as f:
        probs = [float(line.strip()) for line in f]
    return np.array(probs, dtype=np.float32)


def read_asr_probabilities(asr_npy_file: str) -> np.ndarray:
    """
    Load ASR probabilities from a .npy file that has sample-level probabilities.
    Returns a 1D numpy array of probabilities (length = number of audio samples).
    """
    return np.load(asr_npy_file)


def combine_vad_asr_framewise(
    vad_probs: np.ndarray,
    asr_probs: np.ndarray,
    alpha: float,
    audio_length_s: float,
    sr: int = 16000
) -> np.ndarray:
    """
    Given frame-level VAD probabilities (vad_probs) and sample-level ASR probabilities (asr_probs),
    combine them into a single array of frame-level probabilities using an alpha weight.

    The `vad_probs` array length = total_frames
    The `asr_probs` array length = total_samples

    We need to average (or otherwise combine) asr_probs over each frame.

    alpha: weighting factor for combination
        e.g., combined_prob[frame] = alpha * vad_prob[frame] + (1 - alpha) * asr_mean_in_that_frame
    """
    total_frames = len(vad_probs)
    frame_duration = audio_length_s / total_frames
    samples_per_frame = int(frame_duration * sr)

    combined = np.zeros(total_frames, dtype=np.float32)
    num_asr_samples = len(asr_probs)

    for i in range(total_frames):
        start_sample = i * samples_per_frame
        end_sample = start_sample + samples_per_frame
        if end_sample > num_asr_samples:
            end_sample = num_asr_samples
        if start_sample >= end_sample:
            # If we are beyond the audio length, just take the VAD prob
            asr_mean = 0.0
        else:
            asr_mean = np.mean(asr_probs[start_sample:end_sample])

        # Combine
        combined[i] = alpha * vad_probs[i] + (1.0 - alpha) * asr_mean

    print(combined)
    print(vad_probs)
    return combined


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


def generate_vad_segments_from_probs(
    speech_probs: np.ndarray,
    audio_length_s: float,
    onset: float = 0.5,
    offset: float = 0.3,
    min_duration_on: float = 0.0,
    min_duration_off: float = 0.15,
    pad_onset: float = 0.0,
    pad_offset: float = 0.0,
    merge_silence: float = 0.5,
) -> List[SpeechSegment]:
    """
    Convert an array of frame-level speech probabilities into a list of SpeechSegment
    objects based on the onset/offset thresholds. Also merges any consecutive
    segments that are within 'merge_silence' seconds of each other.
    """
    total_frames = len(speech_probs)
    frame_duration = audio_length_s / total_frames

    # Binary decisions: initially mark frames with prob >= onset as speech
    predictions = (speech_probs >= onset)

    segments = []
    in_speech = False
    start_frame_idx = 0

    for i, prob in enumerate(speech_probs):
        if (prob >= onset) and (not in_speech):
            # Transition to speech
            start_frame_idx = i
            in_speech = True
        elif in_speech and (prob < offset):
            # Transition out of speech
            end_frame_idx = i
            duration = (end_frame_idx - start_frame_idx) * frame_duration
            if duration >= min_duration_off:
                seg_start = max(0, start_frame_idx * frame_duration - pad_onset)
                seg_end = min(audio_length_s, end_frame_idx * frame_duration + pad_offset)
                segments.append(SpeechSegment(start=seg_start, end=seg_end))
            in_speech = False

    # If still in speech at end of file
    if in_speech:
        end_frame_idx = total_frames
        duration = (end_frame_idx - start_frame_idx) * frame_duration
        if duration >= min_duration_off:
            seg_start = max(0, start_frame_idx * frame_duration - pad_onset)
            seg_end = min(audio_length_s, end_frame_idx * frame_duration + pad_offset)
            segments.append(SpeechSegment(start=seg_start, end=seg_end))

    # Optionally merge segments if the silence gap is short
    if merge_silence > 0:
        segments = merge_segments(segments, min_silence_duration=merge_silence)

    # Filter out segments that are too short
    final_segments = []
    for seg in segments:
        if (seg.end - seg.start) >= min_duration_on:  # use min_duration_on if needed
            final_segments.append(seg)

    return final_segments


def write_vad_manifest(speech_segments: List[SpeechSegment], audio_path: str, manifest_path: str):
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    
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



def analyze_rttm_vs_predictions(
    rttm_path: str,
    speech_prob_accumulator: np.ndarray,
    onset: float,
    offset: float,
    total_audio_length: float,
    sr: int = 16000
):
    """
    Compare final predictions with RTTM ground-truth at frame-level,
    then compute average probability in FN/FP frames.
    """
    from typing import Tuple

    def _read_rttm(rttm_file: str, audio_duration: float, sr: int) -> np.ndarray:
        num_samples = int(audio_duration * sr)
        labels = np.zeros(num_samples, dtype=np.int8)
        with open(rttm_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # "SPEAKER <file> 1 start duration ..."
                start_time = float(parts[3])
                duration = float(parts[4])
                start_sample = int(round(start_time * sr))
                end_sample = int(round((start_time + duration) * sr))
                end_sample = min(end_sample, num_samples)
                labels[start_sample:end_sample] = 1
        return labels

    def samples_to_frames(sample_labels: np.ndarray, total_frames: int, frame_duration: float, sr: int) -> np.ndarray:
        frame_labels = np.zeros(total_frames, dtype=np.int8)
        samples_per_frame = int(round(frame_duration * sr))
        for frame_idx in range(total_frames):
            start_samp = frame_idx * samples_per_frame
            end_samp = start_samp + samples_per_frame
            if end_samp > len(sample_labels):
                end_samp = len(sample_labels)
            if start_samp >= end_samp:
                break
            frame_mean = np.mean(sample_labels[start_samp:end_samp])
            frame_labels[frame_idx] = 1 if frame_mean >= 0.5 else 0
        return frame_labels

    total_frames = len(speech_prob_accumulator)
    frame_duration = total_audio_length / total_frames

    # 1) read RTTM -> sample-level label
    sample_labels = _read_rttm(rttm_path, total_audio_length, sr)
    # 2) convert to frame-level
    gt_frames = samples_to_frames(sample_labels, total_frames, frame_duration, sr)

    # 3) VAD predictions from speech_prob_accumulator
    # Mark frame as speech if prob >= onset, once in speech keep going until prob < offset
    predictions = np.zeros(total_frames, dtype=int)
    in_speech = False
    for i, prob in enumerate(speech_prob_accumulator):
        if prob >= onset and not in_speech:
            in_speech = True
            predictions[i] = 1
        elif in_speech:
            if prob < offset:
                in_speech = False
                predictions[i] = 0
            else:
                predictions[i] = 1

    # 4) compare
    false_negatives = (predictions == 0) & (gt_frames == 1)
    false_positives = (predictions == 1) & (gt_frames == 0)

    avg_prob_fn = (
        np.mean(speech_prob_accumulator[false_negatives]) if np.any(false_negatives) else 0.0
    )
    avg_prob_fp = (
        np.mean(speech_prob_accumulator[false_positives]) if np.any(false_positives) else 0.0
    )

    print(f"  MISS (gt=1, pred=0): avg prob={avg_prob_fn:.4f}, count={np.sum(false_negatives)}")
    print(f"  FA   (gt=0, pred=1): avg prob={avg_prob_fp:.4f}, count={np.sum(false_positives)}\n")


def write_vad_manifest(speech_segments: List[SpeechSegment], audio_path: str, manifest_path: str):
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    
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

            
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VAD from precomputed .frame files + optional .npy ASR files.")
    parser.add_argument(
        "--manifest_file",
        type=str,
        required=True,
        help="Path to input manifest file (JSON lines). Each line must contain 'audio_filepath'."
    )
    parser.add_argument(
        "--frame_dir",
        type=str,
        required=True,
        help="Directory containing .frame files (each named <basename>.frame)."
    )
    parser.add_argument(
        "--asr_dir",
        type=str,
        default=None,
        help="Directory containing .npy files for ASR-based probabilities (each named <basename>.npy). If not provided, ASR will be skipped."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Weight factor for VAD vs ASR combination: combined = alpha * VAD + (1-alpha) * ASR."
    )
    parser.add_argument(
        "--onset",
        type=float,
        default=0.55,
        help="Onset threshold for VAD."
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.10,
        help="Offset threshold for VAD."
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="vad_segments_out",
        help="Directory to store the resulting segment files."
    )

    args = parser.parse_args()

    # Make sure manifest exists
    if not os.path.isfile(args.manifest_file):
        print(f"ERROR: Manifest file not found: {args.manifest_file}")
        exit(1)

    # Prepare output directory

    # Read the manifest lines
    with open(args.manifest_file, 'r') as f:
        manifest_lines = f.readlines()

    print(f"Loaded {len(manifest_lines)} lines from {args.manifest_file}")
    print("Starting VAD segmentation...")

    print(f"USING: onset:{args.onset}, {type(args.onset)} offset={args.offset} {type(args.offset)}")
    for line in tqdm(manifest_lines, desc="Processing Audio"):
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line:\n{line}")
            continue

        audio_path = data.get("audio_filepath", None)
        if not audio_path:
            print("Skipping entry with no 'audio_filepath':", data)
            continue

        # 1) Extract base name for .frame and .npy file
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        frame_file = os.path.join(args.frame_dir, base_name + ".frame")

        # Check existence of .frame
        if not os.path.isfile(frame_file):
            print(f"WARNING: .frame file not found for {audio_path} -> {frame_file}. Skipping.")
            continue

        # 2) Read audio to get total duration
        if not os.path.isfile(audio_path):
            print(f"WARNING: Audio file not found: {audio_path}. Skipping.")
            continue

        audio, sr = librosa.load(audio_path, sr=16000)
        audio_length_s = len(audio) / sr

        # 3) Load VAD frame probs
        vad_probs = read_frame_probabilities(frame_file)

        # 4) Optionally combine with ASR
        combined_probs = vad_probs
        if args.asr_dir is not None:
            asr_file = os.path.join(args.asr_dir, base_name + ".npy")
            if os.path.isfile(asr_file):
                asr_probs = read_asr_probabilities(asr_file)
                combined_probs = combine_vad_asr_framewise(
                    vad_probs,
                    asr_probs,
                    alpha=args.alpha,
                    audio_length_s=audio_length_s,
                    sr=sr
                )
                print("COMB:", combined_probs)
            else:
                print(f"ASR file not found for {audio_path}: {asr_file} (continuing with only VAD).")

        # 5) Generate segments
        segments = generate_vad_segments_from_probs(
            speech_probs=combined_probs,
            audio_length_s=audio_length_s,
            onset=args.onset,
            offset=args.offset,
        )

        # 6) Save segments (RTTM or .txt)

        write_vad_manifest(segments, audio_path, args.out_dir)

    print("VAD segmentation complete.")