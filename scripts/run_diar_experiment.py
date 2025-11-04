#!/usr/bin/env python
import argparse
import subprocess
import shutil
import re
import json
from pathlib import Path

import soundfile as sf
import librosa


def run(cmd, cwd=None, log_path=None):
    if log_path is not None:
        with open(log_path, 'w', encoding='utf-8') as log_file:
            result = subprocess.run(cmd, cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    else:
        result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def append_der_metrics(project_root: Path, experiment: str, log_path: Path, alpha: float, onset: float, offset: float):
    text = log_path.read_text(encoding='utf-8', errors='ignore')
    pattern = r"\| FA: ([^|]+) \| MISS: ([^|]+) \| CER: ([^|]+) \| DER: ([^|]+) \| Spk\. Count Acc\. ([^\r\n]+)"
    match = re.findall(pattern, text)
    if not match:
        raise RuntimeError('No DER metrics found in NeMo log.')
    fa, miss, cer, der, sca = match[-1]
    metrics_csv = project_root / 'reports/der_metrics.csv'
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    if not metrics_csv.exists():
        metrics_csv.write_text('source,alpha,onset,offset,FA,MISS,CER,DER,SpkCountAcc\n', encoding='utf-8')
    with metrics_csv.open('a', encoding='utf-8') as fp:
        fp.write(
            f"nemo_gpu_run_{experiment},{alpha:.2f},{onset:.2f},{offset:.2f},{fa},{miss},{cer},{der},{sca}\n"
        )


def append_label_summary(project_root: Path, experiment: str, audio_path: Path, gt_path: Path, pred_path: Path):
    labels_csv = project_root / 'reports/label_summary.csv'
    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    if not labels_csv.exists():
        labels_csv.write_text(
            'file_name,audio_path,audio_exists,duration_sec,gt_rttm_path,gt_exists,gt_segments,gt_unique_speakers,gt_total_speech_sec,pred_rttm_path,pred_exists,pred_segments,pred_unique_speakers,pred_total_speech_sec\n',
            encoding='utf-8',
        )
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    duration = len(data) / sr

    def parse_rttm(path: Path):
        if not path.exists():
            return False, 0, 0, 0.0
        segments = 0
        speakers = set()
        total = 0.0
        with path.open('r', encoding='utf-8') as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                segments += 1
                total += float(parts[4])
                speakers.add(parts[7])
        return True, segments, len(speakers), total

    gt_exists, gt_segments, gt_unique, gt_total = parse_rttm(gt_path)
    pred_exists, pred_segments, pred_unique, pred_total = parse_rttm(pred_path)

    def rel(path: Path):
        return str(path.relative_to(project_root)).replace('\\', '/')

    row = [
        experiment,
        rel(audio_path),
        'True',
        f"{duration:.5f}",
        rel(gt_path) if gt_exists else '',
        str(gt_exists),
        str(gt_segments if gt_exists else 0),
        str(gt_unique if gt_exists else 0),
        f"{gt_total:.2f}" if gt_exists else '0.0',
        rel(pred_path) if pred_exists else '',
        str(pred_exists),
        str(pred_segments if pred_exists else 0),
        str(pred_unique if pred_exists else 0),
        f"{pred_total:.2f}" if pred_exists else '0.0',
    ]
    with labels_csv.open('a', encoding='utf-8') as fp:
        fp.write(','.join(row) + '\n')


def count_speakers_from_rttm(rttm: Path) -> int:
    try:
        speakers = set()
        with rttm.open('r', encoding='utf-8', errors='ignore') as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) >= 8:
                    speakers.add(parts[7])
        return len(speakers)
    except Exception:
        return 0


def write_single_line_manifest(path: Path, audio_fp: Path, rttm_fp: Path | None, num_speakers: int | None):
    entry = {
        "audio_filepath": str(audio_fp).replace('\\', '/'),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": int(num_speakers) if num_speakers is not None else None,
        "rttm_filepath": str(rttm_fp).replace('\\', '/') if rttm_fp else None,
        "uem_filepath": None,
        "ctm_filepath": None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding='utf-8')


def resolve_rttm(project_root: Path, audio_file: Path, explicit_rttm: str | None) -> Path | None:
    if explicit_rttm:
        rp = Path(explicit_rttm)
        if not rp.is_absolute():
            rp = (project_root / rp).resolve()
        return rp if rp.exists() else None
    stem = audio_file.stem
    cand1 = audio_file.with_suffix('.rttm')
    if cand1.exists():
        return cand1
    cand2 = project_root / 'classbank_audio_data' / 'rttm' / f'{stem}.rttm'
    if cand2.exists():
        return cand2
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-file', required=True)
    parser.add_argument('--experiment', default=None)
    parser.add_argument('--keep-new-audio', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--onset', type=float, default=0.55)
    parser.add_argument('--offset', type=float, default=0.10)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--denoise', default='auto', choices=['auto', 'dfnet3', 'none'])
    parser.add_argument('--rttm-file', default=None, help='Path to reference RTTM (defaults to stem match)')
    parser.add_argument('--whisper-model', default='base')
    parser.add_argument('--msdd-model', default=None, help='NeMo MSDD model name/path to enable overlap-aware decoding')
    parser.add_argument('--spk-embedder', default='titanet_large', help='Speaker embedding model (titanet_large, ecapa_tdnn, speakerverification_speakernet)')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    experiment = args.experiment or Path(args.audio_file).stem

    # Resolve input
    input_audio = Path(args.audio_file)
    if not input_audio.is_absolute():
        input_audio = (project_root / input_audio).resolve()
    if not input_audio.exists():
        raise FileNotFoundError(f'Audio file not found: {input_audio}')

    y, sr = librosa.load(str(input_audio), sr=16000, mono=True)
    # Optional denoising (DeepFilterNet3) before downstream steps
    if args.denoise != 'none':
        try:
            from denoise_dfnet3 import denoise_dfnet3
            y_denoised, info = denoise_dfnet3(y, sr, enable=(args.denoise in ('auto', 'dfnet3')))
            if y_denoised is not None:
                y = y_denoised
                print(f"Applied DeepFilterNet3 denoise: {info}")
            else:
                print(f"Denoise skipped: {info}")
        except Exception as e:
            print(f"Denoise error; proceeding without: {e}")
    # Write temp audio copy used by the manifest
    tmp_audio_dir = project_root / 'classbank_audio_data' / 'audio_tmp'
    tmp_audio_dir.mkdir(parents=True, exist_ok=True)
    tmp_audio = tmp_audio_dir / f'{experiment}.wav'
    sf.write(tmp_audio, y, 16000)

    for rel in ['vad_output_frames', 'whisper_output_frames', 'diarization_output']:
        target = project_root / rel
        if target.exists():
            shutil.rmtree(target)
    for rel in ['vad_outs.json', 'vad_outs_abs.json']:
        target = project_root / rel
        if target.exists():
            target.unlink()

    alpha_str = f"{args.alpha:.2f}"
    onset_str = f"{args.onset:.2f}"
    offset_str = f"{args.offset:.2f}"

    # Build per-file manifest (audio + mapped RTTM + num_speakers)
    rttm_path = resolve_rttm(project_root, input_audio, args.rttm_file)
    ns = count_speakers_from_rttm(rttm_path) if rttm_path else None
    tmp_manifest = project_root / f'manifests/tmp_{experiment}.jsonl'
    write_single_line_manifest(tmp_manifest, tmp_audio, rttm_path, ns)

    run(
        [
            'python',
            'generate_w2v2_speech_labels/run_vad.py',
            '--manifest_file',
            str(tmp_manifest),
            '--checkpoint_path',
            'checkpoints/w2v2.ckpt',
            '--vad_manifest_path',
            'null.json',
            '--frames_output_path',
            'vad_output_frames',
        ],
        cwd=project_root,
    )
    run(
        [
            'python',
            'generate_whisper_speech_labels/whisper_transcribe.py',
            '--manifest_file',
            str(tmp_manifest),
            '--output_dir',
            'whisper_output_frames',
            '--model',
            args.whisper_model,
            '--device',
            args.device,
        ],
        cwd=project_root,
    )
    run(
        [
            'python',
            'run_diarization/tune_vad_params.py',
            '--manifest_file',
            str(tmp_manifest),
            '--frame_dir',
            'vad_output_frames',
            '--asr_dir',
            'whisper_output_frames',
            '--alpha',
            alpha_str,
            '--onset',
            onset_str,
            '--offset',
            offset_str,
            '--out_dir',
            'vad_outs_abs.json',
        ],
        cwd=project_root,
    )

    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f'neMo_run_{experiment}.log'
    nemo_cmd = [
        'python',
        'NeMo/offline_diar_infer.py',
        f'diarizer.manifest_filepath={tmp_manifest}',
        'diarizer.out_dir=diarization_output',
        'diarizer.vad.model_path=null',
        'diarizer.vad.external_vad_manifest=vad_outs_abs.json',
        'diarizer.speaker_embeddings.parameters.save_embeddings=False',
        f'diarizer.speaker_embeddings.model_path={args.spk_embedder}',
        'diarizer.clustering.parameters.oracle_num_speakers=True',
        'num_workers=0',
        'hydra.job.chdir=false',
    ]
    if args.msdd_model:
        nemo_cmd.append(f'diarizer.msdd_model.model_path={args.msdd_model}')
    run(nemo_cmd, cwd=project_root, log_path=log_path)

    try:
        append_der_metrics(project_root, experiment, log_path, args.alpha, args.onset, args.offset)
    except Exception as e:
        print(f'DER extraction warning: {e}')

    pred_rttm = project_root / 'diarization_output' / 'pred_rttms' / f'{tmp_audio.stem}.rttm'
    append_label_summary(project_root, experiment, tmp_audio, rttm_path if rttm_path else Path(''), pred_rttm)

    print(
        f"Experiment completed: {experiment} (alpha={alpha_str}, onset={onset_str}, offset={offset_str}, device={args.device})"
    )


if __name__ == '__main__':
    main()
