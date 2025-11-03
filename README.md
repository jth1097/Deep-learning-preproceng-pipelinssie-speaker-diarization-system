# 🎙️ Speaker Diarization Enhancement Using Denoising DL-Model
**딥러닝 기반 잡음 제거 모델을 활용한 화자 분리 정확도 향상 시스템**

## 🧩 프로젝트 한 줄 요약
**딥러닝 기반 잡음 제거(denoising) + 듀얼 소스 VAD 하이브리드 모델을 통해 실제 교실/회의 환경에서도 DER을 낮춘 고정확도 화자 분리 시스템**

## 🔗 기존 프로젝트 및 참고 링크
- **기존 프로젝트 GitHub:** [nemo-multistage-classroom-diarization](https://github.com/EduNLP/nemo-multistage-classroom-diarization.git)
- **Deep Learning Model GitHub:** [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- **참고 논문:** [EDM 2025 - Multistage Classroom Diarization](https://educationaldatamining.org/edm2025/proceedings/2025.EDM.short-papers.199/)  

## 📘 Overview  

### 🎯 문제점  
기존의 화자 분리 시스템은 **시끄럽고 다양한 소음이 존재하는 교실·회의 환경**에서 성능이 급격히 저하됨.  

### 💡 해결방안
본 프로젝트는 다음을 결합한 **다단계 하이브리드 파이프라인**을 제안함.  
1. **딥러닝 기반 Speech Enhancement (DeepFilterNet)**
   - 잡음 억제 + 화자 음색 보존  
2. **듀얼 소스 VAD (wav2vec2 + Whisper)**  
   - 소음 환경에서도 명확한 발화 구간 검출  
3. **NeMo 기반 Speaker Embedding + Clustering + Labeling**  
   - 깨끗한 오디오를 기반으로 화자 임베딩 품질 향상  
   - DER(Diarization Error Rate) 감소  

## 🧠 System Pipeline 

```
Noisy Audio
     ↓
[Phase 1] Deep Learning Speech Enhancement (DeepFilterNet V3)
     ↓
[Phase 2] Dual-Source VAD (wav2vec2 + Whisper)
     ↓
[Phase 3] VAD Fusion & Segmentation
     ↓
[Phase 4] Speaker Embedding & Clustering (NeMo)
     ↓
[Phase 5] Speaker Labeling
     ↓
Enhanced & Tagged Audio Output
```

---

## 💻 Demo  

### 🎧 Input Example
```
classbank_audio_data/audio/2.wav
```

### ⚙️ Output Example
```
diarization_output/pred_rttms/2_denoised_diarized.rttm
vad_outs.json
```

| File | Description |
|------|--------------|
| `.wav` | 입력 오디오 파일 |
| `.json` | VAD 결과 (음성 구간 정보) |
| `.rttm` | 화자 분리 결과 (who spoke when) |

## 📈 Result & Performance  

### 🧮 평가 지표 (DER)
```
DER = (FA + MISS + CER) / Duration
```
| Metric | 의미 |
|---------|------|
| FA (False Alarm) | 발화 없음 → 있음으로 오탐 |
| MISS | 실제 발화 → 미탐지 |
| CER (Confusion Error Rate) | 발화는 탐지했으나 화자 할당 오류 |

### 📊 결과
- 기존 파이프라인 대비 DER 감소
- Whisper + wav2vec2 병합 시 안정적 발화 검출 향상
- 잡음 환경 강건성 향상  

## ⚙️ Installation  
- OS: linux ubuntu22.04.5 LTS
- GPU Recommand

```bash
# 1. Clone repository
git clone https://github.com/jth1097/Deep-learning-preproceng-pipelinssie-speaker-diarization-system.git
cd Deep-learning-preproceng-pipelinssie-speaker-diarization-system
rm -rf NeMo
git clone https://github.com/NVIDIA/NeMo.git

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate     # (Windows: .venv\Scripts\activate)

# 3. Install dependencies
pip install nunpy
pip install typting-extension
pip install -r requirements.txt

# 4. Fail
./.venv/src/kenlm/python/BuildStandalone.cmake # cmake_minimum_required(VERSION 3.1) => 3.5
./.venv/src/kenlm/CMakeLists.txt # cmake_minimum_required(VERSION 3.5) => 3.5
pip install -r requirements.txt
```

## ▶️ `run_dl.sh` 실행 환경 & Requirements

### 지원 플랫폼
- Linux: Ubuntu 22.04 LTS 기준으로 개발 및 테스트됨.
- Windows: WSL2(Ubuntu 22.04) 권장. 순수 Windows PowerShell에서도 실행 가능하지만 CUDA/FFmpeg 설치가 복잡하므로 Git Bash 또는 WSL 사용을 추천.

### 필수 시스템 구성 요소
- Git + Git LFS (`git lfs install`) : LFS 오디오/체크포인트를 내려받기 위해 필요.
- 오디오 툴체인: `ffmpeg`, `sox`, `libsndfile1` (Ubuntu `sudo apt install ffmpeg sox libsndfile1`).
- 빌드 도구: `cmake (>=3.18)`, `build-essential`, `python3-dev` (Ubuntu `sudo apt install build-essential cmake python3-dev`).
- Windows 네이티브라면 Chocolatey 등으로 FFmpeg, Git LFS 설치 후 PowerShell을 관리권한으로 실행하세요.

### Python 환경
- Python 3.10 ~ 3.11 권장 (PyTorch 2.6, NeMo 2.2와 호환).
- 가상환경 생성 후 의존성 설치:
  ```bash
  python -m venv .venv
  source .venv/bin/activate              # Windows: .venv\Scripts\activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
- DeepFilterNet3 기반 디노이징을 활성화하려면 아래 중 하나를 추가 설치:
  ```bash
  pip install deepfilternet              # 공식 DeepFilterNet 패키지
  # 또는 (fallback)
  pip install df
  ```
  두 패키지 중 하나라도 설치되어 있으면 `run_dl.sh` 실행 시 자동으로 디노이징이 적용됩니다.

### 추가 리소스 준비
- `NeMo` 리포지토리를 반드시 `./NeMo` 하위에 클론하고, `pip install -r requirements.txt` 내부에서 설치되는 `nemo-toolkit`과 동일한 버전을 유지하세요.
- VAD 모델 체크포인트: `generate_w2v2_speech_labels/run_vad.py`는 `checkpoints/w2v2.ckpt`를 요구합니다. 팀에서 학습한 모델을 `project_root/checkpoints/w2v2.ckpt` 경로로 복사하거나, 새 모델을 학습해 동일한 이름으로 저장하세요.
- Whisper/Transformers 모델은 최초 실행 시 Hugging Face/OpenAI에서 자동으로 다운로드됩니다. 방화벽 환경이라면 사전 다운로드 후 `HF_HOME`, `WHISPER_CACHE_DIR` 등을 설정해 오프라인 캐시를 사용하세요.
- CUDA 실행을 원한다면 NVIDIA Driver + CUDA 12.4 호환 버전을 설치하고, `pip install -r requirements.txt`가 제공하는 `torch==2.6.0`, `torchaudio==2.6.0`이 GPU를 인식하는지 `python -c "import torch; print(torch.cuda.is_available())"`로 확인하세요.

### `run_dl.sh` 실행 순서
```bash
# 1. (선택) 실행 권한 부여
chmod +x run_dl.sh

# 2. 오디오 파일 단일 평가
./run_dl.sh path/to/audio.wav       # Windows WSL/Git Bash
# 또는 PowerShell
bash run_dl.sh path/to/audio.wav
# 또는 Python 인터페이스
python scripts/run_diar_experiment.py --audio-file path/to/audio.wav --denoise auto
```
- 스크립트는 입력 WAV를 16kHz 모노로 내부 변환 후 DeepFilterNet3 디노이즈 → Wav2Vec2 기반 VAD → Whisper ASR 라벨 → NeMo diarization 순으로 수행합니다.
- 실행 로그는 `logs/neMo_run_<파일명>_dl.log`, DER 결과는 `reports/der_metrics.csv`, RTTM 출력은 `diarization_output/pred_rttms`에 저장됩니다.
- 첫 실행 시 모델 다운로드로 인해 시간이 오래 걸릴 수 있으며, GPU 없는 환경에서는 처리 시간이 크게 증가합니다.

## 🚀 Usage  

```bash
# 전체 파이프라인 실행
chnod +x run.sh
./run.sh
```

- `manifests/test.json` : 오디오 경로 및 메타데이터 목록  
- `vad_outs.json` : VAD 결과 중간 산출물  
- `diarization_output` : 최종 화자 분리 결과 저장 폴더  

## ⚠️ Common Issues & Solutions  

| 문제 | 원인 | 해결 방법 |
|------|------|------------|
| `PySoundFile failed` | libsndfile 미설치 | `sudo apt install libsndfile1` |
| CUDA 오류 (`device not found`) | GPU 환경 미설정 | CUDA 11.8 + cuDNN 8.6 버전 확인 |
| VAD 결과 없음 | 입력 파일 형식 불일치 | 16kHz mono PCM 형식으로 변환 |
| 0 bytes output | ffmpeg 변환 실패 | `ffmpeg -i input.wav -ar 16000 -ac 1 output.wav` 로 재생성 |
| 메모리 부족 | 딥러닝 모델 메모리 초과 | `--batch_size` 감소 또는 GPU 메모리 증가 필요 |


## 🧩 Future Work  
- 딥러닝 전처리 모델 **Fine-tuning** (잡음 포함 vs 제거 데이터 병합 학습)  
- 라벨링 + Audio-to-Text 연동으로 **시각적 화자 구분 자료 생성**  
- 실시간 스트리밍 환경 적용 (on-device inference 최적화)


## 👥 Team “Alone”
| 역할 | 이름 |
|------|------|
| Researcher |  신홍규 |
| Researcher |  남경식 |
| Researcher |  양평화 |
| Researcher |  장태환 |


## 🧾 License  
This project is for **academic research** purposes under the **Konkuk University Capstone Design (졸업프로젝트)** program.  
For any citation or reuse, please credit:  
> *ALONE et al., “Speaker diarization enhancement using denoising DL-model”, Konkuk Univ., 2025.*
