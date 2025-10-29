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
