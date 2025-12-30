# Qwen Image Edit - Offline Editor

[Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) 모델 기반 오프라인 이미지 편집기

## 주요 기능

✅ **완전한 오프라인 동작**: 모델을 한 번 다운로드하면 인터넷 없이 사용 가능
✅ **배치 처리**: 폴더 내 모든 이미지를 한 번에 편집
✅ **간단한 사용법**: 커맨드라인 인터페이스로 쉽게 사용
✅ **유연한 설정**: 프롬프트, 입출력 경로 자유롭게 설정

## 시스템 요구사항

### 하드웨어 요구사항

#### GPU (권장)
- **최소**: NVIDIA GPU 12GB VRAM (GGUF 양자화 사용 시)
- **권장**: NVIDIA GPU 24GB VRAM (RTX 4090, RTX 5080 등)
- **최적**: NVIDIA GPU 40GB+ VRAM (A100, RTX 5090 등)

#### CPU 모드
- CPU에서도 실행 가능하지만 매우 느립니다
- RAM: 16GB 이상 권장

#### 디스크 공간
- **모델 저장**: 약 10-60GB (모델 변형에 따라 다름)
- **작업 공간**: 추가 10GB 이상 권장

### 소프트웨어 요구사항

#### CUDA (GPU 사용 시)
- **권장**: CUDA 11.8 이상 (12.x 지원)
- **최소**: CUDA 11.7
- PyTorch 2.0+는 CUDA 11.7 미만을 지원하지 않습니다

#### Python
- **권장**: Python 3.8 - 3.11
- **필수 라이브러리**:
  - PyTorch 2.0+
  - diffusers >= 0.30.0
  - transformers >= 4.37.0
  - Pillow, tqdm, accelerate

### Hugging Face 토큰

**Qwen-Image-Edit-2511은 공개 모델이므로 Hugging Face 토큰이 필요하지 않습니다.**

단, 다운로드 속도 제한을 피하려면 토큰 사용을 권장합니다:

```bash
# 선택사항: Hugging Face 로그인
pip install huggingface-hub
huggingface-cli login
```

## 설치

### 0. CUDA 설치 (GPU 사용 시)

GPU를 사용하려면 먼저 NVIDIA CUDA Toolkit을 설치하세요:

- **Windows**: [CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)
- **Linux**:
  ```bash
  # Ubuntu/Debian 예시
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update
  sudo apt-get -y install cuda-toolkit-12-1
  ```

**권장 버전**: CUDA 11.8 또는 12.1

### 1. 저장소 클론

```bash
git clone <repository-url>
cd qwen-image
```

### 2. 가상환경 생성 (권장)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

### 3. PyTorch 설치 (CUDA 버전에 맞게)

```bash
# CUDA 12.1 사용 시
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 사용 시
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU만 사용 시
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. 의존성 설치

```bash
# 기본 의존성 설치
pip install -r requirements.txt

# diffusers 최신 버전 설치 (권장)
pip install git+https://github.com/huggingface/diffusers
```

## 사용 방법

### Step 1: 모델 다운로드 (최초 1회만)

```bash
python download_model.py \
  --model_id Qwen/Qwen-Image-Edit-2511 \
  --save_path ./models/qwen-image-edit
```

**주의**:
- 모델 크기가 크므로 다운로드에 시간이 걸릴 수 있습니다
- 안정적인 인터넷 연결이 필요합니다
- 한 번만 다운로드하면 이후 오프라인에서 사용 가능합니다

### Step 2: 이미지 편집

#### 단일 이미지 편집

```bash
python image_editor.py \
  --model_path ./models/qwen-image-edit \
  --image input.jpg \
  --prompt "하늘을 파란색으로 바꿔주세요" \
  --output output.jpg
```

#### 폴더 내 모든 이미지 일괄 편집

```bash
python image_editor.py \
  --model_path ./models/qwen-image-edit \
  --input_folder ./images \
  --prompt "배경을 흐리게 해주세요" \
  --output_folder ./edited_images
```

## 사용 예시

### 예시 1: 배경 변경

```bash
python image_editor.py \
  --model_path ./models/qwen-image-edit \
  --image photo.jpg \
  --prompt "배경을 해변으로 바꿔주세요" \
  --output photo_beach.jpg
```

### 예시 2: 색상 보정

```bash
python image_editor.py \
  --model_path ./models/qwen-image-edit \
  --image dark_image.jpg \
  --prompt "이미지를 밝게 만들어주세요" \
  --output bright_image.jpg
```

### 예시 3: 여러 이미지 한 번에 처리

```bash
python image_editor.py \
  --model_path ./models/qwen-image-edit \
  --input_folder ./vacation_photos \
  --prompt "선명도를 높여주세요" \
  --output_folder ./enhanced_photos
```

## 고급 옵션

### GPU 사용

```bash
python image_editor.py \
  --model_path ./models/qwen-image-edit \
  --image input.jpg \
  --prompt "편집 프롬프트" \
  --output output.jpg \
  --device cuda
```

### 생성 파라미터 조정

```bash
python image_editor.py \
  --model_path ./models/qwen-image-edit \
  --image input.jpg \
  --prompt "선명하게 만들어주세요" \
  --output output.jpg \
  --num_inference_steps 50 \
  --true_cfg_scale 5.0 \
  --guidance_scale 1.0 \
  --seed 42
```

### 네거티브 프롬프트 사용

```bash
python image_editor.py \
  --model_path ./models/qwen-image-edit \
  --image input.jpg \
  --prompt "아름다운 풍경" \
  --output output.jpg \
  --negative_prompt "흐릿한, 저품질, 왜곡된"
```

## 명령줄 옵션

### 필수 옵션

| 옵션 | 설명 |
|------|------|
| `--model_path` | 모델 경로 (로컬 경로) |
| `--prompt` | 이미지 편집 프롬프트 |
| `--image` 또는 `--input_folder` | 입력 이미지/폴더 |
| `--output` 또는 `--output_folder` | 출력 이미지/폴더 |

### 선택 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--device` | auto | 디바이스 (cuda, cpu, mps) |
| `--negative_prompt` | " " | 네거티브 프롬프트 |
| `--num_inference_steps` | 40 | 추론 스텝 수 (높을수록 품질 향상) |
| `--guidance_scale` | 1.0 | 가이던스 스케일 |
| `--true_cfg_scale` | 4.0 | True CFG 스케일 |
| `--seed` | None | 랜덤 시드 (재현성) |
| `--num_images` | 1 | 생성할 이미지 수 |

## 프로젝트 구조

```
qwen-image/
├── image_editor.py      # 메인 이미지 편집 스크립트
├── download_model.py    # 모델 다운로드 스크립트
├── requirements.txt     # Python 의존성
├── README.md           # 이 파일
└── models/             # 다운로드된 모델 저장 (gitignore)
    └── qwen-image-edit/
```

## 문제 해결

### Windows에서 CUDA_PATH 에러

```
TypeError: expected str, bytes or os.PathLike object, not NoneType
```

**해결책**:
이 에러는 이미 코드에서 자동으로 처리됩니다. 하지만 여전히 발생한다면:

1. CUDA가 설치되어 있다면 환경 변수 설정:
   ```cmd
   setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
   ```
   (실제 CUDA 버전에 맞게 경로 수정)

2. 또는 CPU 모드로 실행:
   ```bash
   python image_editor.py --device cpu --model_path ... --image ... --prompt ... --output ...
   ```

### 모델 다운로드 중 MemoryError

```
MemoryError
```

**원인**: 모델이 너무 커서 메모리에 모두 로드할 수 없습니다.

**해결책**:
이 에러는 이미 수정되었습니다. 최신 코드를 pull하세요:
```bash
git pull origin claude/offline-image-editor-ircBo
```

새 버전은 `snapshot_download`를 사용하여 메모리를 적게 사용합니다.

### 모델 로딩 실패

```
에러: 모델 로딩 실패
```

**해결책**:
1. 먼저 모델을 다운로드하세요:
   ```bash
   python download_model.py --model_id Qwen/Qwen-Image-Edit-2511 --save_path ./models/qwen-image-edit
   ```
2. 모델이 올바르게 다운로드되었는지 확인
3. `--model_path`가 정확한지 확인
4. 디스크 공간이 충분한지 확인 (60GB+ 권장)

### CUDA 메모리 부족

```
RuntimeError: CUDA out of memory
```

**해결책**:
1. `--device cpu` 옵션으로 CPU 사용
2. 다른 GPU 프로그램 종료
3. 더 작은 배치로 처리

### 이미지 파일을 찾을 수 없음

```
경고: 폴더에 이미지 파일이 없습니다
```

**해결책**:
1. 입력 폴더 경로 확인
2. 지원되는 형식인지 확인 (jpg, png, bmp, webp, tiff)

## 라이선스

이 프로젝트는 Qwen-Image-Edit-2511 모델을 사용합니다.
모델의 라이선스는 [Hugging Face 모델 페이지](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)를 참조하세요.

## 파라미터 가이드

### num_inference_steps
- **설명**: 디퓨전 프로세스의 스텝 수
- **범위**: 20-100
- **권장값**: 40 (기본값)
- **효과**: 높을수록 품질이 향상되지만 속도가 느려짐

### true_cfg_scale
- **설명**: True Classifier-Free Guidance 스케일
- **범위**: 1.0-10.0
- **권장값**: 4.0 (기본값)
- **효과**: 프롬프트 충실도 조절

### guidance_scale
- **설명**: 기본 가이던스 스케일
- **범위**: 0.0-20.0
- **권장값**: 1.0 (기본값)
- **효과**: 프롬프트와 이미지의 균형 조절

### seed
- **설명**: 랜덤 시드 고정
- **사용 목적**: 동일한 결과 재현
- **예시**: `--seed 42`

## GPU VRAM별 최적화 팁

### 12-16GB VRAM (RTX 3060, RTX 4060 Ti 등)
- 모델을 bfloat16으로 로드 (기본값)
- 필요시 `--num_inference_steps 30`으로 감소
- 한 번에 하나의 이미지만 처리

### 24GB VRAM (RTX 3090, RTX 4090 등)
- 기본 설정 사용
- 배치 처리 가능
- `--num_inference_steps 50`까지 가능

### 8GB 이하 VRAM
- CPU 모드 사용 권장:
  ```bash
  python image_editor.py --device cpu ...
  ```
- 또는 양자화된 모델 사용 고려 (GGUF, FP8)

## 참고 링크

- [Qwen-Image-Edit-2511 모델](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [Diffusers 문서](https://huggingface.co/docs/diffusers)
- [PyTorch 문서](https://pytorch.org/docs/)

## 기여

이슈 및 Pull Request를 환영합니다!