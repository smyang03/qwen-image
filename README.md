# Qwen Image Edit - Offline Editor

[Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) 모델 기반 오프라인 이미지 편집기

## 주요 기능

✅ **완전한 오프라인 동작**: 모델을 한 번 다운로드하면 인터넷 없이 사용 가능
✅ **배치 처리**: 폴더 내 모든 이미지를 한 번에 편집
✅ **간단한 사용법**: 커맨드라인 인터페이스로 쉽게 사용
✅ **유연한 설정**: 프롬프트, 입출력 경로 자유롭게 설정

## 설치

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

### 3. 의존성 설치

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

### 모델 로딩 실패

```
에러: 모델 로딩 실패
```

**해결책**:
1. 모델이 올바르게 다운로드되었는지 확인
2. `--model_path`가 정확한지 확인
3. 디스크 공간이 충분한지 확인

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

## 참고 링크

- [Qwen-Image-Edit-2511 모델](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [Diffusers 문서](https://huggingface.co/docs/diffusers)
- [PyTorch 문서](https://pytorch.org/docs/)

## 기여

이슈 및 Pull Request를 환영합니다!