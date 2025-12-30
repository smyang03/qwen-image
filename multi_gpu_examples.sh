#!/bin/bash
# Multi-GPU Usage Examples for Qwen Image Edit
# A6000 GPU (48GB VRAM x 8) 최적화 예제

# ============================================
# Model Parallelism (VRAM 공유 방식)
# ============================================
# 사용 시나리오: 모델이 단일 GPU VRAM(48GB)보다 클 경우
# - 모델을 여러 GPU에 자동 분산 (device_map="auto")
# - 단일 추론이지만 VRAM을 여러 GPU가 공유
# - 예: 모델이 60GB라면 GPU 0에 30GB, GPU 1에 30GB 분산

echo "=== Model Parallelism 예제 ==="
python image_editor.py \
  --model_path ./models/qwen-image-edit \
  --image input.jpg \
  --prompt "make the sky blue and vibrant" \
  --output output.jpg \
  --dtype bfloat16 \
  --multi-gpu-model

# ============================================
# Data Parallelism (배치 병렬 처리 방식)
# ============================================
# 사용 시나리오: 모델이 단일 GPU에 맞지만 여러 이미지를 병렬 처리하고 싶을 경우
# - 모델 복사본을 각 GPU에 로드 (~30GB x 8 = 240GB 총 VRAM 사용)
# - 8개 GPU가 동시에 각각 다른 이미지를 처리
# - 예: 800장 이미지를 8개 GPU가 각각 100장씩 병렬 처리

echo "=== Data Parallelism 예제 ==="
python image_editor.py \
  --model_path ./models/qwen-image-edit \
  --input_folder ./input_images \
  --prompt "enhance image quality and brightness" \
  --output_folder ./output_images \
  --dtype bfloat16 \
  --multi-gpu-data

# ============================================
# 단일 GPU 사용 (기본)
# ============================================
# 사용 시나리오: 테스트하거나 특정 GPU만 사용하고 싶을 경우

echo "=== 단일 GPU 예제 (GPU 0 사용) ==="
python image_editor.py \
  --model_path ./models/qwen-image-edit \
  --image input.jpg \
  --prompt "add more contrast" \
  --output output.jpg \
  --gpu_id 0 \
  --dtype bfloat16

# ============================================
# Lightning 모델 + Data Parallelism
# ============================================
# 사용 시나리오: 경량화 모델로 빠른 배치 처리

echo "=== Lightning 모델 + Data Parallelism 예제 ==="
python image_editor.py \
  --model_path ./models/qwen-image-edit-lightning \
  --input_folder ./large_batch \
  --prompt "make it professional looking" \
  --output_folder ./processed_batch \
  --dtype bfloat16 \
  --multi-gpu-data \
  --num_inference_steps 4

# ============================================
# 고급 옵션: CPU 오프로딩 + Model Parallelism
# ============================================
# 사용 시나리오: VRAM이 매우 부족한 경우 (일반적으로 A6000에서는 불필요)

echo "=== CPU 오프로딩 + Model Parallelism 예제 ==="
python image_editor.py \
  --model_path ./models/qwen-image-edit \
  --image input.jpg \
  --prompt "artistic rendering" \
  --output output.jpg \
  --dtype bfloat16 \
  --multi-gpu-model \
  --sequential-cpu-offload

# ============================================
# 성능 비교 가이드
# ============================================
cat << 'EOF'

=== 성능 비교 및 선택 가이드 ===

1. Model Parallelism (--multi-gpu-model):
   장점:
   - 큰 모델도 로드 가능
   - VRAM 부족 문제 해결

   단점:
   - GPU 간 데이터 전송 오버헤드
   - 단일 이미지 처리 시간은 동일

   권장 사용:
   - 모델 크기 > 48GB (단일 A6000 VRAM 초과)
   - 단일 고해상도 이미지 처리

2. Data Parallelism (--multi-gpu-data):
   장점:
   - 배치 처리 속도 8배 향상 (8 GPU 사용 시)
   - 대량 이미지 처리에 최적

   단점:
   - 각 GPU에 모델 복사본 필요 (총 VRAM 사용량 증가)
   - 단일 이미지에는 이점 없음

   권장 사용:
   - 모델 크기 < 48GB
   - 대량 배치 처리 (수백~수천 장 이미지)
   - A6000 8장 환경에 최적화

3. 단일 GPU:
   권장 사용:
   - 테스트 및 개발
   - 소량 이미지 처리
   - 특정 GPU만 사용하고 싶을 때

=== GPU 메모리 예상 사용량 ===

Qwen-Image-Edit-2511 (bfloat16):
- 모델: ~25-30GB
- 추론 시 추가: ~5-10GB
- 총: ~35-40GB (A6000 48GB에 여유 있음)

Model Parallelism 사용 시 (2 GPU):
- GPU 0: ~15-20GB
- GPU 1: ~15-20GB

Data Parallelism 사용 시 (8 GPU):
- 각 GPU: ~35-40GB
- 총 시스템: ~280-320GB VRAM 사용

=== 권장 설정 (A6000 x 8 환경) ===

대량 배치 처리 (추천):
  python image_editor.py \
    --model_path ./models/qwen-image-edit \
    --input_folder ./images \
    --output_folder ./edited \
    --prompt "your prompt" \
    --multi-gpu-data \
    --dtype bfloat16

단일 이미지 고품질:
  python image_editor.py \
    --model_path ./models/qwen-image-edit \
    --image input.jpg \
    --output output.jpg \
    --prompt "your prompt" \
    --gpu_id 0 \
    --dtype bfloat16 \
    --num_inference_steps 50

EOF
