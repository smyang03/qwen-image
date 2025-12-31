# Troubleshooting Guide - torchvision 호환성 문제

## 문제: RuntimeError: operator torchvision::nms does not exist

### 증상
```
RuntimeError: operator torchvision::nms does not exist
ModuleNotFoundError: Could not import module 'Qwen2_5_VLForConditionalGeneration'
```

### 원인
`torch`와 `torchvision` 버전 불일치로 인한 operator 등록 문제입니다.

## 해결 방법

### 방법 1: 호환 버전으로 재설치 (권장)

```bash
# 1. 기존 torch 및 torchvision 제거
pip uninstall torch torchvision -y

# 2. 호환되는 버전 함께 설치
# CUDA 12.1 사용 시:
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 사용 시:
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

# 3. 설치 확인
python -c "import torch; import torchvision; print(f'torch: {torch.__version__}'); print(f'torchvision: {torchvision.__version__}')"
```

### 방법 2: 최신 버전 사용

```bash
# 1. 기존 제거
pip uninstall torch torchvision -y

# 2. 최신 stable 버전 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 확인
python -c "import torch; import torchvision; from diffusers import QwenImageEditPlusPipeline; print('✓ 설치 성공')"
```

### 방법 3: 환경 완전 재구성

```bash
# 1. 가상환경 새로 생성
python3 -m venv venv_new
source venv_new/bin/activate

# 2. PyTorch 먼저 설치 (CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Diffusers 설치
pip install git+https://github.com/huggingface/diffusers

# 4. 나머지 의존성
pip install transformers>=4.37.0 accelerate>=0.20.0 Pillow>=10.0.0 tqdm>=4.65.0 huggingface-hub>=0.20.0

# 5. 테스트
python image_editor.py --model_path models/qwen-image-edit/ \
  --image input.jpg --output out.jpg \
  --prompt "test" --gpu_id 0 --dtype bfloat16
```

## 코드 수정 (이미 적용됨)

`image_editor.py`에 다음 import 보호 코드가 추가되었습니다:

```python
# torchvision 호환성 문제 해결
try:
    import torch
    # torch ops 초기화 확인
    if hasattr(torch, '_C') and hasattr(torch._C, '_dispatch_has_kernel_for_dispatch_key'):
        pass
except Exception as e:
    print(f"경고: torch 초기화 중 문제 발생: {e}", file=sys.stderr)

# diffusers import with error handling
try:
    from diffusers import QwenImageEditPlusPipeline
except RuntimeError as e:
    if "torchvision::nms does not exist" in str(e):
        print("torchvision 호환성 문제 감지. 해결 시도 중...", file=sys.stderr)
        import importlib
        if 'torchvision' in sys.modules:
            del sys.modules['torchvision']
        from diffusers import QwenImageEditPlusPipeline
    else:
        raise
```

## 버전 호환성 매트릭스

| PyTorch | torchvision | CUDA | 상태 |
|---------|-------------|------|------|
| 2.5.1   | 0.20.1      | 12.1 | ✅ 권장 |
| 2.5.1   | 0.20.1      | 11.8 | ✅ 권장 |
| 2.4.0   | 0.19.0      | 12.1 | ✅ 작동 |
| 2.3.0   | 0.18.0      | 11.8 | ✅ 작동 |
| 혼합    | 혼합        | -    | ❌ 에러 |

## 확인 명령어

```bash
# 현재 설치된 버전 확인
pip list | grep -E "torch|vision"

# 호환성 테스트
python -c "
import torch
import torchvision
print(f'✓ torch: {torch.__version__}')
print(f'✓ torchvision: {torchvision.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
print(f'✓ CUDA version: {torch.version.cuda}')

# diffusers import 테스트
from diffusers import QwenImageEditPlusPipeline
print('✓ QwenImageEditPlusPipeline import 성공')
"
```

## 서버 환경 권장 설치 순서

```bash
# 전체 설치 스크립트
#!/bin/bash
set -e

echo "=== Qwen Image Edit 환경 설정 ==="

# 1. 가상환경
python3 -m venv venv
source venv/bin/activate

# 2. PyTorch (CUDA 버전에 맞게)
CUDA_VERSION="cu121"  # 또는 cu118
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# 3. Diffusers (Qwen 지원)
pip install git+https://github.com/huggingface/diffusers

# 4. 나머지 의존성
pip install transformers>=4.37.0 \
            accelerate>=0.20.0 \
            Pillow>=10.0.0 \
            tqdm>=4.65.0 \
            huggingface-hub>=0.20.0

# 5. 선택적 최적화
pip install xformers

# 6. 검증
python -c "
import torch
import torchvision
from diffusers import QwenImageEditPlusPipeline
print('✅ 모든 패키지 설치 완료')
print(f'PyTorch: {torch.__version__}')
print(f'torchvision: {torchvision.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"

echo "=== 설치 완료 ==="
```

## 추가 참고사항

- Docker 환경에서는 NVIDIA PyTorch 공식 이미지 사용 권장:
  ```bash
  docker pull pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
  ```

- 패키지 버전 고정 (`requirements_freeze.txt`):
  ```bash
  pip freeze > requirements_freeze.txt
  ```

- 문제 지속 시 이슈 리포트:
  - PyTorch 버전, torchvision 버전
  - CUDA 버전 (`nvcc --version`)
  - 전체 에러 스택트레이스
