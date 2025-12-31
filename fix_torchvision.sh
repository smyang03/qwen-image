#!/bin/bash
# torchvision 호환성 문제 자동 해결 스크립트

set -e

echo "================================================================"
echo "Qwen Image Edit - torchvision 호환성 문제 자동 해결"
echo "================================================================"
echo ""

# 1. 현재 설치된 버전 확인
echo "[1/5] 현재 패키지 버전 확인..."
echo ""

if python3 -c "import torch" 2>/dev/null; then
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    echo "  현재 torch: $TORCH_VER"
else
    TORCH_VER="not installed"
    echo "  torch: 설치 안 됨"
fi

if python3 -c "import torchvision" 2>/dev/null; then
    TORCHVISION_VER=$(python3 -c "import torchvision; print(torchvision.__version__)" 2>/dev/null || echo "unknown")
    echo "  현재 torchvision: $TORCHVISION_VER"
else
    TORCHVISION_VER="not installed"
    echo "  torchvision: 설치 안 됨"
fi

if python3 -c "import torch; print(f'CUDA: {torch.version.cuda}')" 2>/dev/null; then
    CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
    echo "  CUDA 버전: $CUDA_VER"
else
    CUDA_VER="unknown"
fi

echo ""
echo "================================================================"

# 2. CUDA 버전 감지
echo ""
echo "[2/5] CUDA 버전 감지..."

if command -v nvcc &> /dev/null; then
    NVCC_VER=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo "  nvcc 버전: $NVCC_VER"

    if [[ "$NVCC_VER" == "12."* ]]; then
        CUDA_INDEX="cu121"
        echo "  → CUDA 12.x 감지, cu121 사용"
    elif [[ "$NVCC_VER" == "11.8"* ]]; then
        CUDA_INDEX="cu118"
        echo "  → CUDA 11.8 감지, cu118 사용"
    else
        CUDA_INDEX="cu121"
        echo "  → 기본값 cu121 사용"
    fi
else
    CUDA_INDEX="cu121"
    echo "  nvcc를 찾을 수 없습니다. 기본값 cu121 사용"
fi

echo ""
echo "================================================================"

# 3. 기존 torch/torchvision 제거
echo ""
echo "[3/5] 기존 torch/torchvision 제거..."
echo ""

pip uninstall -y torch torchvision torchaudio || true

echo ""
echo "================================================================"

# 4. 호환되는 버전 설치
echo ""
echo "[4/5] 호환되는 torch 및 torchvision 설치..."
echo "  torch==2.5.1"
echo "  torchvision==0.20.1"
echo "  CUDA: $CUDA_INDEX"
echo ""

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/${CUDA_INDEX}

echo ""
echo "================================================================"

# 5. 설치 확인
echo ""
echo "[5/5] 설치 확인..."
echo ""

python3 << 'EOF'
import sys

try:
    import torch
    import torchvision

    print(f"✓ torch: {torch.__version__}")
    print(f"✓ torchvision: {torchvision.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # QwenImageEditPlusPipeline import 테스트
    print("")
    print("diffusers import 테스트 중...")
    from diffusers import QwenImageEditPlusPipeline
    print("✓ QwenImageEditPlusPipeline import 성공!")

    print("")
    print("=" * 60)
    print("모든 패키지가 성공적으로 설치되었습니다!")
    print("=" * 60)
    sys.exit(0)

except Exception as e:
    print(f"✗ 에러 발생: {e}")
    print("")
    print("=" * 60)
    print("설치 실패. 에러를 확인하세요.")
    print("=" * 60)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo "수정 완료! 이제 image_editor.py를 실행할 수 있습니다."
    echo "================================================================"
    echo ""
    echo "테스트 명령어:"
    echo "  python3 image_editor.py --model_path models/qwen-image-edit/ \\"
    echo "    --image input.jpg --output out.jpg \\"
    echo "    --prompt \"lying person\" --gpu_id 0 --dtype bfloat16"
    echo ""
else
    echo ""
    echo "================================================================"
    echo "수정 실패. TROUBLESHOOTING.md를 참고하세요."
    echo "================================================================"
    echo ""
    exit 1
fi
