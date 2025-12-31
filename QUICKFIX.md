# 빠른 수정 가이드 - torchvision 에러

## ⚠️ 에러 발생 시

```
RuntimeError: operator torchvision::nms does not exist
```

이 에러는 torch와 torchvision 버전 불일치로 발생합니다.

## 🚀 자동 수정 (권장)

**서버에서 다음 스크립트를 실행하세요:**

```bash
bash fix_torchvision.sh
```

이 스크립트가 자동으로:
1. 현재 버전 확인
2. CUDA 버전 감지
3. 기존 torch/torchvision 제거
4. 호환되는 버전 설치 (torch==2.5.1, torchvision==0.20.1)
5. 설치 확인

## 🔧 수동 수정

Docker 또는 서버 환경에서:

```bash
# 1. 기존 제거
pip uninstall -y torch torchvision torchaudio

# 2. CUDA 12.x 사용 시
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 또는 CUDA 11.8 사용 시
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

# 3. 확인
python3 -c "import torch; import torchvision; from diffusers import QwenImageEditPlusPipeline; print('✓ 성공')"
```

## ✅ 수정 후 테스트

```bash
python3 image_editor.py \
  --model_path models/qwen-image-edit/ \
  --image input.jpg \
  --output out.jpg \
  --prompt "lying person" \
  --gpu_id 0 \
  --dtype bfloat16
```

## 📋 현재 버전 확인

```bash
python3 << 'EOF'
import torch
import torchvision
print(f"torch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
EOF
```

## 💡 예상 출력 (정상)

```
torch: 2.5.1
torchvision: 0.20.1
CUDA: 12.1
```

## ❌ 문제 지속 시

1. **가상환경 재생성**:
   ```bash
   python3 -m venv venv_new
   source venv_new/bin/activate
   bash fix_torchvision.sh
   ```

2. **Docker 재시작**:
   ```bash
   docker restart <container_id>
   ```

3. **상세 가이드**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md) 참고

## 📞 추가 도움

- 에러 메시지 전체를 복사하여 이슈 제출
- torch, torchvision, CUDA 버전 정보 포함
