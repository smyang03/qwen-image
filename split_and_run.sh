#!/bin/bash
# 2개 프로세스로 병렬 처리 스크립트

set -e

PROMPT="$1"
OUTPUT_BASE="$2"

if [ -z "$PROMPT" ] || [ -z "$OUTPUT_BASE" ]; then
    echo "사용법: $0 \"프롬프트\" 출력폴더명"
    echo "예시: $0 \"Add a worker in navy blue work uniform lying on the ground\" output_navy"
    exit 1
fi

# 1. 이미지 폴더 분리
echo "====================="
echo "1. 이미지 분리 중..."
echo "====================="

INPUT_DIR="background/JPEGImages"
PART1_DIR="background/JPEGImages_part1"
PART2_DIR="background/JPEGImages_part2"

mkdir -p "$PART1_DIR"
mkdir -p "$PART2_DIR"

# 전체 이미지 파일 목록
ALL_IMAGES=($(ls "$INPUT_DIR" | grep -E '\.(jpg|jpeg|png|JPG|JPEG|PNG)$'))
TOTAL=${#ALL_IMAGES[@]}
HALF=$((TOTAL / 2))

echo "총 이미지: $TOTAL 개"
echo "Part 1: $HALF 개"
echo "Part 2: $((TOTAL - HALF)) 개"

# Part 1 (첫 절반)
for ((i=0; i<$HALF; i++)); do
    ln -sf "$(pwd)/$INPUT_DIR/${ALL_IMAGES[$i]}" "$PART1_DIR/${ALL_IMAGES[$i]}" 2>/dev/null || true
done

# Part 2 (나머지 절반)
for ((i=$HALF; i<$TOTAL; i++)); do
    ln -sf "$(pwd)/$INPUT_DIR/${ALL_IMAGES[$i]}" "$PART2_DIR/${ALL_IMAGES[$i]}" 2>/dev/null || true
done

echo "✓ 이미지 분리 완료 (심볼릭 링크 생성)"

# 2. 출력 폴더 생성
OUTPUT1="background/${OUTPUT_BASE}_part1"
OUTPUT2="background/${OUTPUT_BASE}_part2"
OUTPUT_MERGED="background/${OUTPUT_BASE}"

mkdir -p "$OUTPUT1"
mkdir -p "$OUTPUT2"
mkdir -p "$OUTPUT_MERGED"

# 3. 2개 프로세스 동시 실행
echo ""
echo "====================="
echo "2. 병렬 처리 시작..."
echo "====================="
echo "GPU 0-3: Part 1 ($HALF 개)"
echo "GPU 4-7: Part 2 ($((TOTAL - HALF)) 개)"
echo ""

# GPU 0-3에서 Part 1 처리 (백그라운드)
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 image_editor.py \
  --model_path models/qwen-image-edit/ \
  --input_folder "$PART1_DIR" \
  --output_folder "$OUTPUT1" \
  --prompt "$PROMPT" \
  --dtype bfloat16 \
  --multi-gpu-model \
  --batch-size 4 \
  > logs/part1.log 2>&1 &

PID1=$!
echo "프로세스 1 시작 (PID: $PID1) - GPU 0-3"

# 잠시 대기 (모델 로딩 충돌 방지)
sleep 30

# GPU 4-7에서 Part 2 처리 (백그라운드)
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 image_editor.py \
  --model_path models/qwen-image-edit/ \
  --input_folder "$PART2_DIR" \
  --output_folder "$OUTPUT2" \
  --prompt "$PROMPT" \
  --dtype bfloat16 \
  --multi-gpu-model \
  --batch-size 4 \
  > logs/part2.log 2>&1 &

PID2=$!
echo "프로세스 2 시작 (PID: $PID2) - GPU 4-7"

echo ""
echo "====================="
echo "진행 상황 모니터링"
echo "====================="
echo "로그 확인:"
echo "  tail -f logs/part1.log"
echo "  tail -f logs/part2.log"
echo ""
echo "GPU 사용률:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "완료 대기 중..."

# 두 프로세스 완료 대기
wait $PID1
wait $PID2

echo ""
echo "====================="
echo "3. 결과 합치기..."
echo "====================="

# 결과 합치기
cp -v "$OUTPUT1"/* "$OUTPUT_MERGED/" 2>/dev/null || true
cp -v "$OUTPUT2"/* "$OUTPUT_MERGED/" 2>/dev/null || true

RESULT_COUNT=$(ls "$OUTPUT_MERGED" | wc -l)

echo "✓ 완료!"
echo "  처리된 이미지: $RESULT_COUNT 개"
echo "  출력 폴더: $OUTPUT_MERGED"
