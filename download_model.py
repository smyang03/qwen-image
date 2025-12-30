#!/usr/bin/env python3
"""
Qwen 이미지 편집 모델 다운로드 스크립트
오프라인 사용을 위해 모델을 로컬에 저장합니다.
"""

import argparse
import os
from pathlib import Path
from transformers import AutoProcessor, AutoModelForVision2Seq


def download_model(model_id: str, save_path: str):
    """
    Hugging Face에서 모델을 다운로드하여 로컬에 저장

    Args:
        model_id: Hugging Face 모델 ID (예: Qwen/Qwen-Image-Edit-2511)
        save_path: 저장할 로컬 경로
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"모델 다운로드 중: {model_id}")
    print(f"저장 경로: {save_path}\n")

    try:
        # 프로세서 다운로드
        print("1/2: 프로세서 다운로드 중...")
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        processor.save_pretrained(save_path)
        print("프로세서 저장 완료!\n")

        # 모델 다운로드
        print("2/2: 모델 다운로드 중...")
        print("(모델 크기가 클 수 있어 시간이 걸릴 수 있습니다)\n")
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        model.save_pretrained(save_path)
        print("모델 저장 완료!\n")

        print("=" * 60)
        print("다운로드 완료!")
        print("=" * 60)
        print(f"\n저장 위치: {save_path.absolute()}")
        print("\n사용 방법:")
        print(f"  python image_editor.py --model_path {save_path} \\")
        print(f"                         --image input.jpg \\")
        print(f"                         --prompt \"편집 프롬프트\" \\")
        print(f"                         --output output.jpg")

    except Exception as e:
        print(f"\n에러: 다운로드 실패 - {e}")
        print("\n문제 해결:")
        print("1. 인터넷 연결을 확인하세요")
        print("2. Hugging Face 토큰이 필요한 모델인지 확인하세요")
        print("3. 모델 ID가 올바른지 확인하세요")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Qwen 이미지 편집 모델 다운로드",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 다운로드
  python download_model.py --model_id Qwen/Qwen-Image-Edit-2511 \\
                           --save_path ./models/qwen-image-edit

  # 다른 경로에 저장
  python download_model.py --model_id Qwen/Qwen-Image-Edit-2511 \\
                           --save_path /data/models/qwen
        """
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen-Image-Edit-2511",
        help="Hugging Face 모델 ID (기본값: Qwen/Qwen-Image-Edit-2511)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="모델을 저장할 로컬 경로"
    )

    args = parser.parse_args()

    download_model(args.model_id, args.save_path)


if __name__ == "__main__":
    main()
