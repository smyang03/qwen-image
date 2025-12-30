#!/usr/bin/env python3
"""
Qwen 이미지 편집 모델 다운로드 스크립트
오프라인 사용을 위해 모델을 로컬에 저장합니다.
"""

import os
import sys

# Windows 호환성: triton CUDA_PATH 에러 방지
if sys.platform == "win32" and "CUDA_PATH" not in os.environ:
    # CUDA_PATH가 설정되지 않은 경우 기본값 설정
    # 일반적인 CUDA 설치 경로들을 확인
    possible_cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6",
    ]

    cuda_path_found = False
    for path in possible_cuda_paths:
        if os.path.exists(path):
            os.environ["CUDA_PATH"] = path
            cuda_path_found = True
            break

    # CUDA 경로를 찾지 못한 경우 더미 경로 설정 (CPU 모드용)
    if not cuda_path_found:
        os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def download_model(model_id: str, save_path: str):
    """
    Hugging Face에서 모델을 다운로드하여 로컬에 저장

    메모리 효율적인 방식으로 직접 파일을 다운로드합니다.

    Args:
        model_id: Hugging Face 모델 ID (예: Qwen/Qwen-Image-Edit-2511)
        save_path: 저장할 로컬 경로
    """
    save_path = Path(save_path)

    print("=" * 60)
    print(f"모델 다운로드 중: {model_id}")
    print(f"저장 경로: {save_path.absolute()}")
    print("=" * 60)
    print("\n주의사항:")
    print("- 모델 크기가 매우 크므로 다운로드에 시간이 걸립니다 (약 10-60GB)")
    print("- 안정적인 인터넷 연결이 필요합니다")
    print("- 충분한 디스크 공간이 있는지 확인하세요\n")

    try:
        print("파이프라인 및 모든 구성 요소 다운로드 중...")
        print("(UNet, VAE, Text Encoder, Tokenizer 등이 포함됩니다)")
        print("이 작업은 메모리를 적게 사용하는 직접 파일 다운로드 방식을 사용합니다.\n")

        # snapshot_download를 사용하여 메모리 효율적으로 다운로드
        # 모델을 메모리에 로드하지 않고 파일을 직접 디스크에 저장
        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=save_path,
            local_dir_use_symlinks=False,  # 심볼릭 링크 대신 실제 파일 복사
            resume_download=True,  # 중단된 다운로드 재개 가능
        )

        print("\n" + "=" * 60)
        print("다운로드 완료!")
        print("=" * 60)
        print(f"\n저장 위치: {Path(downloaded_path).absolute()}")

        # 폴더 크기 계산
        print("\n폴더 크기 계산 중...")
        total_size = 0
        file_count = 0
        for dirpath, dirnames, filenames in os.walk(save_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
                    file_count += 1

        size_gb = total_size / (1024**3)
        print(f"  파일 수: {file_count}개")
        print(f"  총 크기: {size_gb:.2f} GB\n")

        print("=" * 60)
        print("사용 방법:")
        print("=" * 60)
        print(f"\npython image_editor.py \\")
        print(f"  --model_path {save_path} \\")
        print(f"  --image input.jpg \\")
        print(f"  --prompt \"편집 프롬프트\" \\")
        print(f"  --output output.jpg")
        print("\n이제 오프라인에서도 사용 가능합니다!")
        print("=" * 60)

    except Exception as e:
        print(f"\n" + "=" * 60)
        print("에러: 다운로드 실패")
        print("=" * 60)
        print(f"\n에러 메시지: {e}\n")
        print("문제 해결:")
        print("1. 인터넷 연결을 확인하세요")
        print("2. Hugging Face 토큰이 필요한 모델인지 확인하세요")
        print("   토큰 설정: huggingface-cli login")
        print("3. 모델 ID가 올바른지 확인하세요: " + model_id)
        print("4. 디스크 공간이 충분한지 확인하세요 (최소 60GB 권장)")
        print("5. 다운로드가 중단된 경우 같은 명령을 다시 실행하면 이어서 다운로드됩니다")
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

참고:
  - diffusers 라이브러리가 설치되어 있어야 합니다
  - 인터넷 연결이 필요합니다
  - 충분한 디스크 공간이 있어야 합니다 (약 10GB+)
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
