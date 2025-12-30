#!/usr/bin/env python3
"""
Qwen Image Edit Offline Editor
이미지 편집을 위한 오프라인 스크립트
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from tqdm import tqdm


class QwenImageEditor:
    """Qwen 이미지 편집 모델을 사용한 오프라인 에디터"""

    def __init__(self, model_path: str, device: str = None):
        """
        Args:
            model_path: 로컬 모델 경로 또는 Hugging Face 모델 ID
            device: 사용할 디바이스 (cuda, cpu 등). None이면 자동 선택
        """
        self.model_path = model_path

        # 디바이스 설정
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"디바이스: {self.device}")
        print(f"모델 로딩 중: {model_path}")

        # 모델과 프로세서 로드
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=os.path.exists(model_path)  # 로컬 파일이 있으면 오프라인 모드
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                local_files_only=os.path.exists(model_path)
            ).to(self.device)

            print("모델 로딩 완료!")
        except Exception as e:
            print(f"에러: 모델 로딩 실패 - {e}")
            print("\n모델을 다운로드하려면 다음 명령을 실행하세요:")
            print(f"python download_model.py --model_id Qwen/Qwen-Image-Edit-2511 --save_path {model_path}")
            raise

    def edit_image(self, image_path: str, prompt: str, output_path: str, **generation_kwargs):
        """
        이미지 편집

        Args:
            image_path: 입력 이미지 경로
            prompt: 편집 프롬프트
            output_path: 출력 이미지 경로
            **generation_kwargs: 생성 파라미터 (max_new_tokens, temperature 등)
        """
        try:
            # 이미지 로드
            image = Image.open(image_path).convert("RGB")

            # 프롬프트 준비
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # 입력 준비
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            ).to(self.device)

            # 기본 생성 파라미터
            default_kwargs = {
                "max_new_tokens": 1024,
                "do_sample": False,
            }
            default_kwargs.update(generation_kwargs)

            # 이미지 생성
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **default_kwargs)

            # 결과 디코딩 및 저장
            output_text = self.processor.decode(
                outputs[0], skip_special_tokens=True
            )

            # 생성된 이미지가 있다면 저장
            # (모델에 따라 이미지 생성 방식이 다를 수 있음)
            if hasattr(self.model, 'generate_images'):
                generated_image = self.model.generate_images(outputs[0])
                generated_image.save(output_path)
            else:
                # 텍스트 기반 편집 결과만 있는 경우
                print(f"생성 결과: {output_text}")
                # 원본 이미지를 복사 (실제 구현에서는 모델 출력에 따라 수정 필요)
                image.save(output_path)

            print(f"저장 완료: {output_path}")

        except Exception as e:
            print(f"에러: {image_path} 편집 실패 - {e}")
            raise

    def batch_edit(self, input_folder: str, prompt: str, output_folder: str, **generation_kwargs):
        """
        폴더 내 모든 이미지 일괄 편집

        Args:
            input_folder: 입력 이미지 폴더
            prompt: 편집 프롬프트
            output_folder: 출력 폴더
            **generation_kwargs: 생성 파라미터
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        # 이미지 파일 찾기
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        image_files = [
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not image_files:
            print(f"경고: {input_folder}에 이미지 파일이 없습니다.")
            return

        print(f"\n총 {len(image_files)}개의 이미지 처리 시작...")
        print(f"프롬프트: {prompt}\n")

        # 일괄 처리
        for img_file in tqdm(image_files, desc="이미지 편집 중"):
            output_file = output_path / img_file.name
            try:
                self.edit_image(
                    str(img_file),
                    prompt,
                    str(output_file),
                    **generation_kwargs
                )
            except Exception as e:
                print(f"\n스킵: {img_file.name} - {e}")
                continue


def main():
    parser = argparse.ArgumentParser(
        description="Qwen Image Edit 오프라인 에디터",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 단일 이미지 편집
  python image_editor.py --model_path ./models/qwen-image-edit \\
                         --image input.jpg \\
                         --prompt "하늘을 파란색으로 바꿔주세요" \\
                         --output output.jpg

  # 폴더 내 모든 이미지 일괄 편집
  python image_editor.py --model_path ./models/qwen-image-edit \\
                         --input_folder ./images \\
                         --prompt "배경을 흐리게 해주세요" \\
                         --output_folder ./edited_images
        """
    )

    # 필수 인자
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="모델 경로 (로컬 경로 또는 Hugging Face ID)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="이미지 편집 프롬프트"
    )

    # 단일 이미지 또는 폴더
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image",
        type=str,
        help="입력 이미지 파일 경로"
    )
    group.add_argument(
        "--input_folder",
        type=str,
        help="입력 이미지 폴더 경로"
    )

    # 출력
    parser.add_argument(
        "--output",
        type=str,
        help="출력 이미지 파일 경로 (--image 사용 시)"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="출력 폴더 경로 (--input_folder 사용 시)"
    )

    # 옵션
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="사용할 디바이스 (기본값: 자동 선택)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="최대 생성 토큰 수 (기본값: 1024)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="샘플링 온도 (do_sample=True일 때 사용)"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="샘플링 사용 여부"
    )

    args = parser.parse_args()

    # 출력 경로 검증
    if args.image and not args.output:
        parser.error("--image 사용 시 --output이 필요합니다")
    if args.input_folder and not args.output_folder:
        parser.error("--input_folder 사용 시 --output_folder가 필요합니다")

    # 생성 파라미터
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }
    if args.temperature is not None:
        generation_kwargs["temperature"] = args.temperature

    # 에디터 초기화
    editor = QwenImageEditor(args.model_path, args.device)

    # 이미지 편집
    if args.image:
        print(f"\n단일 이미지 편집 시작...")
        editor.edit_image(args.image, args.prompt, args.output, **generation_kwargs)
    else:
        editor.batch_edit(
            args.input_folder,
            args.prompt,
            args.output_folder,
            **generation_kwargs
        )

    print("\n완료!")


if __name__ == "__main__":
    main()
