#!/usr/bin/env python3
"""
Qwen Image Edit Offline Editor - Linux Server Edition
A6000 GPU (48GB VRAM x 8) 최적화 버전
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import torch
from diffusers import QwenImageEditPlusPipeline
from tqdm import tqdm
import multiprocessing as mp
from typing import List, Tuple


class QwenImageEditor:
    """Qwen 이미지 편집 모델을 사용한 오프라인 에디터"""

    def __init__(
        self,
        model_path: str,
        device: str = None,
        torch_dtype=None,
        gpu_id: int = None,
        enable_cpu_offload: bool = False,
        sequential_cpu_offload: bool = False,
        lora_path: str = None,
        multi_gpu_model: bool = False,
        multi_gpu_data: bool = False
    ):
        """
        Args:
            model_path: 로컬 모델 경로 또는 Hugging Face 모델 ID
            device: 사용할 디바이스 (cuda, cpu 등). None이면 자동 선택
            torch_dtype: torch 데이터 타입 (기본값: bfloat16 for A6000)
            gpu_id: 사용할 GPU 인덱스 (0-7, None이면 자동 선택)
            enable_cpu_offload: CPU 오프로딩 (A6000 48GB에서는 불필요)
            sequential_cpu_offload: Sequential CPU 오프로딩 (A6000에서는 불필요)
            lora_path: LoRA 가중치 경로
            multi_gpu_model: Model Parallelism (모델을 여러 GPU에 분산, VRAM 공유)
            multi_gpu_data: Data Parallelism (배치를 여러 GPU에 분산, 병렬 추론)
        """
        self.model_path = model_path
        self.enable_cpu_offload = enable_cpu_offload
        self.sequential_cpu_offload = sequential_cpu_offload
        self.lora_path = lora_path
        self.multi_gpu_model = multi_gpu_model
        self.multi_gpu_data = multi_gpu_data
        self.is_lightning = "lightning" in model_path.lower()

        # GPU ID가 지정된 경우 해당 GPU 사용
        if gpu_id is not None and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"GPU {gpu_id} 사용으로 설정")

        # 디바이스 설정
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 멀티 GPU 정보 출력
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"\n{'='*60}")
            print(f"GPU 환경 정보")
            print(f"{'='*60}")
            print(f"사용 가능한 GPU 수: {gpu_count}")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            if multi_gpu_model and gpu_count > 1:
                print(f"멀티 GPU 모델 모드: 모델을 {gpu_count}개 GPU에 분산 (VRAM 공유)")
            elif multi_gpu_data and gpu_count > 1:
                print(f"멀티 GPU 데이터 모드: {gpu_count}개 GPU 병렬 배치 처리")
            print(f"{'='*60}\n")

        # dtype 설정 (A6000 최적화: bfloat16 기본)
        if torch_dtype is None:
            if self.device == "cuda":
                torch_dtype = torch.bfloat16  # A6000에 최적화
            else:
                torch_dtype = torch.float32

        self.torch_dtype = torch_dtype

        print(f"디바이스: {self.device}")
        print(f"데이터 타입: {torch_dtype}")
        if multi_gpu_model:
            print(f"Model Parallelism: 활성화 (VRAM 부족 시 자동 분산)")
        elif multi_gpu_data:
            print(f"Data Parallelism: 활성화 (배치 병렬 처리)")
        if sequential_cpu_offload:
            print(f"Sequential CPU 오프로딩: 활성화 (메모리 절약 모드)")
        elif enable_cpu_offload:
            print(f"CPU 오프로딩: 활성화 (메모리 절약 모드)")
        if self.is_lightning:
            print(f"Lightning 모델 감지: 4-step 최적화 모드")
        print(f"모델 로딩 중: {model_path}\n")

        # 파이프라인 로드
        try:
            print("=" * 60)
            print("안전 모드로 모델 로딩을 시작합니다...")
            print("=" * 60)

            # 로컬 경로가 존재하면 오프라인 모드로 로드
            if os.path.exists(model_path):
                print("로컬 모델 사용 (오프라인 모드)\n")

                # 더 안전한 로딩을 위한 추가 옵션
                print("1/4: 파이프라인 설정 준비 중...")
                load_kwargs = {
                    "torch_dtype": torch_dtype,
                    "local_files_only": True,
                    "low_cpu_mem_usage": True,  # 메모리 효율적 로딩
                    "use_safetensors": True,     # safetensors 우선 사용
                }

                # Model Parallelism: 모델을 여러 GPU에 자동 분산
                if multi_gpu_model:
                    load_kwargs["device_map"] = "auto"
                    print("Model Parallelism 활성화: device_map='auto'")

                print("2/4: 모델 파일 로딩 중 (시간이 걸릴 수 있습니다)...")
                self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
                    model_path,
                    **load_kwargs
                )
            else:
                print("Hugging Face에서 모델 다운로드 중...")
                load_kwargs = {
                    "torch_dtype": torch_dtype,
                    "low_cpu_mem_usage": True
                }
                if multi_gpu_model:
                    load_kwargs["device_map"] = "auto"
                    print("Model Parallelism 활성화: device_map='auto'")

                self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
                    model_path,
                    **load_kwargs
                )

            print("3/4: 모델을 GPU/CPU로 이동 중...")
            # CPU 오프로딩 또는 일반 디바이스 이동
            if multi_gpu_model:
                print("Model Parallelism: 모델이 이미 여러 GPU에 분산되어 있습니다")
                # device_map="auto" 사용 시 자동으로 분산되므로 .to() 호출 불필요
            elif sequential_cpu_offload and self.device == "cuda":
                print("Sequential CPU 오프로딩 설정 중 (최대 메모리 절약)...")
                self.pipeline.enable_sequential_cpu_offload()
                print("Sequential CPU 오프로딩 활성화 완료")
            elif enable_cpu_offload and self.device == "cuda":
                print("CPU 오프로딩 설정 중 (GPU VRAM과 시스템 RAM을 균형있게 사용)...")
                self.pipeline.enable_model_cpu_offload()
                print("CPU 오프로딩 활성화 완료")
            else:
                print(f"모델을 {self.device}로 이동 중...")
                self.pipeline.to(self.device)

            print("4/4: 메모리 최적화 설정 중...")
            # 메모리 최적화 옵션
            if self.device == "cuda":
                self.pipeline.enable_attention_slicing(1)
                print("Attention slicing 활성화 (메모리 최적화)")

                # VAE slicing도 활성화 (추가 메모리 절약)
                try:
                    self.pipeline.enable_vae_slicing()
                    print("VAE slicing 활성화 (메모리 최적화)")
                except:
                    pass

            self.pipeline.set_progress_bar_config(disable=False)

            # LoRA 로딩 (있는 경우)
            if lora_path:
                print(f"\nLoRA 가중치 로딩 중: {lora_path}")
                try:
                    if os.path.exists(lora_path):
                        self.pipeline.load_lora_weights(lora_path)
                        print("LoRA 로딩 완료")
                    else:
                        self.pipeline.load_lora_weights(lora_path)
                        print("LoRA 로딩 완료 (온라인)")
                except Exception as e:
                    print(f"경고: LoRA 로딩 실패 - {e}")
                    print("기본 모델로 계속 진행합니다...")

            print("\n" + "=" * 60)
            print("모델 로딩 완료!")
            print("=" * 60)
            print()

        except Exception as e:
            print(f"에러: 모델 로딩 실패 - {e}")
            print("\n모델을 다운로드하려면 다음 명령을 실행하세요:")
            print(f"python download_model.py --model_id Qwen/Qwen-Image-Edit-2511 --save_path {model_path}")
            raise

    def edit_image(
        self,
        image_path,
        prompt: str,
        output_path: str,
        negative_prompt: str = " ",
        num_inference_steps: int = 40,
        guidance_scale: float = 1.0,
        true_cfg_scale: float = 4.0,
        seed: int = None,
        num_images_per_prompt: int = 1
    ):
        """
        이미지 편집

        Args:
            image_path: 입력 이미지 경로 (단일 경로 또는 경로 리스트)
            prompt: 편집 프롬프트
            output_path: 출력 이미지 경로
            negative_prompt: 네거티브 프롬프트 (기본값: " ")
            num_inference_steps: 추론 스텝 수 (기본값: 40)
            guidance_scale: 가이던스 스케일 (기본값: 1.0)
            true_cfg_scale: True CFG 스케일 (기본값: 4.0)
            seed: 랜덤 시드 (재현성을 위해 사용)
            num_images_per_prompt: 프롬프트당 생성할 이미지 수
        """
        try:
            # Lightning 모델 자동 최적화
            if self.is_lightning and num_inference_steps == 40:
                num_inference_steps = 4
                print(f"Lightning 모델 감지: 추론 스텝을 {num_inference_steps}로 자동 조정")

            # 이미지 로드
            if isinstance(image_path, (list, tuple)):
                images = [Image.open(path).convert("RGB") for path in image_path]
            else:
                images = [Image.open(image_path).convert("RGB")]

            # 생성기 설정 (시드 고정)
            generator = None
            if seed is not None:
                generator = torch.manual_seed(seed)

            # 파이프라인 입력 준비
            inputs = {
                "image": images,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "true_cfg_scale": true_cfg_scale,
                "num_images_per_prompt": num_images_per_prompt,
            }
            if generator is not None:
                inputs["generator"] = generator

            # 이미지 생성
            print(f"이미지 생성 중... (steps: {num_inference_steps})")
            with torch.inference_mode():
                output = self.pipeline(**inputs)
                output_image = output.images[0]

            # 결과 저장
            output_image.save(output_path)
            print(f"저장 완료: {os.path.abspath(output_path)}")

        except Exception as e:
            print(f"에러: 이미지 편집 실패 - {e}")
            raise

    def batch_edit(
        self,
        input_folder: str,
        prompt: str,
        output_folder: str,
        **kwargs
    ):
        """
        폴더 내 모든 이미지 일괄 편집

        Args:
            input_folder: 입력 이미지 폴더
            prompt: 편집 프롬프트
            output_folder: 출력 폴더
            **kwargs: edit_image에 전달할 추가 파라미터
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

        # Data Parallelism: 여러 GPU에 이미지 분산 처리
        if self.multi_gpu_data and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                print(f"Data Parallelism: {gpu_count}개 GPU로 병렬 처리 시작")
                self._batch_edit_parallel(image_files, prompt, output_path, gpu_count, **kwargs)
                return

        # 단일 GPU 또는 일반 처리
        for idx, img_file in enumerate(tqdm(image_files, desc="이미지 편집 중")):
            output_file = output_path / img_file.name
            try:
                self.edit_image(
                    str(img_file),
                    prompt,
                    str(output_file),
                    **kwargs
                )
            except Exception as e:
                print(f"\n스킵: {img_file.name} - {e}")
                continue

    def _batch_edit_parallel(
        self,
        image_files: List[Path],
        prompt: str,
        output_path: Path,
        gpu_count: int,
        **kwargs
    ):
        """
        Data Parallelism을 사용한 병렬 배치 처리

        Args:
            image_files: 처리할 이미지 파일 리스트
            prompt: 편집 프롬프트
            output_path: 출력 폴더
            gpu_count: 사용할 GPU 개수
            **kwargs: edit_image에 전달할 추가 파라미터
        """
        # 이미지를 GPU 수만큼 분할
        chunk_size = (len(image_files) + gpu_count - 1) // gpu_count
        image_chunks = [
            image_files[i:i + chunk_size]
            for i in range(0, len(image_files), chunk_size)
        ]

        print(f"총 {len(image_files)}개 이미지를 {len(image_chunks)}개 GPU에 분산:")
        for i, chunk in enumerate(image_chunks):
            print(f"  GPU {i}: {len(chunk)}개 이미지")

        # 멀티프로세싱 시작
        ctx = mp.get_context('spawn')  # CUDA와 호환되는 spawn 방식 사용
        processes = []

        for gpu_id, chunk in enumerate(image_chunks):
            if not chunk:  # 빈 청크는 건너뛰기
                continue

            p = ctx.Process(
                target=_process_images_on_gpu,
                args=(
                    gpu_id,
                    chunk,
                    prompt,
                    output_path,
                    self.model_path,
                    self.torch_dtype,
                    self.enable_cpu_offload,
                    self.sequential_cpu_offload,
                    self.lora_path,
                    self.is_lightning,
                    kwargs
                )
            )
            p.start()
            processes.append(p)

        # 모든 프로세스 완료 대기
        for p in processes:
            p.join()

        print(f"\n병렬 처리 완료: {len(image_files)}개 이미지 처리됨")


def _process_images_on_gpu(
    gpu_id: int,
    image_files: List[Path],
    prompt: str,
    output_path: Path,
    model_path: str,
    torch_dtype,
    enable_cpu_offload: bool,
    sequential_cpu_offload: bool,
    lora_path: str,
    is_lightning: bool,
    kwargs: dict
):
    """
    특정 GPU에서 이미지 배치를 처리하는 워커 함수

    Args:
        gpu_id: GPU 번호
        image_files: 처리할 이미지 파일 리스트
        prompt: 편집 프롬프트
        output_path: 출력 폴더
        model_path: 모델 경로
        torch_dtype: PyTorch dtype
        enable_cpu_offload: CPU 오프로드 활성화 여부
        sequential_cpu_offload: Sequential CPU 오프로드 활성화 여부
        lora_path: LoRA 경로
        is_lightning: Lightning 모델 여부
        kwargs: edit_image에 전달할 추가 파라미터
    """
    try:
        # GPU 설정
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = f"cuda:0"  # 프로세스 내에서는 항상 0번 GPU

        print(f"[GPU {gpu_id}] 모델 로딩 시작...")

        # 각 GPU에서 독립적으로 모델 로드
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "local_files_only": True if os.path.exists(model_path) else False,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }

        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_path,
            **load_kwargs
        )

        # CPU 오프로딩 또는 GPU로 이동
        if sequential_cpu_offload:
            pipeline.enable_sequential_cpu_offload()
        elif enable_cpu_offload:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(device)

        # 메모리 최적화
        pipeline.enable_attention_slicing(1)
        try:
            pipeline.enable_vae_slicing()
        except:
            pass

        # LoRA 로딩
        if lora_path:
            try:
                pipeline.load_lora_weights(lora_path)
            except Exception as e:
                print(f"[GPU {gpu_id}] LoRA 로딩 실패: {e}")

        print(f"[GPU {gpu_id}] 모델 로딩 완료, {len(image_files)}개 이미지 처리 시작")

        # 이미지 처리
        for img_file in tqdm(image_files, desc=f"GPU {gpu_id}", position=gpu_id):
            output_file = output_path / img_file.name
            try:
                # 이미지 로드
                image = Image.open(str(img_file)).convert("RGB")

                # Lightning 모델 최적화
                num_steps = kwargs.get("num_inference_steps", 40)
                if is_lightning and num_steps == 40:
                    kwargs["num_inference_steps"] = 4

                # 생성기 설정
                generator = None
                seed = kwargs.get("seed")
                if seed is not None:
                    generator = torch.manual_seed(seed)

                # 파이프라인 입력 준비
                inputs = {
                    "image": [image],
                    "prompt": prompt,
                    "negative_prompt": kwargs.get("negative_prompt", " "),
                    "num_inference_steps": kwargs.get("num_inference_steps", 40),
                    "guidance_scale": kwargs.get("guidance_scale", 1.0),
                    "true_cfg_scale": kwargs.get("true_cfg_scale", 4.0),
                    "num_images_per_prompt": kwargs.get("num_images_per_prompt", 1),
                }
                if generator is not None:
                    inputs["generator"] = generator

                # 이미지 생성
                with torch.inference_mode():
                    output = pipeline(**inputs)
                    output_image = output.images[0]

                # 저장
                output_image.save(str(output_file))

            except Exception as e:
                print(f"\n[GPU {gpu_id}] 스킵: {img_file.name} - {e}")
                continue

        print(f"[GPU {gpu_id}] 처리 완료")

    except Exception as e:
        print(f"[GPU {gpu_id}] 에러: {e}")
        import traceback
        traceback.print_exc()


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

  # 고급 옵션 사용
  python image_editor.py --model_path ./models/qwen-image-edit \\
                         --image input.jpg \\
                         --prompt "선명하게 만들어주세요" \\
                         --output output.jpg \\
                         --num_inference_steps 50 \\
                         --true_cfg_scale 5.0 \\
                         --seed 42
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
        "--gpu_id",
        type=int,
        default=None,
        help="사용할 GPU 번호 (멀티 GPU 환경에서 특정 GPU 지정, 예: 0 또는 1)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["bfloat16", "float16", "float32"],
        help="데이터 타입 (기본값: bfloat16 for GPU, float32 for CPU)"
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="CPU 오프로딩 사용 (GPU VRAM과 시스템 RAM을 균형있게 사용, 메모리 절약)"
    )
    parser.add_argument(
        "--sequential-cpu-offload",
        action="store_true",
        help="Sequential CPU 오프로딩 사용 (더 공격적인 메모리 절약, cpu-offload보다 느리지만 메모리 사용량 최소화)"
    )
    parser.add_argument(
        "--multi-gpu-model",
        action="store_true",
        help="Model Parallelism: 모델을 여러 GPU에 분산 (VRAM 공유, 단일 추론)"
    )
    parser.add_argument(
        "--multi-gpu-data",
        action="store_true",
        help="Data Parallelism: 배치를 여러 GPU에 분산 (병렬 추론, ~30GB 모델용)"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="LoRA 가중치 경로 (Lightning LoRA 등 추가 가중치 로딩)"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=" ",
        help="네거티브 프롬프트 (기본값: ' ')"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=40,
        help="추론 스텝 수 (기본값: 40, 더 높을수록 품질 향상)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="가이던스 스케일 (기본값: 1.0)"
    )
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=4.0,
        help="True CFG 스케일 (기본값: 4.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="랜덤 시드 (재현성을 위해 사용)"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="생성할 이미지 수 (기본값: 1)"
    )

    args = parser.parse_args()

    # 출력 경로 검증
    if args.image and not args.output:
        parser.error("--image 사용 시 --output이 필요합니다")
    if args.input_folder and not args.output_folder:
        parser.error("--input_folder 사용 시 --output_folder가 필요합니다")

    # dtype 설정
    torch_dtype = None
    if args.dtype:
        if args.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif args.dtype == "float16":
            torch_dtype = torch.float16
        elif args.dtype == "float32":
            torch_dtype = torch.float32

    # 생성 파라미터
    generation_kwargs = {
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "true_cfg_scale": args.true_cfg_scale,
        "seed": args.seed,
        "num_images_per_prompt": args.num_images,
    }

    # 에디터 초기화
    editor = QwenImageEditor(
        args.model_path,
        device=args.device,
        torch_dtype=torch_dtype,
        gpu_id=args.gpu_id,
        enable_cpu_offload=args.cpu_offload,
        sequential_cpu_offload=args.sequential_cpu_offload,
        lora_path=args.lora_path,
        multi_gpu_model=args.multi_gpu_model,
        multi_gpu_data=args.multi_gpu_data
    )

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
