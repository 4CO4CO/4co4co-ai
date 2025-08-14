import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from typing import Optional
import warnings
import logging

warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MoondreamCaptioner:
    
    def __init__(self, model_path: Optional[str] = None, 
                 model_revision: str = "2025-06-21", 
                 device: str = "cuda",
                 caption_length: str = "short"):
        
        self.model_path = model_path
        self.model_revision = model_revision
        self.device = device
        self.caption_length = caption_length
        
        # 모델 변수 초기화
        self.model = None
        self.tokenizer = None
        
        logger.info(f"MoondreamCaptioner 초기화")
        # logger.info(f"   디바이스: {self.device}")
        # logger.info(f"   모델 리비전: {self.model_revision}")
        # logger.info(f"   캡션 길이: {self.caption_length}")
    
    def load_model(self):
        """Moondream2 모델 로드"""
        if self.model is not None:
            return
            
        logger.info("Moondream2 모델 로드 중...")
        try:
            # GPU 사용 설정
            if self.device == "cuda" and torch.cuda.is_available():
                logger.info(f"   GPU 사용: {torch.cuda.get_device_name(0)}")
                logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
                device_map = "cuda"
                torch_dtype = torch.float16
            else:
                logger.info("   CPU 사용")
                device_map = "cpu"
                torch_dtype = torch.float32
            
            # 모델 로드
            model_name = self.model_path if self.model_path else "vikhyatk/moondream2"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=self.model_revision,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                revision=self.model_revision
            )
            
            # 모델 디바이스 확인
            model_device = next(self.model.parameters()).device
            logger.info(f"Moondream2 모델 로드 완료! 디바이스: {model_device}")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise
    
    def generate_caption(self, image_path: str) -> str:
        try:
            # 모델이 로드되지 않았으면 로드
            if self.model is None:
                self.load_model()
            
            # 이미지 로드 및 전처리
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
            
            image = Image.open(image_path).convert('RGB')
            
            # Moondream2의 caption 메서드 사용
            caption_result = self.model.caption(
                image,
                length=self.caption_length
            )
            
            # 결과가 딕셔너리 형태인 경우 캡션 추출
            if isinstance(caption_result, dict):
                caption = caption_result.get("caption", "")
            else:
                caption = str(caption_result)
            
            if not caption.strip():
                logger.warning(f"빈 캡션 생성: {image_path.name}")
                return "Caption generation returned empty result"
            
            logger.debug(f"캡션 생성 완료: {image_path.name} -> {caption[:50]}...")
            return caption.strip()
            
        except Exception as e:
            logger.error(f"캡션 생성 실패 {image_path}: {e}")
            return f"Caption generation failed: {str(e)}"
      
    def set_caption_length(self, length: str):
        """캡션 길이 설정 변경"""
        if length in ['short', 'normal', 'long']:
            self.caption_length = length
            logger.info(f"캡션 길이 변경: {length}")
        else:
            raise ValueError(f"지원하지 않는 캡션 길이: {length}. 'short', 'normal', 'long' 중 선택하세요.")
    

# 테스트 및 단독 실행용 함수들
def test_captioner():
    """캡션 생성기 테스트"""
    # 테스트 이미지 경로 설정 (실제 이미지로 교체 필요)
    test_image = "test_image.jpg"
    
    try:
        # 캡션 생성기 초기화
        captioner = MoondreamCaptioner(
            device="cuda" if torch.cuda.is_available() else "cpu",
            caption_length="short"
        )
        
        # 테스트 이미지가 존재하는 경우 캡션 생성 테스트
        if Path(test_image).exists():
            print(f"\n=== 캡션 생성 테스트 ===")
            print(f"이미지: {test_image}")
            
            caption = captioner.generate_caption(test_image)
            print(f"생성된 캡션: {caption}")
            
            # 다른 길이로도 테스트
            for length in ['normal', 'long']:
                captioner.set_caption_length(length)
                caption = captioner.generate_caption(test_image)
                print(f"캡션 ({length}): {caption}")
        else:
            print(f"\n테스트 이미지 '{test_image}'를 찾을 수 없습니다.")
            print("실제 이미지 파일로 테스트해보세요.")
            
    except Exception as e:
        print(f"테스트 실패: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Moondream2 Caption Generator Module")
    parser.add_argument('--test', action='store_true', help='모듈 테스트 실행')
    parser.add_argument('--image', type=str, help='단일 이미지 캡션 생성')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--length', type=str, default='short', choices=['short', 'normal', 'long'])
    parser.add_argument('--model_revision', type=str, default='2025-06-21')
    
    args = parser.parse_args()
    
    if args.test:
        # 모듈 테스트 실행
        test_captioner()
    elif args.image:
        # 단일 이미지 캡션 생성
        try:
            captioner = MoondreamCaptioner(
                device=args.device,
                caption_length=args.length,
                model_revision=args.model_revision
            )
            
            caption = captioner.generate_caption(args.image)
            
            print("\n" + "="*60)
            print("MOONDREAM2 CAPTION GENERATION")
            print("="*60)
            print(f"이미지: {args.image}")
            print(f"캡션: {caption}")
            print(f"모델 정보: {captioner.get_model_info()}")
            
        except Exception as e:
            print(f"캡션 생성 실패: {e}")
    else:
        parser.print_help()