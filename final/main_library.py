import numpy as np
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from color_inference import ColorEmotionInference
    COLOR_AVAILABLE = True
except ImportError:
    COLOR_AVAILABLE = False

try:
    from clip_inference import CLIPEmotionInference
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    from face_inference import FaceEmotionAnalyzer
    FACE_AVAILABLE = True
except ImportError:
    FACE_AVAILABLE = False

try:
    from moondream_caption import MoondreamCaptioner
    MOONDREAM_AVAILABLE = True
except ImportError:
    MOONDREAM_AVAILABLE = False

try:
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    MUSICGEN_AVAILABLE = True
except ImportError:
    MUSICGEN_AVAILABLE = False

_MODELS = {
    'yolo': None,
    'color_analyzer': None,
    'face_analyzer': None,  
    'clip_analyzer': None,
    'captioner': None,
    'music_processor': None,
    'music_model': None,
    'music_device': None
}

EMOTIONS = [
    "Happiness", "Confidence", "Surprise", "Pain", "Disquietment",
    "Fear", "Yearning", "Excitement", "Embarrassment", "Affection",
    "Aversion", "Engagement", "Anticipation", "Sensitivity",
    "Annoyance", "Sympathy", "Pleasure"
]

# 1. 모델 로드
def model_load(
    yolo_model_path: str = "model/yolo/yolov8s.pt",
    clip_model_path: str = "model/clip/mlp.pt",
    moondream_model_path: Optional[str] = None,
    face_experiment_path: str = "model",
    face_model_dir: str = "emotic",
    verbose: bool = True
) -> bool:
    if verbose:
        print("감정분석 모델들을 로드하는 중...")
    
    success_count = 0
    total_count = 5
    
    # YOLO 모델 로드
    if YOLO_AVAILABLE:
        try:
            _MODELS['yolo'] = YOLO(yolo_model_path)
            if verbose:
                print(f"YOLO 모델 로드 완료: {yolo_model_path}")
            success_count += 1
        except Exception as e:
            if verbose:
                print(f"YOLO 모델 로드 실패: {e}")
            _MODELS['yolo'] = None
    else:
        if verbose:
            print("YOLO를 사용할 수 없습니다 (라이브러리 없음)")
    
    # 색상 감정 분석기 로드
    if COLOR_AVAILABLE:
        try:
            _MODELS['color_analyzer'] = ColorEmotionInference()
            if verbose:
                print("색상 감정 분석기 로드 완료")
            success_count += 1
        except Exception as e:
            if verbose:
                print(f"색상 감정 분석기 로드 실패: {e}")
            _MODELS['color_analyzer'] = None
    else:
        if verbose:
            print("색상 감정 분석기를 사용할 수 없습니다 (모듈 없음)")
    
    # 얼굴 감정 분석기 로드
    if FACE_AVAILABLE:
        try:
            _MODELS['face_analyzer'] = FaceEmotionAnalyzer(
                experiment_path=face_experiment_path,
                model_dir=face_model_dir
            )
            if verbose:
                print("얼굴 감정 분석기 로드 완료")
            success_count += 1
        except Exception as e:
            if verbose:
                print(f"얼굴 감정 분석기 로드 실패: {e}")
            _MODELS['face_analyzer'] = None
    else:
        if verbose:
            print("얼굴 감정 분석기를 사용할 수 없습니다 (모듈 없음)")
    
    # CLIP 감정 분석기 로드
    if CLIP_AVAILABLE:
        try:
            _MODELS['clip_analyzer'] = CLIPEmotionInference(clip_model_path)
            if verbose:
                print("CLIP 감정 분석기 로드 완료")
            success_count += 1
        except Exception as e:
            if verbose:
                print(f"CLIP 감정 분석기 로드 실패: {e}")
            _MODELS['clip_analyzer'] = None
    else:
        if verbose:
            print("CLIP 감정 분석기를 사용할 수 없습니다 (모듈 없음)")
    
    # Moondream 캡션 생성기 로드
    if MOONDREAM_AVAILABLE:
        try:
            _MODELS['captioner'] = MoondreamCaptioner(moondream_model_path)
            if verbose:
                print("Moondream2 캡션 생성기 로드 완료")
            success_count += 1
        except Exception as e:
            if verbose:
                print(f"Moondream2 로드 실패: {e}")
            _MODELS['captioner'] = None
    else:
        if verbose:
            print("Moondream2 캡션 생성기를 사용할 수 없습니다 (모듈 없음)")
    
    if verbose:
        print(f"모델 로드 완료! ({success_count}/{total_count} 성공)")
    
    return success_count > 0

# 2. YOLO 사람 탐지
def yolo(image_path: str, confidence: float = 0.5) -> Dict:
    if _MODELS['yolo'] is None:
        return {"has_person": False, "error": "YOLO 모델이 로드되지 않음"}
    
    try:
        results = _MODELS['yolo'](image_path, verbose=False)
        person_detections, has_person = [], False
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if class_id == 0 and conf >= confidence:  # person
                        has_person = True
                        bbox = box.xyxy[0].cpu().numpy()
                        person_detections.append({
                            "bbox": bbox.tolist(),
                            "confidence": conf,
                            "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        })
        
        return {
            "has_person": has_person,
            "person_count": len(person_detections),
            "detections": person_detections,
            "confidence_threshold": confidence
        }
    except Exception as e:
        return {"has_person": False, "error": str(e)}

# 3. 이미지 캡션 생성
def moondream2(image_path: str) -> str:
    """이미지 캡션 생성"""
    if _MODELS['captioner'] is None:
        return "Caption generation not available"
    
    try:
        return _MODELS['captioner'].generate_caption(image_path)
    except Exception as e:
        print(f"캡션 생성 실패: {e}")
        return "Caption generation failed"
    
# 4.1. 얼굴 행동 감정 추출
def emotic(image_path: str, person_detections: List[Dict]) -> Optional[Dict]:
    if _MODELS['face_analyzer'] is None:
        return None
    
    try:
        return _MODELS['face_analyzer'].analyze_emotions(image_path, person_detections)
    except Exception as e:
        print(f"얼굴 감정 분석 실패: {e}")
        return None

# 4.2. 색상 감정 추출
def color(image_path: str, n_colors: int = 5) -> Optional[Dict]:
    if _MODELS['color_analyzer'] is None:
        return None
    
    try:
        return _MODELS['color_analyzer'].predict_emotions(image_path, n_colors=n_colors)
    except Exception as e:
        print(f"색상 감정 분석 실패: {e}")
        return None

# 4.3. 캡션 감정 추출 (백슬래시 제거)
def caption(image_path: str, caption_text: str) -> Optional[Dict]:
    if _MODELS['clip_analyzer'] is None or not caption_text or caption_text == "Caption generation not available":
        return None
    
    try:
        return _MODELS['clip_analyzer'].predict_emotions(image_path, caption=caption_text)
    except Exception as e:
        print(f"캡션 감정 분석 실패: {e}")
        return None

# 5. 감정 통합 (함수명을 integrate_emotions로 변경)
def integrate_emotions(
    emotic_result: Optional[Dict] = None,
    color_result: Optional[Dict] = None,
    caption_result: Optional[Dict] = None,
    has_person: bool = True
) -> str:
    
    if has_person:
        # 사람이 있을 때: 얼굴(0.5) + 캡션(0.3) + 색상(0.2)
        face_weight = 0.5 if emotic_result and 'error' not in emotic_result else 0.0
        caption_weight = 0.3 if caption_result and 'error' not in caption_result else 0.0
        color_weight = 0.2 if color_result and 'error' not in color_result else 0.0
    else:
        # 사람이 없을 때: 색상(0.6) + 캡션(0.4)
        face_weight = 0.0
        caption_weight = 0.4 if caption_result and 'error' not in caption_result else 0.0
        color_weight = 0.6 if color_result and 'error' not in color_result else 0.0
    
    total_weight = face_weight + caption_weight + color_weight
    if total_weight == 0:
        return "Unknown"
    
    # 정규화
    face_weight /= total_weight
    caption_weight /= total_weight  
    color_weight /= total_weight
    
    # 감정 점수 통합
    integrated_scores = {}
    for emotion in EMOTIONS:
        score = 0.0
        if face_weight and emotic_result:
            score += face_weight * emotic_result.get("emotion_percentages", {}).get(emotion, 0.0)
        if caption_weight and caption_result:
            score += caption_weight * caption_result.get("emotion_percentages", {}).get(emotion, 0.0)
        if color_weight and color_result:
            score += color_weight * color_result.get("emotion_percentages", {}).get(emotion, 0.0)
        integrated_scores[emotion] = score
    
    # 가장 높은 감정 반환
    if integrated_scores:
        top_emotion = max(integrated_scores.items(), key=lambda x: x[1])
        return top_emotion[0]
    
    return "Unknown"

# 6. 전체 감정 분석 파이프라인
def emotion(image_path: str, confidence: float = 0.5, n_colors: int = 5) -> Dict:
    print(f"감정 분석 시작: {Path(image_path).name}")
    
    # 1. 사람 탐지
    person_detection = yolo(image_path, confidence)
    has_person = person_detection.get("has_person", False)
    person_detections = person_detection.get("detections", [])
    
    print(f"사람 탐지: {'있음' if has_person else '없음'}")
    
    # 2. 캡션 생성
    caption_text = moondream2(image_path)
    print(f"캡션: {caption_text[:50]}...")
    
    # 3. 개별 감정 분석
    emotic_result = None
    if has_person:
        emotic_result = emotic(image_path, person_detections)
        if emotic_result and emotic_result.get('top_emotions'):
            print(f"얼굴/행동 감정: {emotic_result['top_emotions'][0][0]} ({emotic_result['top_emotions'][0][1]:.1f}%)")
        else:
            print("얼굴/행동 감정: 분석 실패")
            
    color_result = color(image_path, n_colors)
    if color_result and color_result.get('top_emotions'):
        print(f"색상 감정: {color_result['top_emotions'][0][0]} ({color_result['top_emotions'][0][1]:.1f}%)")
    else:
        print("색상 감정: 분석 실패")
    
    caption_result = caption(image_path, caption_text)
    if caption_result and caption_result.get('top_emotions'):
        print(f"캡션 감정: {caption_result['top_emotions'][0][0]} ({caption_result['top_emotions'][0][1]:.1f}%)")
    else:
        print("캡션 감정: 분석 실패")
    
    # 4. 결과 통합
    final_emotion = integrate_emotions(emotic_result, color_result, caption_result, has_person)
    
    print(f"최종 감정: {final_emotion}")
    
    return {
        "emotion": final_emotion,
        "caption": caption_text,
        "has_person": has_person
    }

# 7. 음악 생성
def _load_musicgen_model(verbose: bool = True):
    if not MUSICGEN_AVAILABLE:
        if verbose:
            print("MusicGen을 사용할 수 없습니다 (transformers 라이브러리 없음)")
        return False
    
    if _MODELS.get('music_processor') is not None and _MODELS.get('music_model') is not None:
        if verbose:
            print("기존 MusicGEN 모델 재사용")
        return True
    
    try:
        if verbose:
            print("MusicGEN 모델 로딩 중...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODELS['music_processor'] = AutoProcessor.from_pretrained("facebook/musicgen-small")
        _MODELS['music_model'] = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(device)
        _MODELS['music_device'] = device
        
        if verbose:
            print(f"MusicGEN 모델 로딩 완료! (디바이스: {device})")
        return True
    except Exception as e:
        if verbose:
            print(f"MusicGEN 로딩 실패: {e}")
        return False

def emotion_to_music_prompt(emotion: str, caption: str = "") -> str:
    # 기본 감정 프롬프트
    base_prompt = f"music expressing {emotion.lower()} emotion"
    
    # 캡션이 있으면 추가
    if caption and caption not in ["Caption generation not available", "Caption generation failed"]:
        full_prompt = f"{base_prompt}, inspired by: {caption}"
    else:
        full_prompt = base_prompt
    
    return full_prompt

def generate_music(emotion_text: str, caption: str = "", max_duration: int = 10, verbose: bool = True) -> Dict:
    # MusicGen 모델 로드
    if not _load_musicgen_model(verbose):
        return {
            "success": False,
            "emotion": emotion_text,
            "error": "MusicGen 모델을 로드할 수 없습니다"
        }
    
    # 감정과 캡션을 음악 프롬프트로 변환
    prompt = emotion_to_music_prompt(emotion_text, caption)
    
    if verbose:
        print(f"음악 생성: {emotion_text}")
        print(f"프롬프트: {prompt}")
    
    try:
        processor = _MODELS['music_processor']
        model = _MODELS['music_model']
        device = _MODELS['music_device']
        
        # 입력 처리 후 GPU로 이동
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True,
                temperature=1.0,
                top_k=250,
                top_p=0.95
            )
        
        # CPU로 이동
        audio_data = audio_values[0, 0].detach().cpu().numpy()
        sampling_rate = model.config.audio_encoder.sampling_rate
        duration = len(audio_data) / sampling_rate
        volume = (audio_data**2).mean()**0.5
        
        if verbose:
            print(f"음악 생성 완료: {duration:.1f}초")
        
        return {
            "success": True,
            "emotion": emotion_text,
            "caption": caption,
            "prompt": prompt,
            "duration": duration,
            "volume": volume,
            "audio_data": audio_data,
            "sampling_rate": sampling_rate
        }
        
    except Exception as e:
        if verbose:
            print(f"음악 생성 실패: {e}")
        return {
            "success": False,
            "emotion": emotion_text,
            "caption": caption,
            "error": str(e)
        }
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# 백엔드 호환 함수들
def initialize_system(**kwargs) -> bool:
    return model_load(**kwargs)

def analyze_image_emotion(image_path: str, **kwargs) -> str:
    result = emotion(image_path, **kwargs)
    return result["emotion"]

# CLI 실행 (감정 분석 + 음악 생성 포함)
def main():
    parser = argparse.ArgumentParser(description="감정 분석 및 음악 생성 시스템")
    parser.add_argument('--image', required=True, help='이미지 파일 경로')
    parser.add_argument('--yolo_model', default='model/yolo/yolov8s.pt', help='YOLO 모델 경로')
    parser.add_argument('--clip_model', default='model/clip/mlp.pt', help='CLIP 모델 경로')
    parser.add_argument('--moondream_model', help='Moondream2 모델 경로')
    parser.add_argument('--face_experiment_path', default='model', help='EMOTIC 실험 경로')
    parser.add_argument('--face_model_dir', default='emotic', help='EMOTIC 모델 디렉토리')
    parser.add_argument('--confidence', type=float, default=0.5, help='YOLO 신뢰도 임계값')
    parser.add_argument('--colors', type=int, default=5, help='추출할 색상 수')
    parser.add_argument('--generate_music', action='store_true', help='음악 생성 여부')
    parser.add_argument('--music_duration', type=int, default=10, help='음악 길이 (초)')
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"이미지 파일을 찾을 수 없습니다: {args.image}")
        return

    # 모델 로드
    model_load(
        yolo_model_path=args.yolo_model,
        clip_model_path=args.clip_model,
        moondream_model_path=args.moondream_model,
        face_experiment_path=args.face_experiment_path,
        face_model_dir=args.face_model_dir
    )

    # 감정 분석 실행
    emotion_result = emotion(
        args.image,
        confidence=args.confidence,
        n_colors=args.colors
    )

    print("\n" + "="*60)
    print("감정 분석 결과")
    print("="*60)
    print(f"이미지: {args.image}")
    print(f"최종 감정: {emotion_result['emotion']}")
    print(f"캡션: {emotion_result['caption']}")

    # 음악 생성 (옵션)
    if args.generate_music:
        print("\n" + "="*60)
        print("음악 생성 시작")
        print("="*60)
        
        music_result = generate_music(
            emotion_text=emotion_result['emotion'],
            caption=emotion_result['caption'],
            max_duration=args.music_duration
        )
        
        if music_result['success']:
            print("음악 생성 성공!")
            print(f"프롬프트: {music_result['prompt']}")
            print(f"음악 길이: {music_result['duration']:.1f}초")
            
            # 음악 파일 저장 (옵션)
            output_path = Path(args.image).stem + "_generated_music.wav"
            try:
                import soundfile as sf
                sf.write(output_path, music_result['audio_data'], music_result['sampling_rate'])
                print(f"음악이 저장되었습니다: {output_path}")
            except ImportError:
                print("soundfile 라이브러리가 없어 음악 파일을 저장할 수 없습니다.")
                print("pip install soundfile로 설치 후 다시 실행하세요.")
        else:
            print(f"음악 생성 실패: {music_result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()