# main.py (기존 conditional_emotion_single.py)
import numpy as np
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# ── 외부 모듈 로드 (선택 모듈은 없어도 동작) ─────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("YOLOv8이 설치되지 않았습니다. pip install ultralytics")
    YOLO_AVAILABLE = False

try:
    from color_inference import ColorEmotionInference
    COLOR_AVAILABLE = True
except ImportError:
    print("color_inference.py 모듈을 찾을 수 없습니다.")
    COLOR_AVAILABLE = False

try:
    from clip_inference import CLIPEmotionInference
    CLIP_AVAILABLE = True
except ImportError:
    print("clip_inference.py 모듈을 찾을 수 없습니다.")
    CLIP_AVAILABLE = False

try:
    from face_inference import FaceEmotionAnalyzer
    FACE_AVAILABLE = True
except ImportError:
    print("face_inference.py 모듈을 찾을 수 없습니다. ")
    FACE_AVAILABLE = False

try:
    from moondream_caption import MoondreamCaptioner
    MOONDREAM_AVAILABLE = True
except ImportError:
    print("moondream_caption.py 모듈을 찾을 수 없습니다. ")
    MOONDREAM_AVAILABLE = False


# ── 구성 요소 ───────────────────────────────────────────────────────────
class PersonDetector:
    """YOLO를 사용한 사람 탐지"""
    def __init__(self, model_path: str = "yolov8n.pt"):
        if not YOLO_AVAILABLE:
            self.model = None
            print("YOLO를 사용할 수 없습니다.")
            return
        try:
            self.model = YOLO(model_path)
            print(f"YOLO 모델 로드 완료: {model_path}")
        except Exception as e:
            print(f"YOLO 모델 로드 실패: {e}")
            self.model = None

    def detect_person(self, image_path: str, confidence_threshold: float = 0.5) -> Dict:
        if self.model is None:
            return {"has_person": False, "error": "YOLO 모델이 로드되지 않음"}
        try:
            results = self.model(image_path, verbose=False)
            person_detections, has_person = [], False
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        if class_id == 0 and confidence >= confidence_threshold:  # person
                            has_person = True
                            bbox = box.xyxy[0].cpu().numpy()
                            person_detections.append({
                                "bbox": bbox.tolist(),
                                "confidence": confidence,
                                "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            })
            return {
                "has_person": has_person,
                "person_count": len(person_detections),
                "detections": person_detections,
                "confidence_threshold": confidence_threshold
            }
        except Exception as e:
            return {"has_person": False, "error": str(e)}


class CaptionGenerator:
    """Moondream2를 사용한 캡션 생성 (선택)"""
    def __init__(self, model_path: Optional[str] = None):
        if MOONDREAM_AVAILABLE:
            try:
                self.captioner = MoondreamCaptioner(model_path)
                print("Moondream2 캡션 생성기 로드 완료")
            except Exception as e:
                print(f"Moondream2 로드 실패: {e}")
                self.captioner = None
        else:
            self.captioner = None

    def generate_caption(self, image_path: str) -> str:
        if self.captioner is None:
            return "Caption generation not available"
        try:
            return self.captioner.generate_caption(image_path)
        except Exception as e:
            print(f"캡션 생성 실패: {e}")
            return "Caption generation failed"




class ConditionalEmotionSystem:
    """조건부 감정 분석 시스템 - 단일 이미지 전용"""
    def __init__(self, yolo_model_path: str = "yolov8n.pt",
                 clip_model_path: Optional[str] = None,
                 moondream_model_path: Optional[str] = None,
                 face_experiment_path: str = "model",
                 face_model_dir: str = "emotic"):
        print("조건부 감정 분석 시스템 초기화 중...")
        self.person_detector = PersonDetector(yolo_model_path)
        self.caption_generator = CaptionGenerator(moondream_model_path)

        if COLOR_AVAILABLE:
            self.color_analyzer = ColorEmotionInference()
            print("색상 감정 분석기 로드 완료")
        else:
            self.color_analyzer = None
            print("색상 감정 분석기를 사용할 수 없습니다.")

        # 🚨 버그 수정: face_analyzer 초기화 부분
        if FACE_AVAILABLE:
            try:
                self.face_analyzer = FaceEmotionAnalyzer(
                    experiment_path=face_experiment_path,
                    model_dir=face_model_dir
                )
                print("얼굴 감정 분석기 로드 완료")
            except Exception as e:
                print(f"얼굴 감정 분석기 로드 실패: {e}")
                self.face_analyzer = None
        else:
            self.face_analyzer = None 
            print("얼굴 감정 분석기를 사용할 수 없습니다.")

        if CLIP_AVAILABLE:
            try:
                self.clip_analyzer = CLIPEmotionInference(clip_model_path)
                print("CLIP 감정 분석기 로드 완료")
            except Exception as e:
                print(f"CLIP 감정 분석기 로드 실패: {e}")
                self.clip_analyzer = None
        else:
            self.clip_analyzer = None
            print("CLIP 감정 분석기를 사용할 수 없습니다.")

        self.emotions = [
            "Happiness", "Confidence", "Surprise", "Pain", "Disquietment",
            "Fear", "Yearning", "Excitement", "Embarrassment", "Affection",
            "Aversion", "Engagement", "Anticipation", "Sensitivity",
            "Annoyance", "Sympathy", "Pleasure"
        ]
        print("시스템 초기화 완료!")

    def analyze_emotion(self, image_path: str, **kwargs) -> Dict:
        print(f"\n{'='*60}")
        print(f"조건부 감정 분석 시작: {Path(image_path).name}")
        print(f"{'='*60}")

        # Step 1: 사람 탐지
        print("Step 1: 사람 탐지 중...")
        detection_result = self.person_detector.detect_person(
            image_path,
            kwargs.get('confidence_threshold', 0.5)
        )
        if detection_result.get("error"):
            print(f"사람 탐지 실패: {detection_result['error']}")
            return {"error": detection_result["error"]}

        has_person = detection_result["has_person"]
        person_detections = detection_result.get("detections", [])
        print(f"사람 탐지 결과: {'있음' if has_person else '없음'} (총 {len(person_detections)}명)")

        analysis_results = {
            "image_path": image_path,
            "has_person": has_person,
            "person_count": len(person_detections),
            "detection_info": detection_result
        }

        if has_person:
            analysis_results.update(self._analyze_person_present(image_path, person_detections, **kwargs))
        else:
            analysis_results.update(self._analyze_scene_only(image_path, **kwargs))

        print("감정 분석 완료!")
        return analysis_results

    def _analyze_person_present(self, image_path: str, person_detections: List[Dict], **kwargs) -> Dict:
        print("\n--- 사람 있음: 3가지 요소 분석 ---")
        
        # (A) 얼굴/표정 분석
        face_results = None
        if self.face_analyzer is not None:
            print("Step 2a: 얼굴/표정 감정 분석 중...")
            try:
                face_results = self.face_analyzer.analyze_emotions(image_path, person_detections)
                if face_results and 'error' not in face_results:
                    print(f"얼굴 감정 분석 완료 (상위 감정: {face_results['top_emotions'][0][0] if face_results.get('top_emotions') else 'N/A'})")
                else:
                    print(f"얼굴 감정 분석 실패: {face_results.get('error', 'Unknown error')}")
                    face_results = None
            except Exception as e:
                print(f"얼굴 감정 분석 중 오류: {e}")
                face_results = None
        else:
            print("Step 2a: 얼굴/표정 감정 분석 건너뜀 (모듈 없음)")

        color_results = None
        if self.color_analyzer is not None:
            print("Step 2b: 색상 감정(장면) 분석 중...")
            try:
                color_results = self.color_analyzer.predict_emotions(
                    image_path,
                    n_colors=kwargs.get('n_colors', 5)
                )
                print(f"색상 감정 분석 완료 (상위 감정: {color_results['top_emotions'][0][0] if color_results.get('top_emotions') else 'N/A'})")
            except Exception as e:
                print(f"색상 감정 분석 실패: {e}")
                color_results = None
        else:
            print("Step 2b: 색상 감정 분석 건너뜀 (모듈 없음)")

        # (C) 캡션 생성
        print("Step 3a: 캡션 생성 중...")
        caption = self.caption_generator.generate_caption(image_path)
        print(f"캡션 생성 완료: {caption[:50]}..." if isinstance(caption, str) and len(caption) > 50 else f"캡션: {caption}")

        # (D) 캡션 기반 감정 분석
        caption_results = None
        if self.clip_analyzer and caption and caption != "Caption generation not available":
            print("Step 4a: 캡션 기반 감정 분석 중...")
            try:
                caption_results = self.clip_analyzer.predict_emotions(image_path, caption=caption)
                print(f"캡션 감정 분석 완료 (상위 감정: {caption_results['top_emotions'][0][0] if caption_results and caption_results.get('top_emotions') else 'N/A'})")
            except Exception as e:
                print(f"캡션 감정 분석 실패: {e}")
                caption_results = None
        else:
            print("Step 4a: 캡션 기반 감정 분석 건너뜀")

        print("Step 5a: 결과 통합 중...")
        # ✅ color_results를 함께 넘김
        final_results = self._integrate_person_analysis(face_results, caption_results, caption, color_results)
        print(f"통합 완료 (최종 상위 감정: {final_results['top_emotions'][0][0] if final_results['top_emotions'] else 'N/A'})")

        return {
            "analysis_type": "person_present",
            "caption": caption,
            "face_emotion_results": face_results,
            "color_emotion_results": color_results,   
            "caption_emotion_results": caption_results,
            "final_emotions": final_results
        }


    def _analyze_scene_only(self, image_path: str, **kwargs) -> Dict:
        print("\n--- 사람 없음: 2가지 요소 분석 ---")
        print("Step 2b: 색상 감정 분석 중...")
        if self.color_analyzer:
            color_results = self.color_analyzer.predict_emotions(
                image_path,
                n_colors=kwargs.get('n_colors', 5)
            )
            print(f"색상 감정 분석 완료 (상위 감정: {color_results['top_emotions'][0][0] if color_results.get('top_emotions') else 'N/A'})")
        else:
            print("색상 분석기를 사용할 수 없습니다.")
            return {"error": "색상 분석기를 사용할 수 없습니다."}

        print("Step 3b: 캡션 생성 중...")
        caption = self.caption_generator.generate_caption(image_path)
        print(f"캡션 생성 완료: {caption[:50]}..." if isinstance(caption, str) and len(caption) > 50 else f"캡션: {caption}")

        caption_results = None
        if self.clip_analyzer and caption and caption != "Caption generation not available":
            print("Step 4b: 캡션 기반 감정 분석 중...")
            try:
                caption_results = self.clip_analyzer.predict_emotions(image_path, caption=caption)
                print(f"캡션 감정 분석 완료 (상위 감정: {caption_results['top_emotions'][0][0] if caption_results and caption_results.get('top_emotions') else 'N/A'})")
            except Exception as e:
                print(f"캡션 감정 분석 실패: {e}")
                caption_results = None
        else:
            print("Step 4b: 캡션 기반 감정 분석 건너뜀")

        print("Step 5b: 결과 통합 중...")
        final_results = self._integrate_scene_analysis(color_results, caption_results)
        print(f"통합 완료 (최종 상위 감정: {final_results['top_emotions'][0][0] if final_results['top_emotions'] else 'N/A'})")

        return {
            "analysis_type": "scene_only",
            "caption": caption,
            "color_emotion_results": color_results,
            "caption_emotion_results": caption_results,
            "final_emotions": final_results
        }

    # ── 통합 로직 ────────────────────────────────────────────
    def _integrate_person_analysis(self,
                                face_results: Dict,
                                caption_results: Optional[Dict],
                                caption: str,
                                color_results: Optional[Dict] = None) -> Dict:
        """사람이 있을 때: 얼굴/캡션/색상 점수 통합"""

        # 기본 가중치 (원하면 여기 숫자 조정)
        face_weight    = 0.5 if face_results   and 'error' not in face_results   else 0.0
        caption_weight = 0.3 if caption_results and 'error' not in caption_results else 0.0
        color_weight   = 0.2 if color_results   and 'error' not in color_results   else 0.0

        total_weight = face_weight + caption_weight + color_weight
        if total_weight == 0:
            return {
                "method": "integrated_person_analysis",
                "weights": {"face": 0, "caption": 0, "color": 0},
                "emotion_percentages": {e: 0.0 for e in self.emotions},
                "top_emotions": [],
                "predicted_emotions": [],
                "error": "모든 분석 방법이 실패했습니다."
            }

        # 정규화
        face_weight    /= total_weight
        caption_weight /= total_weight
        color_weight   /= total_weight

        # 감정 점수 통합
        integrated_scores = {}
        for emotion in self.emotions:
            s = 0.0
            if face_weight:
                s += face_weight * face_results.get("emotion_percentages", {}).get(emotion, 0.0)
            if caption_weight:
                s += caption_weight * caption_results.get("emotion_percentages", {}).get(emotion, 0.0)
            if color_weight:
                s += color_weight * color_results.get("emotion_percentages", {}).get(emotion, 0.0)
            integrated_scores[emotion] = s

        # 총합 0 아닌 경우 퍼센트 정규화
        total = sum(integrated_scores.values())
        if total > 0:
            integrated_scores = {e: (v / total) * 100.0 for e, v in integrated_scores.items()}

        sorted_emotions = sorted(integrated_scores.items(), key=lambda x: x[1], reverse=True)
        return {
            "method": "integrated_person_analysis",
            "weights": {"face": round(face_weight, 3),
                        "caption": round(caption_weight, 3),
                        "color": round(color_weight, 3)},
            "emotion_percentages": {e: round(p, 3) for e, p in integrated_scores.items()},
            "top_emotions": [(e, round(p, 3)) for e, p in sorted_emotions[:5]],
            "predicted_emotions": [e for e, p in sorted_emotions if p > 5.0]
        }


    def _integrate_scene_analysis(self, color_results: Dict, caption_results: Optional[Dict]) -> Dict:
        """사람이 없을 때의 감정 분석 결과 통합"""
        # 기본 가중치 설정
        color_weight = 0.6 if color_results and 'error' not in color_results else 0
        caption_weight = 0.4 if caption_results and 'error' not in caption_results else 0
        
        # 가중치 정규화
        total_weight = color_weight + caption_weight
        if total_weight > 0:
            color_weight /= total_weight
            caption_weight /= total_weight
        else:
            # 둘 다 실패한 경우 기본값 반환
            return {
                "method": "integrated_scene_analysis",
                "weights": {"color": 0, "caption": 0},
                "emotion_percentages": {e: 0.0 for e in self.emotions},
                "top_emotions": [],
                "predicted_emotions": [],
                "error": "모든 분석 방법이 실패했습니다."
            }

        # 감정 점수 통합
        integrated_scores = {}
        for emotion in self.emotions:
            score = 0.0
            if color_results and 'error' not in color_results:
                score += color_results.get("emotion_percentages", {}).get(emotion, 0) * color_weight
            if caption_results and 'error' not in caption_results:
                score += caption_results.get("emotion_percentages", {}).get(emotion, 0) * caption_weight
            integrated_scores[emotion] = score

        # 정규화 (총합을 100%로)
        total = sum(integrated_scores.values())
        if total > 0:
            integrated_scores = {e: (s / total) * 100 for e, s in integrated_scores.items()}
        
        # 정렬 및 결과 생성
        sorted_emotions = sorted(integrated_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "method": "integrated_scene_analysis",
            "weights": {"color": color_weight, "caption": caption_weight},
            "emotion_percentages": {e: round(p, 3) for e, p in integrated_scores.items()},
            "top_emotions": [(e, round(p, 3)) for e, p in sorted_emotions[:5]],
            "predicted_emotions": [e for e, p in sorted_emotions if p > 5.0]
        }


# ── CLI: 단일 이미지 전용 ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="조건부 감정 분석 시스템 (단일 이미지 전용)")
    parser.add_argument('--image', required=True, help='이미지 파일 경로')
    parser.add_argument('--yolo_model', default='model/yolo/yolov8s.pt', help='YOLO 모델 경로')
    parser.add_argument('--clip_model', default='model/clip/mlp.pt', help='훈련된 CLIP 모델 경로')
    parser.add_argument('--moondream_model', help='Moondream2 모델 경로')
    parser.add_argument('--face_experiment_path', default='model', help='EMOTIC 실험 경로')
    parser.add_argument('--face_model_dir', default='emotic', help='EMOTIC 모델 디렉토리')
    parser.add_argument('--confidence', type=float, default=0.5, help='YOLO 신뢰도 임계값')
    parser.add_argument('--colors', type=int, default=5, help='추출할 색상 수')
    args = parser.parse_args()

    # 파일 존재 확인
    if not Path(args.image).exists():
        print(f" 이미지 파일을 찾을 수 없습니다: {args.image}")
        return

    system = ConditionalEmotionSystem(
        yolo_model_path=args.yolo_model,
        clip_model_path=args.clip_model,
        moondream_model_path=args.moondream_model,
        face_experiment_path=args.face_experiment_path,
        face_model_dir=args.face_model_dir
    )

    result = system.analyze_emotion(
        args.image,
        confidence_threshold=args.confidence,
        n_colors=args.colors
    )

    print("\n" + "="*80)
    print("조건부 감정 분석 결과")
    print("="*80)
    print(f"이미지: {args.image}")
    print(f"사람 탐지: {'예' if result.get('has_person') else '아니오'} ({result.get('person_count', 0)}명)")
    print(f"분석 유형: {result.get('analysis_type', 'unknown')}")
    print(f"캡션: {result.get('caption', 'N/A')}")

    final_emotions = result.get('final_emotions', {})
    if final_emotions and 'error' not in final_emotions:
        print(f"\n최종 감정 결과 ({final_emotions.get('method', 'unknown')}):")
        print("상위 5가지 감정:")
        for emotion, pct in final_emotions.get('top_emotions', []):
            print(f"  {emotion:15s}: {pct:.3f}%")

        if final_emotions.get('top_emotions'):
            top1_emotion, top1_pct = final_emotions['top_emotions'][0]
            print(f"\n가장 높은 감정: {top1_emotion} ")
        elif final_emotions.get('emotion_percentages'):
            top1_emotion, top1_pct = max(
                final_emotions['emotion_percentages'].items(),
                key=lambda x: x[1]
            )
            print(f"\n가장 높은 감정: {top1_emotion} ({top1_pct:.3f}%)")
    #     print(f"\n예측된 감정 (>5%): {final_emotions.get('predicted_emotions', [])}")
    #     if 'weights' in final_emotions:
    #         print(f"\n통합 가중치: {final_emotions['weights']}")
    # elif final_emotions and 'error' in final_emotions:
    #     print(f"\n 감정 분석 실패: {final_emotions['error']}")
    # elif 'error' in result:
    #     print(f"\n 전체 분석 실패: {result['error']}")

if __name__ == "__main__":
    main()