import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import colorsys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

class ColorEmotionInference:
    def __init__(self):
        """
        색상 기반 감정 추론기
        색채 심리학 이론을 기반으로 한 색상-감정 매핑
        """
        # 17가지 감정 정의 (CLIP 모델과 동일)
        self.emotions = [
            "Happiness", "Confidence", "Surprise", "Pain", "Disquietment", 
            "Fear", "Yearning", "Excitement", "Embarrassment", "Affection", 
            "Aversion", "Engagement", "Anticipation", "Sensitivity", 
            "Annoyance", "Sympathy", "Pleasure"
        ]
        
        # 색상-감정 매핑 (퍼센트 기반)
        self.color_emotion_mapping = self._initialize_color_emotion_mapping()
        
    def _initialize_color_emotion_mapping(self) -> Dict:
        """
        색상과 감정 간의 매핑 정의 (퍼센트 기반) - 더 구체적이고 세밀한 분류
        HSV 색상 공간 기반으로 정의
        """
        return {
            # === 빨간색 계열 ===
            # 순수 빨간색 (Hue: 0-15, 345-360)
            "pure_red": {
                "emotions": {
                    "Excitement": 0.850,      # 85.0%
                    "Confidence": 0.780,      # 78.0%
                    "Annoyance": 0.650,       # 65.0%
                    "Pain": 0.520,            # 52.0%
                    "Engagement": 0.480       # 48.0%
                },
                "hue_range": [(0, 15), (345, 360)],
                "saturation_min": 0.6,
                "value_min": 0.4
            },
            # 어두운 빨간색 (마룬, 버건디)
            "dark_red": {
                "emotions": {
                    "Pain": 0.720,
                    "Annoyance": 0.680,
                    "Aversion": 0.620,
                    "Disquietment": 0.580,
                    "Fear": 0.450
                },
                "hue_range": [(0, 15), (345, 360)],
                "saturation_min": 0.4,
                "value_max": 0.4
            },
            # 연한 빨간색 (핑크 레드)
            "light_red": {
                "emotions": {
                    "Affection": 0.680,
                    "Excitement": 0.620,
                    "Happiness": 0.580,
                    "Embarrassment": 0.540,
                    "Pleasure": 0.480
                },
                "hue_range": [(0, 15), (345, 360)],
                "saturation_range": (0.3, 0.6),
                "value_min": 0.7
            },

            # === 주황색 계열 ===
            # 순수 주황색 (Hue: 15-35)
            "pure_orange": {
                "emotions": {
                    "Happiness": 0.880,       # 88.0%
                    "Excitement": 0.820,      # 82.0%
                    "Confidence": 0.750,      # 75.0%
                    "Pleasure": 0.680,        # 68.0%
                    "Anticipation": 0.590     # 59.0%
                },
                "hue_range": [(15, 35)],
                "saturation_min": 0.5,
                "value_min": 0.5
            },
            # 어두운 주황색 (브라운 오렌지)
            "dark_orange": {
                "emotions": {
                    "Confidence": 0.680,
                    "Engagement": 0.620,
                    "Sympathy": 0.580,
                    "Sensitivity": 0.520,
                    "Annoyance": 0.450
                },
                "hue_range": [(15, 35)],
                "saturation_min": 0.3,
                "value_max": 0.5
            },
            # 연한 주황색 (피치)
            "light_orange": {
                "emotions": {
                    "Happiness": 0.780,
                    "Pleasure": 0.720,
                    "Affection": 0.650,
                    "Sympathy": 0.580,
                    "Sensitivity": 0.520
                },
                "hue_range": [(15, 35)],
                "saturation_range": (0.2, 0.5),
                "value_min": 0.7
            },

            # === 노란색 계열 ===
            # 순수 노란색 (Hue: 35-65)
            "pure_yellow": {
                "emotions": {
                    "Happiness": 0.920,       # 92.0%
                    "Surprise": 0.780,        # 78.0%
                    "Excitement": 0.750,      # 75.0%
                    "Anticipation": 0.680,    # 68.0%
                    "Confidence": 0.620       # 62.0%
                },
                "hue_range": [(35, 65)],
                "saturation_min": 0.6,
                "value_min": 0.7
            },
            # 어두운 노란색 (올리브, 머스타드)
            "dark_yellow": {
                "emotions": {
                    "Disquietment": 0.620,
                    "Aversion": 0.580,
                    "Annoyance": 0.540,
                    "Sensitivity": 0.500,
                    "Pain": 0.450
                },
                "hue_range": [(35, 65)],
                "saturation_min": 0.3,
                "value_max": 0.6
            },
            # 연한 노란색 (크림, 아이보리)
            "light_yellow": {
                "emotions": {
                    "Happiness": 0.720,
                    "Pleasure": 0.680,
                    "Sympathy": 0.620,
                    "Affection": 0.580,
                    "Sensitivity": 0.520
                },
                "hue_range": [(35, 65)],
                "saturation_range": (0.1, 0.4),
                "value_min": 0.8
            },

            # === 초록색 계열 ===
            # 순수 초록색 (Hue: 65-140)
            "pure_green": {
                "emotions": {
                    "Pleasure": 0.820,        # 82.0%
                    "Sympathy": 0.780,        # 78.0%
                    "Affection": 0.720,       # 72.0%
                    "Engagement": 0.680,      # 68.0%
                    "Sensitivity": 0.580      # 58.0%
                },
                "hue_range": [(65, 140)],
                "saturation_min": 0.4,
                "value_min": 0.3
            },
            # 어두운 초록색 (포레스트 그린)
            "dark_green": {
                "emotions": {
                    "Engagement": 0.680,
                    "Sympathy": 0.620,
                    "Sensitivity": 0.580,
                    "Disquietment": 0.520,
                    "Aversion": 0.480
                },
                "hue_range": [(65, 140)],
                "saturation_min": 0.3,
                "value_max": 0.4
            },
            # 연한 초록색 (민트, 라이트 그린)
            "light_green": {
                "emotions": {
                    "Pleasure": 0.750,
                    "Sympathy": 0.690,
                    "Affection": 0.640,
                    "Sensitivity": 0.590,
                    "Happiness": 0.520
                },
                "hue_range": [(65, 140)],
                "saturation_range": (0.2, 0.5),
                "value_min": 0.6
            },

            # === 파란색 계열 ===
            # 순수 파란색 (Hue: 140-240) - 하늘, 바다
            "pure_blue": {
                "emotions": {
                    "Happiness": 0.720,       # 72.0% (추가)
                    "Pleasure": 0.680,        # 68.0% (추가)
                    "Confidence": 0.620,      # 62.0% (추가)
                    "Engagement": 0.580,      # 58.0% (유지)
                    "Sensitivity": 0.480      # 48.0% (대폭 낮춤)
                },
                "hue_range": [(140, 240)],
                "saturation_min": 0.5,
                "value_min": 0.4
            },
            # 어두운 파란색 (네이비, 다크 블루) - 깊은 바다, 밤하늘
            "dark_blue": {
                "emotions": {
                    "Confidence": 0.680,      # 68.0% (추가)
                    "Engagement": 0.620,      # 62.0% (추가)
                    "Yearning": 0.580,        # 58.0% (낮춤)
                    "Sensitivity": 0.520,     # 52.0% (낮춤)
                    "Disquietment": 0.480     # 48.0% (낮춤)
                },
                "hue_range": [(140, 240)],
                "saturation_min": 0.3,
                "value_max": 0.4
            },
            # 연한 파란색 (스카이 블루, 라이트 블루) - 맑은 하늘
            "light_blue": {
                "emotions": {
                    "Happiness": 0.820,       # 82.0% (높임)
                    "Pleasure": 0.780,        # 78.0% (높임)
                    "Confidence": 0.720,      # 72.0% (높임)
                    "Affection": 0.650,       # 65.0% (유지)
                    "Sensitivity": 0.420      # 42.0% (대폭 낮춤)
                },
                "hue_range": [(140, 240)],
                "saturation_range": (0.2, 0.6),
                "value_min": 0.6
            },

            # === 보라색 계열 ===
            # 순수 보라색 (Hue: 240-300)
            "pure_purple": {
                "emotions": {
                    "Yearning": 0.820,        # 82.0%
                    "Sensitivity": 0.750,     # 75.0%
                    "Embarrassment": 0.680,   # 68.0%
                    "Surprise": 0.620,        # 62.0%
                    "Anticipation": 0.540     # 54.0%
                },
                "hue_range": [(240, 300)],
                "saturation_min": 0.5,
                "value_min": 0.4
            },
            # 어두운 보라색
            "dark_purple": {
                "emotions": {
                    "Fear": 0.680,
                    "Disquietment": 0.640,
                    "Yearning": 0.600,
                    "Pain": 0.560,
                    "Aversion": 0.520
                },
                "hue_range": [(240, 300)],
                "saturation_min": 0.3,
                "value_max": 0.4
            },
            # 연한 보라색 (라벤더)
            "light_purple": {
                "emotions": {
                    "Sensitivity": 0.720,
                    "Affection": 0.680,
                    "Sympathy": 0.640,
                    "Pleasure": 0.580,
                    "Embarrassment": 0.520
                },
                "hue_range": [(240, 300)],
                "saturation_range": (0.2, 0.5),
                "value_min": 0.6
            },

            # === 분홍색 계열 ===
            # 순수 분홍색 (Hue: 300-340)
            "pure_pink": {
                "emotions": {
                    "Affection": 0.880,       # 88.0%
                    "Happiness": 0.780,       # 78.0%
                    "Embarrassment": 0.680,   # 68.0%
                    "Pleasure": 0.640,        # 64.0%
                    "Sympathy": 0.580         # 58.0%
                },
                "hue_range": [(300, 340)],
                "saturation_min": 0.4,
                "value_min": 0.5
            },
            # 어두운 분홍색 (로즈)
            "dark_pink": {
                "emotions": {
                    "Affection": 0.720,
                    "Sympathy": 0.680,
                    "Sensitivity": 0.620,
                    "Embarrassment": 0.580,
                    "Yearning": 0.520
                },
                "hue_range": [(300, 340)],
                "saturation_min": 0.3,
                "value_max": 0.6
            },
            # 연한 분홍색 (베이비 핑크)
            "light_pink": {
                "emotions": {
                    "Affection": 0.820,
                    "Happiness": 0.750,
                    "Pleasure": 0.680,
                    "Sympathy": 0.620,
                    "Sensitivity": 0.580
                },
                "hue_range": [(300, 340)],
                "saturation_range": (0.1, 0.4),
                "value_min": 0.7
            },

            # === 갈색 계열 ===
            # 갈색 (Hue: 10-35, 낮은 채도)
            "brown": {
                "emotions": {
                    "Sympathy": 0.680,
                    "Sensitivity": 0.640,
                    "Engagement": 0.580,
                    "Disquietment": 0.520,
                    "Aversion": 0.480
                },
                "hue_range": [(10, 35)],
                "saturation_range": (0.2, 0.6),
                "value_range": (0.2, 0.7)
            },

            # === 무채색 계열 ===
            # 순수 흰색 (매우 밝고 무채색)
            "white": {
                "emotions": {
                    "Happiness": 0.750,
                    "Surprise": 0.680,
                    "Pleasure": 0.620,
                    "Confidence": 0.580,
                    "Sensitivity": 0.520
                },
                "hue_range": [(0, 360)],  # 모든 색조
                "saturation_max": 0.1,
                "value_min": 0.9
            },
            # 밝은 회색/크림 (매우 밝지만 약간 색조 있음)
            "cream": {
                "emotions": {
                    "Sympathy": 0.720,
                    "Sensitivity": 0.680,
                    "Affection": 0.620,
                    "Pleasure": 0.580,
                    "Happiness": 0.520
                },
                "hue_range": [(0, 360)],
                "saturation_range": (0.05, 0.25),
                "value_min": 0.85
            },
            # 중간 회색
            "gray": {
                "emotions": {
                    "Disquietment": 0.720,    # 72.0%
                    "Pain": 0.650,            # 65.0%
                    "Aversion": 0.580,        # 58.0%
                    "Sensitivity": 0.520,     # 52.0%
                    "Sympathy": 0.440         # 44.0%
                },
                "hue_range": [(0, 360)],  # 모든 색조
                "saturation_max": 0.2,
                "value_range": (0.3, 0.85)
            },
            # 어두운 색상 (검은색에 가까움)
            "black": {
                "emotions": {
                    "Fear": 0.850,            # 85.0%
                    "Disquietment": 0.780,    # 78.0%
                    "Pain": 0.720,            # 72.0%
                    "Aversion": 0.680,        # 68.0%
                    "Annoyance": 0.580        # 58.0%
                },
                "hue_range": [(0, 360)],
                "saturation_max": 0.3,
                "value_max": 0.3
            }
        }
    
    def extract_dominant_colors(self, image_path: str, n_colors: int = 5, 
                              resize_width: int = 150) -> Tuple[np.ndarray, np.ndarray]:
        """
        K-means를 사용하여 이미지에서 주요 색상 추출
        
        Args:
            image_path: 이미지 파일 경로
            n_colors: 추출할 색상 수
            resize_width: 처리 속도를 위한 리사이즈 너비
            
        Returns:
            colors: 주요 색상들 (RGB)
            percentages: 각 색상의 비율
        """
        # 이미지 로드 및 전처리
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 리사이즈 (처리 속도 향상)
        original_shape = image.shape
        aspect_ratio = original_shape[1] / original_shape[0]
        new_height = int(resize_width / aspect_ratio)
        image = cv2.resize(image, (resize_width, new_height))
        
        # 픽셀 데이터를 1D 배열로 변환
        pixels = image.reshape(-1, 3)
        
        # K-means 클러스터링
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # 주요 색상과 비율 계산
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        # 각 클러스터의 비율 계산
        unique_labels, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels)
        
        # 비율 순으로 정렬
        sorted_indices = np.argsort(percentages)[::-1]
        colors = colors[sorted_indices]
        percentages = percentages[sorted_indices]
        
        return colors, percentages
    
    def rgb_to_hsv(self, rgb: np.ndarray) -> Tuple[float, float, float]:
        """RGB를 HSV로 변환"""
        r, g, b = rgb / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return h * 360, s, v  # H를 0-360도로 변환
    
    def classify_color(self, rgb: np.ndarray) -> List[str]:
        """
        RGB 색상을 색상 카테고리로 분류
        
        Args:
            rgb: RGB 색상 값
            
        Returns:
            매칭되는 색상 카테고리 리스트
        """
        h, s, v = self.rgb_to_hsv(rgb)
        matched_categories = []
        
        for category, config in self.color_emotion_mapping.items():
            # 색조(Hue) 체크
            hue_match = False
            for hue_min, hue_max in config["hue_range"]:
                if hue_min <= h <= hue_max:
                    hue_match = True
                    break
            
            if not hue_match:
                continue
            
            # 채도(Saturation) 체크
            if "saturation_min" in config and s < config["saturation_min"]:
                continue
            if "saturation_max" in config and s > config["saturation_max"]:
                continue
            
            # 명도(Value) 체크
            if "value_min" in config and v < config["value_min"]:
                continue
            if "value_max" in config and v > config["value_max"]:
                continue
            if "value_range" in config:
                v_min, v_max = config["value_range"]
                if not (v_min <= v <= v_max):
                    continue
            
            matched_categories.append(category)
        
        return matched_categories
    
    def predict_emotions(self, image_path: str, n_colors: int = 5) -> Dict:
        """
        이미지에서 색상을 추출하고 감정을 예측
        
        Args:
            image_path: 이미지 파일 경로
            n_colors: 추출할 색상 수
            
        Returns:
            감정 예측 결과
        """
        # 주요 색상 추출
        colors, percentages = self.extract_dominant_colors(image_path, n_colors)
        
        # 감정 점수 초기화
        emotion_scores = {emotion: 0.0 for emotion in self.emotions}
        
        color_analysis = []
        
        # 각 색상에 대해 감정 점수 계산
        for i, (color, percentage) in enumerate(zip(colors, percentages)):
            h, s, v = self.rgb_to_hsv(color)
            categories = self.classify_color(color)
            
            color_info = {
                "rank": i + 1,
                "rgb": color.tolist(),
                "hsv": [round(h, 1), round(s, 3), round(v, 3)],
                "percentage": round(percentage * 100, 2),
                "categories": categories,
                "emotions": {}
            }
            
            # 각 카테고리에서 감정 점수 추가
            for category in categories:
                config = self.color_emotion_mapping[category]
                for emotion, emotion_percentage in config["emotions"].items():
                    # 색상 비율 × 감정 퍼센트 × 가중치로 점수 계산
                    score = percentage * (emotion_percentage / 100.0)
                    emotion_scores[emotion] += score
                    
                    if emotion not in color_info["emotions"]:
                        color_info["emotions"][emotion] = 0
                    color_info["emotions"][emotion] += score
            
            color_analysis.append(color_info)
        
        # 감정 점수를 퍼센트로 정규화 (총합 100%)
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            percentage_scores = {emotion: (score / total_score) * 100 
                               for emotion, score in emotion_scores.items()}
        else:
            percentage_scores = {emotion: 0.0 for emotion in self.emotions}
        
        # 결과 정리
        sorted_emotions = sorted(percentage_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "image_path": str(image_path),
            "dominant_colors": color_analysis,
            "emotion_percentages": {emotion: round(pct, 3) for emotion, pct in percentage_scores.items()},
            "top_emotions": [(emotion, round(pct, 3)) for emotion, pct in sorted_emotions[:5]],
            "predicted_emotions": [emotion for emotion, pct in sorted_emotions if pct > 5.0]  # 5% 이상
        }
    
    def visualize_color_analysis(self, image_path: str, result: Dict, save_path: str = None):
        """
        색상 분석 결과 시각화
        
        Args:
            image_path: 원본 이미지 경로
            result: predict_emotions 결과
            save_path: 저장 경로
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 원본 이미지
        original_image = Image.open(image_path)
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 2. 주요 색상 팔레트
        colors = [info["rgb"] for info in result["dominant_colors"]]
        percentages = [info["percentage"] for info in result["dominant_colors"]]
        
        # 색상 팔레트 생성
        palette = np.array(colors).reshape(1, -1, 3).astype(np.uint8)
        axes[0, 1].imshow(palette)
        axes[0, 1].set_title('Dominant Colors')
        axes[0, 1].axis('off')
        
        # 색상 정보 텍스트 추가
        for i, (color, pct) in enumerate(zip(colors, percentages)):
            axes[0, 1].text(i, 0.5, f'{pct:.1f}%', ha='center', va='center', 
                           fontsize=10, fontweight='bold')
        
        # 3. 감정 점수 막대 그래프
        emotions = list(result["emotion_percentages"].keys())
        percentages = list(result["emotion_percentages"].values())
        
        bars = axes[1, 0].barh(emotions, percentages, color='skyblue')
        axes[1, 0].set_xlabel('Emotion Percentage (%)')
        axes[1, 0].set_title('Emotion Predictions')
        axes[1, 0].set_xlim(0, max(percentages) * 1.1 if max(percentages) > 0 else 100)
        
        # 퍼센트 텍스트 추가
        for bar, pct in zip(bars, percentages):
            if pct > 1.0:  # 1% 이상만 표시
                axes[1, 0].text(bar.get_width() + max(percentages) * 0.01, 
                               bar.get_y() + bar.get_height()/2, 
                               f'{pct:.1f}%', va='center', fontsize=8)
        
        # 4. 상위 감정들
        top_emotions = result["top_emotions"]
        if len(top_emotions) > 0:
            top_names = [item[0] for item in top_emotions]
            top_percentages = [item[1] for item in top_emotions]
            
            bars = axes[1, 1].bar(range(len(top_names)), top_percentages, 
                                color=['gold', 'silver', 'chocolate', 'lightblue', 'lightgreen'])
            axes[1, 1].set_xticks(range(len(top_names)))
            axes[1, 1].set_xticklabels(top_names, rotation=45, ha='right')
            axes[1, 1].set_ylabel('Percentage (%)')
            axes[1, 1].set_title('Top 5 Emotions')
            
            # 퍼센트 텍스트 추가
            for bar, pct in zip(bars, top_percentages):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                               f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"시각화 결과 저장: {save_path}")
        
        plt.show()
    
    def batch_process(self, folder_path: str, output_csv: str = "color_emotion_results.csv"):
        """
        폴더 내 모든 이미지에 대해 배치 처리
        
        Args:
            folder_path: 이미지 폴더 경로
            output_csv: 결과 CSV 파일 경로
        """
        folder = Path(folder_path)
        results = []
        
        # 지원하는 이미지 확장자
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # 모든 이미지 파일 찾기
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder.rglob(f"*{ext}"))
            image_files.extend(folder.rglob(f"*{ext.upper()}"))
        
        print(f"처리할 이미지: {len(image_files)}개")
        
        for img_path in image_files:
            try:
                print(f"처리 중: {img_path.name}")
                result = self.predict_emotions(str(img_path))
                
                # CSV용 평면 데이터 생성
                row = {
                    "image_path": str(img_path),
                    "top_emotion": result["top_emotions"][0][0] if result["top_emotions"] else "",
                    "top_percentage": result["top_emotions"][0][1] if result["top_emotions"] else 0,
                    "predicted_emotions": ",".join(result["predicted_emotions"])
                }
                
                # 모든 감정 퍼센트 추가
                for emotion, percentage in result["emotion_percentages"].items():
                    row[f"pct_{emotion}"] = percentage
                
                # 주요 색상 정보 추가
                for i, color_info in enumerate(result["dominant_colors"][:3]):  # 상위 3개 색상만
                    row[f"color_{i+1}_rgb"] = str(color_info["rgb"])
                    row[f"color_{i+1}_percentage"] = color_info["percentage"]
                
                results.append(row)
                
            except Exception as e:
                print(f"처리 실패: {img_path} - {e}")
        
        # 결과를 DataFrame으로 변환하고 저장
        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"\n결과 저장 완료: {output_csv}")
            print(f"처리된 이미지: {len(results)}개")
        else:
            print("처리된 이미지가 없습니다.")
        
        return results

# CLI 인터페이스
def main():
    parser = argparse.ArgumentParser(description="Color-based Emotion Inference")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # single 명령어 - 단일 이미지 처리
    single_parser = subparsers.add_parser('single', help='Process single image')
    single_parser.add_argument('--image', required=True, help='Image file path')
    single_parser.add_argument('--visualize', action='store_true', help='Show visualization')
    single_parser.add_argument('--save_viz', help='Save visualization to file')
    single_parser.add_argument('--colors', type=int, default=5, help='Number of colors to extract')
    
    # batch 명령어 - 배치 처리
    batch_parser = subparsers.add_parser('batch', help='Process folder of images')
    batch_parser.add_argument('--folder', required=True, help='Folder containing images')
    batch_parser.add_argument('--output', default='color_emotion_results.csv', help='Output CSV file')
    batch_parser.add_argument('--colors', type=int, default=5, help='Number of colors to extract')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 감정 추론기 초기화
    predictor = ColorEmotionInference()
    
    if args.command == 'single':
        # 단일 이미지 처리
        result = predictor.predict_emotions(args.image, n_colors=args.colors)
        
        print("\n" + "="*60)
        print("COLOR-BASED EMOTION INFERENCE RESULTS")
        print("="*60)
        print(f"Image: {args.image}")
        print(f"Colors extracted: {len(result['dominant_colors'])}")
        
        print("\nDominant Colors:")
        for color_info in result['dominant_colors']:
            rgb = color_info['rgb']
            pct = color_info['percentage']
            categories = color_info['categories']
            print(f"  #{color_info['rank']}: RGB{rgb} ({pct:.1f}%) - {categories}")
        
        print(f"\nTop 5 Emotions:")
        for emotion, percentage in result['top_emotions']:
            print(f"  {emotion:15s}: {percentage:.3f}%")
        
        print(f"\nPredicted Emotions (>5%):")
        print(f"  {result['predicted_emotions']}")
        
        # 시각화
        if args.visualize or args.save_viz:
            predictor.visualize_color_analysis(args.image, result, args.save_viz)
    
    elif args.command == 'batch':
        # 배치 처리
        results = predictor.batch_process(args.folder, args.output)
        
        if results:
            print(f"\n배치 처리 완료!")
            print(f"총 {len(results)}개 이미지 처리")
            
            # 통계 출력
            all_top_emotions = [r['top_emotion'] for r in results if r['top_emotion']]
            from collections import Counter
            emotion_counts = Counter(all_top_emotions)
            
            print(f"\n가장 많이 예측된 감정:")
            for emotion, count in emotion_counts.most_common(5):
                percentage = (count / len(results)) * 100
                print(f"  {emotion:15s}: {count:3d}개 ({percentage:5.1f}%)")

if __name__ == "__main__":
    main()