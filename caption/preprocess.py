import torch
import pandas as pd
import numpy as np
from PIL import Image
import os
import json
from pathlib import Path
import pickle
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class EmotionDataPreprocessor:
    def __init__(self, 
                 model_name: str = "openai/clip-vit-base-patch32",
                 max_length: int = 77,
                 image_size: int = 224):
        """
        CLIP 기반 감정 분류를 위한 전처리기
        
        Args:
            model_name: 사용할 CLIP 모델명
            max_length: 텍스트 최대 길이
            image_size: 이미지 크기
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # CLIP 모델과 프로세서 로드
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # freeze CLIP
        
        self.max_length = max_length
        self.image_size = image_size
        
        # 17가지 감정 라벨 정의
        self.predefined_emotions = [
            "Happiness", "Confidence", "Surprise", "Pain", "Disquietment", 
            "Fear", "Yearning", "Excitement", "Embarrassment", "Affection", 
            "Aversion", "Engagement", "Anticipation", "Sensitivity", 
            "Annoyance", "Sympathy", "Pleasure"
        ]
        
        # 감정 라벨 매핑 (미리 정의된 17개 사용)
        self.emotion_to_id = {emotion: idx for idx, emotion in enumerate(self.predefined_emotions)}
        self.id_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_id.items()}
        self.num_emotions = len(self.predefined_emotions)
        
        print(f"사용할 17가지 감정: {self.predefined_emotions}")
        
    def load_and_analyze_data(self, csv_path: str, image_base_path: str = "") -> pd.DataFrame:
        """
        데이터 로드 및 기본 분석
        
        Args:
            csv_path: CSV 파일 경로
            image_base_path: 이미지 기본 경로
            
        Returns:
            로드된 DataFrame
        """
        print("데이터 로드 중...")
        
        # CSV 로드 (컬럼명이 없다면 수동 설정)
        try:
            df = pd.read_csv(csv_path)
        except:
            df = pd.read_csv(csv_path, header=None, 
                           names=['image_path', 'caption', 'emotions'])
        
        print(f"총 데이터 수: {len(df)}")
        
        # 이미지 경로 처리
        if image_base_path:
            df['full_image_path'] = df['image_path'].apply(
                lambda x: os.path.join(image_base_path, x) if not os.path.isabs(x) else x
            )
        else:
            df['full_image_path'] = df['image_path']
        
        # 경로 변환: 4개 폴더 구조에 맞게 처리
        def convert_image_path(path):
            base_path = 'data/data_person/emotic/emotic/'
            
            # ade20k 경로 변환
            if path.startswith('ade20k/'):
                return path.replace('ade20k/', f'{base_path}ade20k/')
            # mscoco 경로 변환  
            elif path.startswith('mscoco/'):
                return path.replace('mscoco/', f'{base_path}mscoco/')
            # emodb_small 경로 변환
            elif path.startswith('emodb_small/'):
                return path.replace('emodb_small/', f'{base_path}emodb_small/')
            # framesdb 경로 변환
            elif path.startswith('framesdb/'):
                return path.replace('framesdb/', f'{base_path}framesdb/')
            # 기존 ade20k/images/ 패턴 처리
            elif 'ade20k/images/' in path:
                return path.replace('ade20k/images/', f'{base_path}ade20k/images/')
            elif 'mscoco/images/' in path:
                return path.replace('mscoco/images/', f'{base_path}mscoco/images/')
            # 기타 경로
            else:
                if not path.startswith('data/') and not path.startswith('./'):
                    return f'{base_path}{path}'
                else:
                    return path
        
        df['full_image_path'] = df['full_image_path'].apply(convert_image_path)
        
        # 경로 디버깅을 위한 출력
        print(f"샘플 경로들:")
        for i in range(min(5, len(df))):
            print(f"  원본: {df.iloc[i]['image_path']}")
            print(f"  변환: {df.iloc[i]['full_image_path']}")
            print(f"  존재: {os.path.exists(df.iloc[i]['full_image_path'])}")
            print()
        
        # 감정 라벨 파싱
        if isinstance(df['emotions'].iloc[0], str):
            df['emotions_list'] = df['emotions'].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) else []
            )
        else:
            df['emotions_list'] = df['emotions']
        
        # 데이터 유효성 검사
        print("데이터 유효성 검사...")
        valid_images = []
        invalid_count = 0
        
        # 처음 10개만 빠르게 테스트
        test_sample = df.head(10)
        for idx, row in test_sample.iterrows():
            if os.path.exists(row['full_image_path']):
                print(f"✅ 찾음: {row['full_image_path']}")
                break
            else:
                print(f"❌ 없음: {row['full_image_path']}")
        
        # 전체 검사 계속
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="이미지 확인"):
            if os.path.exists(row['full_image_path']):
                try:
                    with Image.open(row['full_image_path']) as img:
                        img.verify()
                    valid_images.append(True)
                except:
                    valid_images.append(False)
                    invalid_count += 1
            else:
                valid_images.append(False)
                invalid_count += 1
        
        df['valid_image'] = valid_images
        print(f"유효하지 않은 이미지: {invalid_count}개")
        
        # 유효한 데이터만 필터링
        df_valid = df[df['valid_image']].copy()
        print(f"유효한 데이터 수: {len(df_valid)}")
        
        # 감정 분석
        self._analyze_emotions(df_valid)
        
        return df_valid
    
    def _analyze_emotions(self, df: pd.DataFrame):
        """감정 라벨 분석 및 매핑 검증"""
        print("감정 라벨 분석...")
        
        # 데이터에서 발견된 모든 감정 수집
        all_emotions_in_data = []
        for emotions_list in df['emotions_list']:
            all_emotions_in_data.extend(emotions_list)
        
        emotion_counts = Counter(all_emotions_in_data)
        unique_emotions_in_data = set(emotion_counts.keys())
        
        print(f"데이터에서 발견된 고유 감정 수: {len(unique_emotions_in_data)}")
        print(f"정의된 감정 수: {self.num_emotions}")
        
        # 정의된 감정과 데이터의 감정 비교
        missing_emotions = unique_emotions_in_data - set(self.predefined_emotions)
        unused_emotions = set(self.predefined_emotions) - unique_emotions_in_data
        
        if missing_emotions:
            print(f"정의되지 않은 감정 발견: {missing_emotions}")
            print("이러한 감정들은 무시됩니다.")
        
        if unused_emotions:
            print(f"데이터에서 사용되지 않은 감정: {unused_emotions}")
        
        # 유효한 감정만으로 카운트 재계산
        valid_emotion_counts = {emotion: count for emotion, count 
                              in emotion_counts.items() 
                              if emotion in self.predefined_emotions}
        
        # 17가지 감정의 분포 시각화
        plt.figure(figsize=(16, 10))
        
        # 1. 실제 데이터 분포
        plt.subplot(2, 2, 1)
        emotions_with_data = []
        counts_with_data = []
        for emotion in self.predefined_emotions:
            count = valid_emotion_counts.get(emotion, 0)
            emotions_with_data.append(emotion)
            counts_with_data.append(count)
        
        plt.bar(range(len(emotions_with_data)), counts_with_data, color='skyblue')
        plt.xticks(range(len(emotions_with_data)), emotions_with_data, rotation=45, ha='right')
        plt.title('17가지 감정 라벨 분포')
        plt.ylabel('빈도')
        plt.grid(axis='y', alpha=0.3)
        
        # 2. 멀티라벨 분포
        plt.subplot(2, 2, 2)
        label_counts = df['emotions_list'].apply(len)
        if len(label_counts) > 0 and label_counts.max() > 0:
            plt.hist(label_counts, bins=range(1, max(label_counts)+2), alpha=0.7, color='lightcoral')
            plt.title('이미지당 감정 라벨 수')
            plt.xlabel('라벨 수')
            plt.ylabel('이미지 수')
            plt.grid(axis='y', alpha=0.3)
        else:
            plt.text(0.5, 0.5, '데이터 없음', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('이미지당 감정 라벨 수 - 데이터 없음')
        
        # 3. 감정 동시 출현 히트맵 (상위 10개 감정)
        plt.subplot(2, 2, 3)
        if valid_emotion_counts:
            top_emotions = [emotion for emotion, count in 
                           sorted(valid_emotion_counts.items(), key=lambda x: x[1], reverse=True)[:10]]
            
            if len(top_emotions) > 0:
                cooccurrence_matrix = np.zeros((len(top_emotions), len(top_emotions)))
                for emotions_list in df['emotions_list']:
                    valid_emotions = [e for e in emotions_list if e in top_emotions]
                    for i, emotion1 in enumerate(top_emotions):
                        for j, emotion2 in enumerate(top_emotions):
                            if emotion1 in valid_emotions and emotion2 in valid_emotions:
                                cooccurrence_matrix[i][j] += 1
                
                if cooccurrence_matrix.sum() > 0:
                    sns.heatmap(cooccurrence_matrix, 
                               xticklabels=[e[:8] for e in top_emotions], 
                               yticklabels=[e[:8] for e in top_emotions],
                               annot=True, fmt='g', cmap='Blues')
                    plt.title('감정 동시 출현 (상위 10개)')
                else:
                    plt.text(0.5, 0.5, '데이터 없음', ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title('감정 동시 출현 - 데이터 없음')
            else:
                plt.text(0.5, 0.5, '감정 없음', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('감정 동시 출현 - 감정 없음')
        else:
            plt.text(0.5, 0.5, '데이터 없음', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('감정 동시 출현 - 데이터 없음')
        
        # 4. 감정별 커버리지
        plt.subplot(2, 2, 4)
        if len(df) > 0:
            coverage = [(count / len(df)) * 100 for count in counts_with_data]
            plt.bar(range(len(emotions_with_data)), coverage, color='lightgreen')
            plt.xticks(range(len(emotions_with_data)), 
                      [e[:8] + '...' if len(e) > 8 else e for e in emotions_with_data], 
                      rotation=45, ha='right')
            plt.title('감정별 데이터 커버리지 (%)')
            plt.ylabel('커버리지 (%)')
            plt.grid(axis='y', alpha=0.3)
        else:
            plt.text(0.5, 0.5, '데이터 없음', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('감정별 데이터 커버리지 - 데이터 없음')
        
        plt.tight_layout()
        plt.savefig('emotion_analysis_17emotions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 통계 출력
        print("\n17가지 감정 통계:")
        for emotion in self.predefined_emotions:
            count = valid_emotion_counts.get(emotion, 0)
            percentage = (count / len(df)) * 100 if len(df) > 0 else 0
            print(f"  {emotion:15s}: {count:4d}개 ({percentage:5.1f}%)")
        
        return valid_emotion_counts
    
    def create_binary_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        멀티라벨을 바이너리 매트릭스로 변환 (17가지 감정 기준)
        
        Args:
            df: DataFrame with emotions_list column
            
        Returns:
            Binary label matrix (N x 17)
        """
        print("바이너리 라벨 매트릭스 생성 (17가지 감정)...")
        
        binary_labels = np.zeros((len(df), self.num_emotions), dtype=np.float32)
        skipped_emotions = set()
        
        for idx, emotions_list in enumerate(df['emotions_list']):
            for emotion in emotions_list:
                if emotion in self.emotion_to_id:
                    emotion_idx = self.emotion_to_id[emotion]
                    binary_labels[idx, emotion_idx] = 1.0
                else:
                    skipped_emotions.add(emotion)
        
        if skipped_emotions:
            print(f"다음 감정들은 17가지 정의된 감정에 포함되지 않아 스킵되었습니다: {skipped_emotions}")
        
        # 라벨 통계
        label_sums = binary_labels.sum(axis=0)
        print(f"라벨 매트릭스 크기: {binary_labels.shape}")
        print(f"평균 라벨 수 per 이미지: {binary_labels.sum(axis=1).mean():.2f}")
        print(f"가장 빈번한 감정: {self.predefined_emotions[np.argmax(label_sums)]} ({int(label_sums.max())}개)")
        print(f"가장 드문 감정: {self.predefined_emotions[np.argmin(label_sums)]} ({int(label_sums.min())}개)")
        
        return binary_labels
    
    def preprocess_images(self, df: pd.DataFrame, batch_size: int = 32) -> torch.Tensor:
        """
        이미지 배치 전처리
        
        Args:
            df: DataFrame with image paths
            batch_size: 배치 크기
            
        Returns:
            전처리된 이미지 텐서
        """
        print("이미지 전처리 중...")
        
        image_features_list = []
        
        for i in tqdm(range(0, len(df), batch_size), desc="이미지 처리"):
            batch_df = df.iloc[i:i+batch_size]
            batch_images = []
            
            for _, row in batch_df.iterrows():
                try:
                    image = Image.open(row['full_image_path']).convert('RGB')
                    batch_images.append(image)
                except Exception as e:
                    print(f"이미지 로드 실패: {row['full_image_path']} - {e}")
                    # 빈 이미지로 대체
                    batch_images.append(Image.new('RGB', (224, 224), color='black'))
            
            # CLIP으로 이미지 처리
            if batch_images:
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                
                with torch.no_grad():
                    pixel_values = inputs['pixel_values'].to(self.device)
                    image_features = self.model.get_image_features(pixel_values)
                    image_features_list.append(image_features.cpu())
        
        # 모든 배치 결합
        all_image_features = torch.cat(image_features_list, dim=0)
        print(f"이미지 특징 크기: {all_image_features.shape}")
        
        return all_image_features
    
    def preprocess_texts(self, df: pd.DataFrame, batch_size: int = 32) -> torch.Tensor:
        """
        텍스트 배치 전처리
        
        Args:
            df: DataFrame with descriptions
            batch_size: 배치 크기
            
        Returns:
            전처리된 텍스트 텐서
        """
        print("텍스트 전처리 중...")
        
        text_features_list = []
        descriptions = df['caption'].tolist()
        
        for i in tqdm(range(0, len(descriptions), batch_size), desc="텍스트 처리"):
            batch_texts = descriptions[i:i+batch_size]
            
            # CLIP으로 텍스트 처리
            inputs = self.processor(text=batch_texts, return_tensors="pt", 
                                  padding=True, truncation=True, max_length=self.max_length)
            
            with torch.no_grad():
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                text_features = self.model.get_text_features(input_ids=input_ids, 
                                                           attention_mask=attention_mask)
                text_features_list.append(text_features.cpu())
        
        # 모든 배치 결합
        all_text_features = torch.cat(text_features_list, dim=0)
        print(f"텍스트 특징 크기: {all_text_features.shape}")
        
        return all_text_features
    
    def create_combined_features(self, 
                               image_features: torch.Tensor, 
                               text_features: torch.Tensor,
                               combination_method: str = 'concat') -> torch.Tensor:
        """
        이미지와 텍스트 특징 결합
        
        Args:
            image_features: 이미지 특징
            text_features: 텍스트 특징
            combination_method: 결합 방식 ('concat', 'add', 'multiply')
            
        Returns:
            결합된 특징
        """
        print(f"특징 결합 중 (방식: {combination_method})...")
        
        if combination_method == 'concat':
            combined_features = torch.cat([image_features, text_features], dim=1)
        elif combination_method == 'add':
            combined_features = image_features + text_features
        elif combination_method == 'multiply':
            combined_features = image_features * text_features
        else:
            raise ValueError(f"지원하지 않는 결합 방식: {combination_method}")
        
        print(f"결합된 특징 크기: {combined_features.shape}")
        return combined_features
    
    def process_single_dataset(self, 
                             csv_path: str, 
                             image_base_path: str = "",
                             dataset_name: str = "dataset") -> Dict:
        """
        단일 데이터셋 전처리 (이미 분할된 데이터용)
        
        Args:
            csv_path: CSV 파일 경로
            image_base_path: 이미지 기본 경로  
            dataset_name: 데이터셋 이름 (train/val/test)
            
        Returns:
            처리된 데이터 딕셔너리
        """
        print(f"{dataset_name.upper()} 데이터셋 처리 중...")
        
        # 1. 데이터 로드
        df = self.load_and_analyze_data(csv_path, image_base_path)
        print(f"{dataset_name} 데이터 수: {len(df):,}개")
        
        # 2. 바이너리 라벨 생성
        binary_labels = self.create_binary_labels(df)
        
        # 3. 이미지 전처리
        image_features = self.preprocess_images(df, batch_size=32)
        
        # 4. 텍스트 전처리
        text_features = self.preprocess_texts(df, batch_size=32)
        
        # 5. 특징 결합
        combined_features = self.create_combined_features(
            image_features, text_features, 'concat'
        )
        
        processed_data = {
            'features': combined_features,
            'labels': binary_labels,
            'metadata': df
        }
        
        print(f"{dataset_name} 처리 완료: 특징 {combined_features.shape}, 라벨 {binary_labels.shape}")
        return processed_data
    
    def save_preprocessed_data(self, 
                             processed_data: Dict,
                             output_dir: str,
                             dataset_name: str):
        """
        전처리된 단일 데이터셋 저장
        
        Args:
            processed_data: 처리된 데이터
            output_dir: 출력 디렉토리
            dataset_name: 데이터셋 이름
        """
        print(f"{dataset_name} 데이터셋 저장 중...")
        
        dataset_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # 특징과 라벨을 .pt 파일로 저장
        torch.save(processed_data['features'], 
                  os.path.join(dataset_dir, 'features.pt'))
        torch.save(torch.from_numpy(processed_data['labels']), 
                  os.path.join(dataset_dir, 'labels.pt'))
        
        # 메타데이터를 CSV로 저장
        processed_data['metadata'].to_csv(
            os.path.join(dataset_dir, 'metadata.csv'), index=False
        )
        
        print(f"{dataset_name} 데이터가 {dataset_dir}에 저장되었습니다.")
    
    def save_metadata(self, output_dir: str):
        """
        전역 메타데이터 저장
        
        Args:
            output_dir: 출력 디렉토리
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 감정 매핑 및 전역 설정 저장
        metadata = {
            'emotion_to_id': self.emotion_to_id,
            'id_to_emotion': self.id_to_emotion,
            'num_emotions': self.num_emotions,
            'predefined_emotions': self.predefined_emotions,
            'clip_model': 'openai/clip-vit-base-patch32',
            'feature_dim_image': 512,
            'feature_dim_text': 512, 
            'feature_dim_combined': 1024
        }
        
        with open(os.path.join(output_dir, 'global_metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"전역 메타데이터가 {output_dir}/global_metadata.json에 저장되었습니다.")

# 이미 분할된 데이터셋 전처리 파이프라인
def process_presplit_datasets(train_csv: str,
                            val_csv: str, 
                            test_csv: str,
                            image_base_path: str = "",
                            output_dir: str = "preprocessed_data",
                            batch_size: int = 32):
    """
    이미 분할된 데이터셋들을 각각 전처리 (권장)
    
    Args:
        train_csv: 훈련 데이터 CSV 경로
        val_csv: 검증 데이터 CSV 경로  
        test_csv: 테스트 데이터 CSV 경로
        image_base_path: 이미지 기본 경로
        output_dir: 출력 디렉토리
        batch_size: 배치 크기
    """
    print("이미 분할된 데이터셋 전처리 시작!")
    print("17가지 감정: Happiness, Confidence, Surprise, Pain, Disquietment,")
    print("            Fear, Yearning, Excitement, Embarrassment, Affection,")
    print("            Aversion, Engagement, Anticipation, Sensitivity,") 
    print("            Annoyance, Sympathy, Pleasure")
    print("-" * 80)
    
    # 전처리기 초기화 (한번만)
    preprocessor = EmotionDataPreprocessor()
    
    # 각 데이터셋별로 처리
    all_datasets = {}
    emotion_counts_total = Counter()
    
    datasets = [
        (train_csv, "train"),
        (val_csv, "val"), 
        (test_csv, "test")
    ]
    
    for csv_path, dataset_name in datasets:
        if csv_path and os.path.exists(csv_path):
            print(f"\n{dataset_name.upper()} 데이터셋 처리 중...")
            
            # 단일 데이터셋 처리
            processed_data = preprocessor.process_single_dataset(
                csv_path, image_base_path, dataset_name
            )
            
            # 저장
            preprocessor.save_preprocessed_data(processed_data, output_dir, dataset_name)
            
            # 통계 수집
            all_datasets[dataset_name] = processed_data
            
            # 감정 통계 합계
            for emotions_list in processed_data['metadata']['emotions_list']:
                for emotion in emotions_list:
                    if emotion in preprocessor.predefined_emotions:
                        emotion_counts_total[emotion] += 1
        else:
            print(f"{dataset_name} 파일을 찾을 수 없습니다: {csv_path}")
    
    # 전역 메타데이터 저장
    preprocessor.save_metadata(output_dir)
    
    # 전체 통계 리포트 생성
    generate_final_report(all_datasets, emotion_counts_total, preprocessor, output_dir)
    
    print(f"\n전처리 완료!")
    print(f"데이터 저장 위치: {output_dir}")
    print(f"처리된 데이터셋: {list(all_datasets.keys())}")
    print(f"다음 단계: 1024->17 차원 MLP 학습 준비 완료")
    
    return preprocessor, all_datasets

def generate_final_report(all_datasets: Dict,
                        emotion_counts_total: Counter,
                        preprocessor: EmotionDataPreprocessor,
                        output_dir: str):
    """
    전체 전처리 결과 리포트 생성
    
    Args:
        all_datasets: 모든 처리된 데이터셋
        emotion_counts_total: 전체 감정 통계
        preprocessor: 전처리기 인스턴스
        output_dir: 출력 디렉토리
    """
    report_path = os.path.join(output_dir, "preprocessing_final_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CLIP 감정 분류 전처리 최종 리포트\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"전체 데이터 개요\n")
        total_samples = sum(len(data['features']) for data in all_datasets.values())
        f.write(f"- 총 데이터 수: {total_samples:,}개\n")
        f.write(f"- 감정 수: 17개 (정의된 감정)\n")
        f.write(f"- 특징 차원: 1024 (이미지 512 + 텍스트 512)\n\n")
        
        f.write(f"데이터셋별 분포\n")
        for dataset_name, data in all_datasets.items():
            count = len(data['features'])
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            avg_labels = data['labels'].sum(axis=1).mean()
            f.write(f"- {dataset_name.upper():5s}: {count:6,}개 ({percentage:5.1f}%) | 평균 라벨: {avg_labels:.2f}\n")
        
        f.write(f"\n17가지 감정 전체 분포\n")
        f.write("-" * 50 + "\n")
        for emotion in preprocessor.predefined_emotions:
            count = emotion_counts_total.get(emotion, 0)
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            coverage = "=" * max(1, int(percentage / 3)) if percentage > 0 else " "
            f.write(f"{emotion:15s}: {count:5,}개 ({percentage:5.1f}%) {coverage}\n")
        
        f.write(f"\n처리 환경 및 설정\n")
        f.write(f"- Device: {preprocessor.device}\n") 
        f.write(f"- CLIP 모델: openai/clip-vit-base-patch32\n")
        f.write(f"- 이미지 크기: {preprocessor.image_size}x{preprocessor.image_size}\n")
        f.write(f"- 텍스트 최대 길이: {preprocessor.max_length} 토큰\n")
        f.write(f"- 특징 결합: concat (이미지+텍스트)\n\n")
        
        f.write(f"생성된 파일들\n")
        for dataset_name in all_datasets.keys():
            f.write(f"- {dataset_name}/\n")
            f.write(f"  * features.pt (특징 벡터)\n")
            f.write(f"  * labels.pt (17차원 바이너리 라벨)\n") 
            f.write(f"  * metadata.csv (원본 정보)\n")
        f.write(f"- global_metadata.json (전역 설정)\n\n")
        
        f.write(f"다음 단계 (4. 모델 구성)\n")
        f.write(f"- 입력: torch.load('train/features.pt')  # shape: (N, 1024)\n")
        f.write(f"- 라벨: torch.load('train/labels.pt')    # shape: (N, 17)\n")
        f.write(f"- 모델: MLP(1024 -> 512 -> 256 -> 17)\n")
        f.write(f"- 손실: BCEWithLogitsLoss (멀티라벨)\n")
        f.write(f"- 메트릭: F1-macro, F1-micro, AUC-ROC\n")
    
    print(f"최종 리포트: {report_path}")

# 사용 예제 (이미 분할된 데이터용)  
if __name__ == "__main__":
    # 이미 분할된 데이터셋 전처리
    preprocessor, datasets = process_presplit_datasets(
        train_csv="./data/data_person/emotic/dataset_min_train.csv",
        val_csv="./data/data_person/emotic/dataset_min_val.csv",
        test_csv="./data/data_person/emotic/dataset_min_test.csv",
        image_base_path="./",  # 현재 디렉토리 기준
        output_dir="clip_preprocessed_data",
        batch_size=16  # GPU 메모리에 따라 조정
    )
    
    print("\n다음 단계에서 사용할 데이터:")
    for name, data in datasets.items():
        print(f"  {name}: 특징 {data['features'].shape}, 라벨 {data['labels'].shape}")