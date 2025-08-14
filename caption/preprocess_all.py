import torch
import pandas as pd
import numpy as np
from PIL import Image
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast
from transformers import CLIPProcessor, CLIPModel
import warnings
warnings.filterwarnings('ignore')

class UnifiedEmotionPreprocessor:
    def __init__(self, 
                 model_name: str = "openai/clip-vit-base-patch32",
                 max_length: int = 77,
                 image_size: int = 224):
        """
        인물 + 자연 사진 통합 감정 분류 전처리기
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # CLIP 모델과 프로세서 로드
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.max_length = max_length
        self.image_size = image_size
        
        # 17가지 감정 라벨 정의
        self.predefined_emotions = [
            "Happiness", "Confidence", "Surprise", "Pain", "Disquietment", 
            "Fear", "Yearning", "Excitement", "Embarrassment", "Affection", 
            "Aversion", "Engagement", "Anticipation", "Sensitivity", 
            "Annoyance", "Sympathy", "Pleasure"
        ]
        
        self.emotion_to_id = {emotion: idx for idx, emotion in enumerate(self.predefined_emotions)}
        self.id_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_id.items()}
        self.num_emotions = len(self.predefined_emotions)
        
    def load_person_data(self, csv_path: str, person_base_path: str = "data/data_person/emotic/emotic") -> pd.DataFrame:
        """인물 데이터 로드 (상대경로 처리)"""
        print(f"인물 데이터 로드: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"  - 원본 데이터: {len(df)}개")
        print(f"  - 컬럼: {df.columns.tolist()}")
        
        # 인물 데이터는 상대경로이므로 base_path와 결합
        def fix_person_path(image_path):
            if pd.isna(image_path):
                return ""
            
            path_str = str(image_path).strip()
            
            # 이미 절대경로인 경우
            if os.path.isabs(path_str) and os.path.exists(path_str):
                return path_str
            
            # 상대경로를 절대경로로 변환
            full_path = os.path.join(person_base_path, path_str)
            if os.path.exists(full_path):
                return full_path
            
            # 다른 패턴들 시도
            alternatives = [
                os.path.join("data/data_person/emotic", path_str),
                os.path.join("./", path_str),
                path_str
            ]
            
            for alt_path in alternatives:
                if os.path.exists(alt_path):
                    return alt_path
            
            return path_str  # 원본 반환
        
        df['full_image_path'] = df['image_path'].apply(fix_person_path)
        df['domain'] = 'person'
        df['domain_id'] = 0
        
        # 감정 라벨 파싱
        if 'emotions' in df.columns:
            df['emotions_list'] = df['emotions'].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) and x.strip() != '' else []
            )
        else:
            df['emotions_list'] = [[] for _ in range(len(df))]
        
        return df
    
    def load_nature_data(self, csv_path: str) -> pd.DataFrame:
        """자연 데이터 로드 (절대경로 처리)"""
        print(f"자연 데이터 로드: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"  - 원본 데이터: {len(df)}개")
        print(f"  - 컬럼: {df.columns.tolist()}")
        
        # 자연 데이터는 이미 절대경로
        def fix_nature_path(image_path):
            if pd.isna(image_path):
                return ""
            
            path_str = str(image_path).strip()
            
            # 절대경로가 존재하는지 확인
            if os.path.exists(path_str):
                return path_str
            
            # 경로 수정이 필요한 경우들
            # Windows 경로 구분자 문제 해결
            if '\\' in path_str:
                normalized_path = path_str.replace('\\', '/')
                if os.path.exists(normalized_path):
                    return normalized_path
            
            return path_str
        
        df['full_image_path'] = df['image_path'].apply(fix_nature_path)
        df['domain'] = 'nature'
        df['domain_id'] = 1
        
        # 감정 라벨 파싱
        if 'emotions' in df.columns:
            df['emotions_list'] = df['emotions'].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) and x.strip() != '' else []
            )
        else:
            df['emotions_list'] = [[] for _ in range(len(df))]
        
        return df
    
    def validate_and_filter_data(self, df: pd.DataFrame, domain_name: str) -> pd.DataFrame:
        """데이터 유효성 검사 및 필터링"""
        print(f"{domain_name} 데이터 유효성 검사...")
        
        valid_images = []
        invalid_count = 0
        
        # 샘플 경로 확인
        print("샘플 경로 확인:")
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            exists = os.path.exists(row['full_image_path'])
            print(f"  {i+1}. {row['full_image_path'][:80]}... → {exists}")
        
        # 전체 이미지 검증
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{domain_name} 이미지 검증"):
            image_path = row['full_image_path']
            if os.path.exists(image_path):
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                    valid_images.append(True)
                except Exception as e:
                    print(f"이미지 검증 실패: {image_path} - {e}")
                    valid_images.append(False)
                    invalid_count += 1
            else:
                valid_images.append(False)
                invalid_count += 1
        
        df['valid_image'] = valid_images
        df_valid = df[df['valid_image']].copy()
        
        print(f"  - 유효하지 않은 이미지: {invalid_count}개")
        print(f"  - 유효한 데이터: {len(df_valid)}개")
        
        return df_valid
    
    def create_binary_labels(self, df: pd.DataFrame) -> np.ndarray:
        """17가지 감정에 대한 바이너리 라벨 매트릭스 생성"""
        print("바이너리 라벨 매트릭스 생성...")
        
        binary_labels = np.zeros((len(df), self.num_emotions), dtype=np.float32)
        skipped_emotions = set()
        
        for idx, emotions_list in enumerate(df['emotions_list']):
            if isinstance(emotions_list, list):
                for emotion in emotions_list:
                    if emotion in self.emotion_to_id:
                        emotion_idx = self.emotion_to_id[emotion]
                        binary_labels[idx, emotion_idx] = 1.0
                    else:
                        skipped_emotions.add(emotion)
        
        if skipped_emotions:
            print(f"스킵된 감정: {skipped_emotions}")
        
        # 통계
        label_sums = binary_labels.sum(axis=0)
        no_label_count = (binary_labels.sum(axis=1) == 0).sum()
        
        print(f"  - 라벨 매트릭스: {binary_labels.shape}")
        print(f"  - 평균 라벨/이미지: {binary_labels.sum(axis=1).mean():.2f}")
        print(f"  - 라벨 없는 이미지: {no_label_count}개")
        
        return binary_labels
    
    def extract_features(self, df: pd.DataFrame, domain_name: str, batch_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        """이미지와 텍스트 특징 추출"""
        print(f"{domain_name} 특징 추출 중...")
        
        # 이미지 특징 추출
        print("  이미지 특징 추출...")
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
                    batch_images.append(Image.new('RGB', (224, 224), color='black'))
            
            if batch_images:
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                with torch.no_grad():
                    pixel_values = inputs['pixel_values'].to(self.device)
                    image_features = self.model.get_image_features(pixel_values)
                    image_features_list.append(image_features.cpu())
        
        all_image_features = torch.cat(image_features_list, dim=0)
        
        # 텍스트 특징 추출
        print("  텍스트 특징 추출...")
        text_features_list = []
        
        # caption 처리
        if 'caption' in df.columns and not df['caption'].isna().all():
            descriptions = df['caption'].fillna("").astype(str).tolist()
        elif 'class_name' in df.columns:
            descriptions = df['class_name'].astype(str).tolist()
        else:
            descriptions = [f"{domain_name} image"] * len(df)
        
        for i in tqdm(range(0, len(descriptions), batch_size), desc="텍스트 처리"):
            batch_texts = descriptions[i:i+batch_size]
            
            inputs = self.processor(text=batch_texts, return_tensors="pt", 
                                  padding=True, truncation=True, max_length=self.max_length)
            
            with torch.no_grad():
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                text_features = self.model.get_text_features(input_ids=input_ids, 
                                                           attention_mask=attention_mask)
                text_features_list.append(text_features.cpu())
        
        all_text_features = torch.cat(text_features_list, dim=0)
        
        print(f"  - 이미지 특징: {all_image_features.shape}")
        print(f"  - 텍스트 특징: {all_text_features.shape}")
        
        return all_image_features, all_text_features
    
    def process_unified_dataset(self,
                              person_csv: str,
                              nature_csv: str,
                              person_base_path: str = "data/data_person/emotic/emotic",
                              dataset_name: str = "dataset",
                              include_domain_features: bool = True,
                              batch_size: int = 16) -> Dict:
        """인물과 자연 데이터를 통합하여 처리"""
        
        print(f"\n{'='*60}")
        print(f"{dataset_name.upper()} 데이터셋 통합 처리")
        print(f"{'='*60}")
        
        all_features = []
        all_labels = []
        all_metadata = []
        
        # 1. 인물 데이터 처리
        if person_csv and os.path.exists(person_csv):
            print(f"\n1. 인물 데이터 처리")
            print("-" * 40)
            
            person_df = self.load_person_data(person_csv, person_base_path)
            person_df = self.validate_and_filter_data(person_df, "인물")
            
            if len(person_df) > 0:
                person_image_features, person_text_features = self.extract_features(person_df, "인물", batch_size)
                person_combined = torch.cat([person_image_features, person_text_features], dim=1)
                
                # 도메인 정보 추가
                if include_domain_features:
                    domain_features = torch.zeros(len(person_combined), 2)
                    domain_features[:, 0] = 1  # Person = [1, 0]
                    person_combined = torch.cat([person_combined, domain_features], dim=1)
                
                person_labels = self.create_binary_labels(person_df)
                
                all_features.append(person_combined)
                all_labels.append(person_labels)
                all_metadata.append(person_df)
                
                print(f"인물 데이터 완료: {len(person_df)}개")
        else:
            print(f"인물 데이터 파일 없음: {person_csv}")
        
        # 2. 자연 데이터 처리
        if nature_csv and os.path.exists(nature_csv):
            print(f"\n2. 자연 데이터 처리")
            print("-" * 40)
            
            nature_df = self.load_nature_data(nature_csv)
            nature_df = self.validate_and_filter_data(nature_df, "자연")
            
            if len(nature_df) > 0:
                nature_image_features, nature_text_features = self.extract_features(nature_df, "자연", batch_size)
                nature_combined = torch.cat([nature_image_features, nature_text_features], dim=1)
                
                # 도메인 정보 추가
                if include_domain_features:
                    domain_features = torch.zeros(len(nature_combined), 2)
                    domain_features[:, 1] = 1  # Nature = [0, 1]
                    nature_combined = torch.cat([nature_combined, domain_features], dim=1)
                
                nature_labels = self.create_binary_labels(nature_df)
                
                all_features.append(nature_combined)
                all_labels.append(nature_labels)
                all_metadata.append(nature_df)
                
                print(f"자연 데이터 완료: {len(nature_df)}개")
        else:
            print(f"자연 데이터 파일 없음: {nature_csv}")
        
        # 3. 통합
        if not all_features:
            print("처리할 데이터가 없습니다.")
            return None
        
        print(f"\n3. 데이터 통합")
        print("-" * 40)
        
        unified_features = torch.cat(all_features, dim=0)
        unified_labels = np.vstack(all_labels)
        unified_metadata = pd.concat(all_metadata, ignore_index=True)
        
        # 데이터 셔플
        indices = torch.randperm(len(unified_features))
        unified_features = unified_features[indices]
        unified_labels = unified_labels[indices.numpy()]
        unified_metadata = unified_metadata.iloc[indices.numpy()].reset_index(drop=True)
        
        # 통계
        person_count = (unified_metadata['domain'] == 'person').sum()
        nature_count = (unified_metadata['domain'] == 'nature').sum()
        
        print(f"통합 완료:")
        print(f"  - 인물: {person_count:,}개")
        print(f"  - 자연: {nature_count:,}개")
        print(f"  - 총합: {len(unified_features):,}개")
        print(f"  - 특징 차원: {unified_features.shape[1]}")
        
        return {
            'features': unified_features,
            'labels': unified_labels,
            'metadata': unified_metadata,
            'stats': {
                'person_count': person_count,
                'nature_count': nature_count,
                'total_count': len(unified_features)
            }
        }

def process_all_splits(person_base_path: str = "data/data_person/emotic/emotic",
                      nature_base_path: str = "data/data_nature/splits_nature",
                      output_dir: str = "unified_emotion_preprocessed",
                      include_domain_features: bool = True,
                      batch_size: int = 16):
    """모든 train/val/test 분할을 통합 처리"""
    
    print("="*80)
    print("UNIFIED PERSON & NATURE EMOTION PREPROCESSING")
    print("17가지 감정: Happiness, Confidence, Surprise, Pain, Disquietment,")
    print("            Fear, Yearning, Excitement, Embarrassment, Affection,")
    print("            Aversion, Engagement, Anticipation, Sensitivity,") 
    print("            Annoyance, Sympathy, Pleasure")
    print("="*80)
    
    # 전처리기 초기화
    preprocessor = UnifiedEmotionPreprocessor()
    
    # 데이터셋 경로 설정
    datasets = [
        {
            'name': 'train',
            'person_csv': f"{person_base_path}/../dataset_min_train.csv",  # 상위 폴더
            'nature_csv': f"{nature_base_path}/train.csv"
        },
        {
            'name': 'val', 
            'person_csv': f"{person_base_path}/../dataset_min_val.csv",
            'nature_csv': f"{nature_base_path}/val.csv"
        },
        {
            'name': 'test',
            'person_csv': f"{person_base_path}/../dataset_min_test.csv", 
            'nature_csv': f"{nature_base_path}/test.csv"
        }
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    processed_datasets = {}
    total_stats = {'person_samples': 0, 'nature_samples': 0, 'total_samples': 0}
    
    # 각 분할 처리
    for dataset_info in datasets:
        dataset_name = dataset_info['name']
        person_csv = dataset_info['person_csv']
        nature_csv = dataset_info['nature_csv']
        
        # 통합 처리
        result = preprocessor.process_unified_dataset(
            person_csv=person_csv,
            nature_csv=nature_csv, 
            person_base_path=person_base_path,
            dataset_name=dataset_name,
            include_domain_features=include_domain_features,
            batch_size=batch_size
        )
        
        if result:
            # 저장
            dataset_dir = os.path.join(output_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            torch.save(result['features'], os.path.join(dataset_dir, 'features.pt'))
            torch.save(torch.from_numpy(result['labels']), os.path.join(dataset_dir, 'labels.pt'))
            result['metadata'].to_csv(os.path.join(dataset_dir, 'metadata.csv'), index=False)
            
            processed_datasets[dataset_name] = result
            
            # 통계 누적
            total_stats['person_samples'] += result['stats']['person_count']
            total_stats['nature_samples'] += result['stats']['nature_count']
            total_stats['total_samples'] += result['stats']['total_count']
            
            print(f"\n{dataset_name} 저장 완료: {dataset_dir}")
    
    # 전역 메타데이터 저장
    feature_dim = 1024 + (2 if include_domain_features else 0)
    
    global_metadata = {
        'emotion_to_id': preprocessor.emotion_to_id,
        'id_to_emotion': preprocessor.id_to_emotion,
        'num_emotions': preprocessor.num_emotions,
        'predefined_emotions': preprocessor.predefined_emotions,
        'clip_model': 'openai/clip-vit-base-patch32',
        'feature_dim_image': 512,
        'feature_dim_text': 512,
        'feature_dim_combined': 1024,
        'include_domain_features': include_domain_features,
        'feature_dim_final': feature_dim,
        'domain_mapping': {'person': 0, 'nature': 1},
        'total_stats': total_stats
    }
    
    with open(os.path.join(output_dir, 'global_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(global_metadata, f, ensure_ascii=False, indent=2)
    
    # 최종 리포트
    generate_final_report(processed_datasets, total_stats, output_dir)
    
    print(f"\n✅ 통합 전처리 완료!")
    print(f"📁 출력 디렉토리: {output_dir}")
    print(f"📊 총 샘플: {total_stats['total_samples']:,}개")
    print(f"   - 인물: {total_stats['person_samples']:,}개")
    print(f"   - 자연: {total_stats['nature_samples']:,}개")
    print(f"🎯 특징 차원: {feature_dim}")
    
    return processed_datasets, global_metadata

def generate_final_report(processed_datasets: Dict, total_stats: Dict, output_dir: str):
    """최종 리포트 생성"""
    
    report_path = os.path.join(output_dir, "unified_preprocessing_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("UNIFIED PERSON & NATURE EMOTION PREPROCESSING REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("데이터 개요\n")
        f.write("-" * 30 + "\n")
        f.write(f"총 샘플: {total_stats['total_samples']:,}개\n")
        f.write(f"  - 인물: {total_stats['person_samples']:,}개 ({total_stats['person_samples']/total_stats['total_samples']*100:.1f}%)\n")
        f.write(f"  - 자연: {total_stats['nature_samples']:,}개 ({total_stats['nature_samples']/total_stats['total_samples']*100:.1f}%)\n\n")
        
        f.write("데이터셋별 분포\n")
        f.write("-" * 30 + "\n")
        for name, data in processed_datasets.items():
            stats = data['stats']
            f.write(f"{name.upper()}: {stats['total_count']:,}개\n")
            f.write(f"  - 인물: {stats['person_count']:,}개\n")
            f.write(f"  - 자연: {stats['nature_count']:,}개\n\n")
        
        f.write("사용법\n")
        f.write("-" * 30 + "\n")
        f.write("# 데이터 로딩\n")
        f.write("train_features = torch.load('train/features.pt')  # (N, 1026)\n")
        f.write("train_labels = torch.load('train/labels.pt')      # (N, 17)\n\n")
        f.write("# 모델 예시\n")
        f.write("model = nn.Sequential(\n")
        f.write("    nn.Linear(1026, 512),\n")
        f.write("    nn.ReLU(),\n")
        f.write("    nn.Dropout(0.3),\n")
        f.write("    nn.Linear(512, 256),\n")
        f.write("    nn.ReLU(),\n")
        f.write("    nn.Dropout(0.3),\n")
        f.write("    nn.Linear(256, 17)  # 17가지 감정\n")
        f.write(")\n\n")
        f.write("# 손실함수: BCEWithLogitsLoss (멀티라벨)\n")
        f.write("# 평가지표: F1-macro, F1-micro, AUC-ROC\n")
    
    print(f"📄 리포트: {report_path}")

# 실행 예시
if __name__ == "__main__":
    # 통합 전처리 실행
    datasets, metadata = process_all_splits(
        person_base_path="data/data_person/emotic/emotic",
        nature_base_path="data/data_nature/splits_nature", 
        output_dir="unified_emotion_data",
        include_domain_features=True,
        batch_size=8  # GPU 메모리에 따라 조정
    )