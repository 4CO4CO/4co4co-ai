# clip_inference.py
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings
import logging

warnings.filterwarnings('ignore')

# ── 로깅 ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ── MLP ─────────────────────────────────────────────────────────
class MLP(nn.Module):
    """감정 분류를 위한 MLP 모델"""
    def __init__(self, in_dim=1024, hidden=(512, 256), out_dim=17, dropout=0.2):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(True), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── CLIP + MLP ──────────────────────────────────────────────────
class CLIPEmotionInference:
    """CLIP + MLP 기반 감정 분석 시스템"""

    def __init__(self, model_path: str, device: str = "auto"):
        # 17가지 감정 리스트
        self.emotions = [
            "Happiness", "Confidence", "Surprise", "Pain", "Disquietment",
            "Fear", "Yearning", "Excitement", "Embarrassment", "Affection",
            "Aversion", "Engagement", "Anticipation", "Sensitivity",
            "Annoyance", "Sympathy", "Pleasure"
        ]

        # 디바이스
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"CLIP 감정 분석기 초기화 - 디바이스: {self.device}")

        self.clip_model = None
        self.clip_processor = None
        self.mlp_model: Optional[MLP] = None
        self.thresholds: Optional[np.ndarray] = None
        self.model_path = model_path
        self.expected_in_dim: Optional[int] = None  # MLP가 기대하는 입력 차원

        # 모델 로드
        self.load_clip_model()
        self.load_mlp_model(model_path)

        logger.info("CLIP 감정 분석기 초기화 완료")

    # ── 로드 ────────────────────────────────────────────────────
    def load_clip_model(self):
        """CLIP 모델 로드"""
        try:
            logger.info("CLIP 모델 로드 중...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.eval()
            logger.info("CLIP 모델 로드 완료")
        except Exception as e:
            logger.error(f"CLIP 모델 로드 실패: {e}")
            raise

    def _infer_dims_from_state_dict(self, state_dict: Dict) -> Dict[str, Union[int, List[int]]]:
        """
        state_dict에서 Linear 레이어들의 (out, in) 차원을 읽어 in_dim/hidden/out_dim 추정
        """
        linear_shapes = []
        for k, v in state_dict.items():
            if k.endswith(".weight") and v.ndim == 2:  # Linear weight: (out, in)
                linear_shapes.append((k, tuple(v.shape)))
        # net.0.weight, net.3.weight, net.6.weight ... 순서일 가능성이 큼
        linear_shapes.sort(key=lambda x: x[0])

        if not linear_shapes:
            raise RuntimeError("state_dict에서 Linear 레이어를 찾지 못했습니다.")

        ins = [s[1][1] for s in linear_shapes]  # in_dims
        outs = [s[1][0] for s in linear_shapes]  # out_dims

        in_dim = ins[0]
        if len(linear_shapes) >= 2:
            hidden = outs[:-1]  # 마지막은 out_dim
        else:
            hidden = []
        out_dim = outs[-1]

        return {"in_dim": in_dim, "hidden": hidden, "out_dim": out_dim}

    def load_mlp_model(self, model_path: str):
        """훈련된 MLP 모델 로드(메타 누락에도 견고)"""
        try:
            logger.info(f"MLP 모델 로드 중: {model_path}")
            p = Path(model_path)
            if not p.exists():
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

            checkpoint = torch.load(str(p), map_location="cpu")

            # checkpoint 형태 정규화
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                meta_in_dim = checkpoint.get("in_dim")
                meta_hidden = checkpoint.get("hidden")
                meta_out_dim = checkpoint.get("out_dim")
                dropout = checkpoint.get("dropout", 0.2)
            elif isinstance(checkpoint, dict):
                # state_dict 자체만 저장된 경우
                state_dict = checkpoint
                meta_in_dim = meta_hidden = meta_out_dim = None
                dropout = 0.2
            else:
                raise RuntimeError("알 수 없는 체크포인트 형식입니다.")

            # 메타가 없으면 state_dict로부터 추론
            if meta_in_dim is None or meta_hidden is None or meta_out_dim is None:
                dims = self._infer_dims_from_state_dict(state_dict)
                in_dim = int(dims["in_dim"])
                hidden = [int(x) for x in dims["hidden"]]
                out_dim = int(dims["out_dim"])
            else:
                in_dim = int(meta_in_dim)
                hidden = [int(x) for x in meta_hidden]
                out_dim = int(meta_out_dim)

            # 감정 라벨 수와 out_dim 불일치 시 경고(라벨 매핑 확인 필요)
            if out_dim != len(self.emotions):
                logger.warning(f"체크포인트 out_dim={out_dim} 이(가) emotions 수={len(self.emotions)}와 다릅니다.")

            # MLP 빌드 및 로드
            self.mlp_model = MLP(in_dim=in_dim, hidden=tuple(hidden), out_dim=out_dim, dropout=dropout).to(self.device)
            self.mlp_model.load_state_dict(state_dict, strict=True)
            self.mlp_model.eval()

            self.expected_in_dim = in_dim  # 이후 정렬용
            logger.info(f"MLP 모델 로드 완료 (in_dim={in_dim}, hidden={hidden}, out_dim={out_dim})")

            # 임계값
            if isinstance(checkpoint, dict) and "thresholds" in checkpoint:
                thr = checkpoint["thresholds"]
                thr = np.array(thr, dtype=np.float32)
                self.thresholds = thr
            else:
                self.thresholds = np.full(out_dim, 0.5, dtype=np.float32)

            # 임계값 길이 확인
            if self.thresholds.shape[0] != out_dim:
                logger.warning(f"임계값 길이({self.thresholds.shape[0]})가 out_dim({out_dim})과 달라 0.5로 초기화합니다.")
                self.thresholds = np.full(out_dim, 0.5, dtype=np.float32)

        except Exception as e:
            logger.error(f"MLP 모델 로드 실패: {e}")
            raise

    # ── 인코딩 ─────────────────────────────────────────────────
    def encode_image_text_features(self, image_path: str, caption: str = "") -> torch.Tensor:
        """이미지/텍스트를 CLIP으로 인코딩 → (1, 1024) 벡터 리턴"""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.clip_processor(text=[caption], images=[image], return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                img_features = self.clip_model.get_image_features(inputs["pixel_values"])     # (1, 512)
                txt_features = self.clip_model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )  # (1, 512)

                # (선택) 정규화 — 학습 시 정규화를 썼다면 주석 해제
                # img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                # txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)

                combined = torch.cat([img_features, txt_features], dim=1)  # (1, 1024)
            return combined  # device: self.device
        except Exception as e:
            logger.error(f"특징 추출 실패: {e}")
            return None

    # ── 차원 정렬/확장 ─────────────────────────────────────────
    def _merge_with_extras(
        self,
        base_feat: torch.Tensor,
        extra_features: Optional[Union[List[float], np.ndarray, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        base_feat: (1, D)
        extra_features가 있으면 뒤에 이어붙이고,
        최종적으로 self.expected_in_dim에 맞춰 0-padding/슬라이스.
        """
        assert self.expected_in_dim is not None, "MLP expected_in_dim이 설정되어야 합니다."
        feat = base_feat
        if extra_features is not None:
            if isinstance(extra_features, np.ndarray):
                extra = torch.from_numpy(extra_features.astype(np.float32)).view(1, -1).to(feat.device)
            elif isinstance(extra_features, list):
                extra = torch.tensor(extra_features, dtype=feat.dtype, device=feat.device).view(1, -1)
            elif isinstance(extra_features, torch.Tensor):
                extra = extra_features.to(feat.device)
                if extra.dim() == 1:
                    extra = extra.view(1, -1)
            else:
                raise TypeError("extra_features는 list/np.ndarray/torch.Tensor 중 하나여야 합니다.")
            feat = torch.cat([feat, extra], dim=1)

        B, D = feat.shape
        T = self.expected_in_dim
        if D == T:
            return feat
        if D < T:
            pad = torch.zeros(B, T - D, device=feat.device, dtype=feat.dtype)
            return torch.cat([feat, pad], dim=1)
        return feat[:, :T]  # 너무 길면 잘라냄

    # ── 추론 ───────────────────────────────────────────────────
    def predict_emotions(
        self,
        image_path: str,
        caption: str = "",
        extra_features: Optional[Union[List[float], np.ndarray, torch.Tensor]] = None
    ) -> Dict:
        """
        extra_features 예:
          - 사람 탐지 분기에서: [has_person(0/1), min(person_count,5)/5]
          - 기타 스칼라 피처들
        아무것도 안 주면 0-padding으로 자동 정렬 → 차원 오류 방지
        """
        try:
            feats = self.encode_image_text_features(image_path, caption)
            if feats is None:
                raise ValueError("특징 추출 실패")

            # 차원 정렬 (1024 → expected_in_dim)
            feats = self._merge_with_extras(feats, extra_features).to(self.device)

            with torch.no_grad():
                logits = self.mlp_model(feats)
                probs = torch.sigmoid(logits).cpu().numpy()[0]  # (out_dim,)

            # 퍼센트 변환
            emotion_percentages = {self.emotions[i] if i < len(self.emotions) else f"Label_{i}": float(p) * 100.0
                                   for i, p in enumerate(probs)}

            # 상위 정렬
            sorted_emotions = sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True)

            # 임계값(멀티라벨) 적용
            thr = self.thresholds if self.thresholds is not None else np.full(len(probs), 0.5, dtype=np.float32)
            if thr.shape[0] != len(probs):
                logger.warning("임계값 길이가 출력 차원과 달라 0.5로 재설정합니다.")
                thr = np.full(len(probs), 0.5, dtype=np.float32)

            predictions = (probs >= thr).astype(int)
            predicted_emotions = []
            for i, pred in enumerate(predictions):
                name = self.emotions[i] if i < len(self.emotions) else f"Label_{i}"
                if pred == 1:
                    predicted_emotions.append(name)

            return {
                "method": "clip_mlp_prediction",
                "image_path": str(image_path),
                "caption": caption,
                "emotion_percentages": {k: round(v, 3) for k, v in emotion_percentages.items()},
                "top_emotions": [(e, round(p, 3)) for e, p in sorted_emotions[:5]],
                "predicted_emotions": predicted_emotions,
                "model_info": {
                    "model_path": self.model_path,
                    "expected_in_dim": self.expected_in_dim,
                    "actual_feat_dim_before_align": 1024,
                    "extra_features_used": (extra_features is not None),
                    "threshold_used": "custom" if (self.thresholds is not None) else "default_0.5"
                }
            }

        except Exception as e:
            logger.error(f"MLP 감정 예측 실패: {e}")
            return {
                "method": "clip_mlp_prediction",
                "error": str(e),
                "emotion_percentages": {emotion: 0.0 for emotion in self.emotions}
            }

    # ── 임계값 로드 ────────────────────────────────────────────
    def load_custom_thresholds(self, threshold_path: str):
        try:
            thr = np.load(threshold_path)
            thr = np.array(thr, dtype=np.float32)
            self.thresholds = thr
            logger.info(f"임계값 로드 완료: {threshold_path}")
        except Exception as e:
            logger.error(f"임계값 로드 실패: {e}")


# ── 단독 실행/테스트 ───────────────────────────────────────────
def test_clip_inference():
    """간단 테스트(경로 수정해서 사용)"""
    test_image = "test_image.jpg"
    test_model = "./model/mlp.pt"
    caption = "a photo of people smiling"

    if not Path(test_model).exists():
        print(f"테스트 모델 '{test_model}'을 찾을 수 없습니다. --model 옵션으로 지정하세요.")
        return

    analyzer = CLIPEmotionInference(model_path=test_model)
    # 예시 extra_features: [has_person, person_]()_
