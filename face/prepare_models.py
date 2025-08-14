# prepare_models.py

import os
import torch
import requests
import torchvision.models as models
from typing import Optional

# ---- 안전한 다운로드 ----
def _download(url: str, path: str, timeout: int = 120) -> bool:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
            r.raise_for_status()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp, path)
        print(f"[INFO] Downloaded: {url}")
        return True
    except Exception as e:
        print(f"[WARN] Download failed from {url}: {e}")
        return False

# ---- pickle latin1 호환 래퍼 (torch.load에 전달) ----
import pickle
class PickleLatin1:
    @staticmethod
    def load(file, **kwargs):
        # torch<1.13 구형 포맷 호환을 위해 encoding='latin1'
        return pickle.load(file, encoding="latin1")
    Unpickler = pickle.Unpickler  # torch 내부에서 클래스 참조함

def _load_checkpoint(pth_path: str):
    return torch.load(pth_path, map_location="cpu", pickle_module=PickleLatin1)

# ---- 메인 준비 함수 ----
def prep_models(
    context_model: str = "resnet18",
    body_model: str = "resnet18",
    model_dir: str = "./",
    timeout: int = 120,
    hf_only: bool = False,             # True면 Places2 건너뛰고 HF만 시도
    hf_urls: Optional[list] = None     # 기본 제공 링크 사용, 필요시 교체 가능
):
    """
    1) Places2에서 {context_model}_places365.pth.tar 다운로드 (기본)
    2) 실패 시 Hugging Face 링크들에서 재시도
    3) 성공하면 Places365 가중치로 context_model 로드/저장
    4) body_model은 ImageNet 가중치 사용
    """

    os.makedirs(model_dir, exist_ok=True)
    fname = f"{context_model}_places365.pth.tar"
    fpath = os.path.join(model_dir, fname)

    # 기본 HF 미러(커밋 고정 → main 브랜치)
    if hf_urls is None:
        hf_urls = [
            "https://huggingface.co/spaces/aimg15/Proyecto_Topicos_Final/resolve/f5913032657e514c0dc4011f304fd556795512d0/resnet18_places365.pth.tar",
            "https://huggingface.co/spaces/aimg15/Proyecto_Topicos_Final/resolve/main/resnet18_places365.pth.tar",
        ]

    # 1) 다운로드 없으면 시도
    if not os.path.exists(fpath):
        tried = False
        ok = False

        # Places2 먼저(원하면 hf_only=True로 건너뛰기)
        if not hf_only:
            for u in [
                f"https://places2.csail.mit.edu/models_places365/{fname}",
                f"http://places2.csail.mit.edu/models_places365/{fname}",
            ]:
                tried = True
                if _download(u, fpath, timeout=timeout):
                    ok = True
                    break

        # HF 시도
        if not ok:
            for u in hf_urls:
                tried = True
                if _download(u, fpath, timeout=timeout):
                    ok = True
                    break

        if not ok and tried:
            raise RuntimeError("Places365 가중치 다운로드 실패 (Places2/HF 모두 실패).")
        if not tried:
            raise RuntimeError("다운로드 시도 링크가 없습니다.")

    # 2) 체크포인트 로드
    checkpoint = _load_checkpoint(fpath)

    # 3) Context 모델 구성(Places365: num_classes=365)
    model_context = models.__dict__[context_model](num_classes=365)

    # state_dict 키 정리
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    if context_model == "densenet161":
        fixed = {}
        for k, v in state_dict.items():
            k = k.replace("norm.", "norm").replace("conv.", "conv")
            k = k.replace("normweight", "norm.weight")
            k = k.replace("normrunning", "norm.running")
            k = k.replace("normbias", "norm.bias")
            k = k.replace("convweight", "conv.weight")
            fixed[k] = v
        state_dict = fixed

    model_context.load_state_dict(state_dict, strict=False)
    model_context.eval()
    torch.save(model_context.cpu(), os.path.join(model_dir, "context_model.pth"))
    print("[OK] Prepared context model (Places365).")

    # 4) Body 모델 (ImageNet 가중치)
    try:
        model_body = models.__dict__[body_model](weights="IMAGENET1K_V1")  # torchvision>=0.13
    except Exception:
        model_body = models.__dict__[body_model](pretrained=True)          # 구버전 호환
    model_body.eval()
    torch.save(model_body.cpu(), os.path.join(model_dir, "body_model.pth"))
    print("[OK] Prepared body model (ImageNet).")

    return model_context, model_body


if __name__ == "__main__":
    # 필요하면 hf_only=True로 바꿔 HF만 사용
    prep_models(
        context_model="resnet18",
        body_model="resnet18",
        model_dir="proj/debug_exp/models",
        timeout=120,
        hf_only=False,  # True면 Places2 건너뛰고 HF만
        # hf_urls=[...], # 직접 다른 HF 링크를 지정하고 싶으면 여기에 리스트로 전달
    )
