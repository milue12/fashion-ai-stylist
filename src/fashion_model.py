# ===================================================================
# [AI 모듈: fashion_model.py]
# 역할: AI 모델(YOLO, H1-H4, CLIP)을 로드하고,
#      이미지 경로를 받아 "순수 AI 분석" JSON만 반환합니다.
#      (LLM 호출은 이 파일에 포함되지 않습니다.)
# ===================================================================

import os, json, math, warnings, io, re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open_clip
from PIL import Image
from ultralytics import YOLO

# --- 1. 환경 변수 및 상수 정의 ---
PROJECT_ROOT = Path.cwd()
CKPT_DIR = PROJECT_ROOT / "checkpoints"
MIXED_CKPT_DIR = PROJECT_ROOT / "checkpoints_mixed"

USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"
VISION_DEVICE = DEVICE

MODEL_NAME = "ViT-B-32"
PRETRAINED_SOURCE = "laion2b_s34b_b79k"

YOLO_LABELS = [
    'short sleeve top','long sleeve top','short sleeve outwear',
    'long sleeve outwear','vest','sling','shorts',
    'trousers','skirt','short sleeve dress',
    'long sleeve dress','vest dress','sling dress'
]
H3_LABELS = YOLO_LABELS.copy()
N_H3_CLASSES = len(H3_LABELS)

MATERIALS = ["denim","leather","cotton","wool","silk","knit","suede","fur"]
COLORS    = ["black","white","gray","red","blue","green","yellow","pink","purple","orange","brown","beige"]

print(f"--- [AI 모듈] 로드 환경: {DEVICE} ---")

# --- 2. 모델 아키텍처 정의 (H1, H2, H3, H4) ---

def find_weight(patterns: List[str], start: Path) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(start.glob(pat))
        if hits: return hits[0]
    return None

def _extract_state_dict(obj: dict) -> dict:
    if isinstance(obj, dict):
        for k in ["state_dict","model","net","module"]:
            if k in obj and isinstance(obj[k], dict):
                return _extract_state_dict(obj[k])
    return obj if isinstance(obj, dict) else {}

def _strip_prefix_keys(sd: dict, prefix: str) -> dict:
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out

def _infer_h4_head_spec(cls_sd: dict, emb_dim: int):
    use_layernorm = False; hidden_dim = 256; out_dim = 2
    w0 = cls_sd.get("0.weight", None); w3 = cls_sd.get("3.weight", None)
    if w0 is not None:
        if getattr(w0, "ndim", None) == 1:
            use_layernorm = True
            for idx in [1,2,3,4]:
                w_lin = cls_sd.get(f"{idx}.weight", None)
                if w_lin is not None and getattr(w_lin, "ndim", None) == 2 and w_lin.shape[1] == emb_dim:
                    hidden_dim = w_lin.shape[0]; break
        elif getattr(w0,"ndim",None) == 2 and w0.shape[1] == emb_dim:
            hidden_dim = w0.shape[0]
    if w3 is not None and getattr(w3,"ndim",None) == 2:
        out_dim = w3.shape[0]
    return use_layernorm, hidden_dim, out_dim

def _build_h4_head_from_spec(emb_dim: int, use_layernorm: bool, hidden_dim: int, out_dim: int):
    layers = []
    if use_layernorm:
        layers += [nn.LayerNorm(emb_dim), nn.Linear(emb_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
    else:
        layers += [nn.Linear(emb_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
    layers += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)

def _safe_load_head(module: nn.Module, sd_full: dict, prefix: str = ""):
    sd = _extract_state_dict(sd_full)
    if prefix: sd = _strip_prefix_keys(sd, prefix)
    cur = module.state_dict()
    compat = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
    module.load_state_dict(compat, strict=False)
    skipped = [k for k in sd if k not in compat]
    if skipped: print(f"[SAFE-LOAD] skipped {len(skipped)} keys for {module.__class__.__name__} (shape mismatch).")

print("Loading OpenCLIP base model...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED_SOURCE, device=VISION_DEVICE
)
clip_model = clip_model.eval()
EMB_DIM = clip_model.visual.output_dim
print(f"Base model loaded. EMB_DIM={EMB_DIM}")

class H4ContextHead(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, classifier_head: Optional[nn.Module]=None):
        super().__init__()
        self.clip_vision_model = clip_model.visual
        for p in self.clip_vision_model.parameters(): p.requires_grad = False
        self.classifier_head = classifier_head or nn.Sequential(
            nn.Linear(emb_dim,256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256,2)
        )
    def forward(self, images):
        return self.classifier_head(self.clip_vision_model(images))

class H1AestheticHead(nn.Module):
    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.clip_vision_model = clip_model.visual
        for p in self.clip_vision_model.parameters(): p.requires_grad = False
        D = emb_dim
        self.aesthetic_head = nn.Sequential(
            nn.LayerNorm(D), nn.Linear(D, D//2), nn.GELU(), nn.Linear(D//2, 1)
        )
    def forward(self, images):
        f = self.clip_vision_model(images)
        output = self.aesthetic_head(f).squeeze(1)
        return torch.sigmoid(output) * 9.0 + 1.0

class H3ItemClassifier(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, num_classes=N_H3_CLASSES):
        super().__init__()
        self.clip_vision_model = clip_model.visual
        for p in self.clip_vision_model.parameters(): p.requires_grad = False
        self.classifier_head = nn.Sequential(
            nn.Linear(emb_dim,512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512,num_classes)
        )
    def forward(self, images):
        return self.classifier_head(self.clip_vision_model(images))

class H2OutfitHead(nn.Module):
    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        D = emb_dim; P = 128
        self.refiner_head = nn.Sequential(nn.Linear(D,D), nn.ReLU(), nn.Linear(D,P))
    def forward(self, raw_clip_features):
        return self.refiner_head(raw_clip_features)

# --- 3. [핵심] 모든 모델 가중치 "미리 로드" (Warm-up) ---
# (이 파일이 import되는 순간, 모든 모델이 메모리에 로드됩니다)

print("\n--- [AI 모듈] 모델 가중치 로드 시작 ---")

# --- H4 (원본) ---
h4_path = find_weight(["h4_best_*.pt","h4*.pt","h4*.pth"], start=CKPT_DIR)
if not h4_path or not h4_path.exists():
    raise FileNotFoundError("H4 weights not found under ./checkpoints")
raw = torch.load(h4_path, map_location="cpu")
sd_full = _extract_state_dict(raw)
sd_cls  = _strip_prefix_keys(sd_full, "classifier_head.")
use_ln, hid_dim, out_dim = _infer_h4_head_spec(sd_cls, EMB_DIM); print("[H4] inferred spec")
h4_head = _build_h4_head_from_spec(EMB_DIM, use_ln, hid_dim, out_dim)
h4_model = H4ContextHead(classifier_head=h4_head).to(VISION_DEVICE).eval()
_safe_load_head(h4_model.classifier_head, sd_full, "classifier_head.")
print(f"Trained H4 loaded: {h4_path}")

# --- H1 (원본) ---
H1_MODEL = H1AestheticHead().to(VISION_DEVICE).eval()
h1_path = find_weight(["h1_best_*.pt","h1*.pt","h1*.pth"], start=CKPT_DIR)
if h1_path and h1_path.exists():
    st = torch.load(h1_path, map_location="cpu")
    _safe_load_head(H1_MODEL.aesthetic_head, st, "head.")
    print(f"Trained H1 loaded: {h1_path}")
else:
    H1_MODEL = None; print("[Warning] H1 weights not found.")

# --- H3 (새 모델) ---
H3_MODEL = H3ItemClassifier(num_classes=N_H3_CLASSES).to(VISION_DEVICE).eval()
h3_path = find_weight(["h3_mixed_best_*.pt"], start=MIXED_CKPT_DIR)
if h3_path and h3_path.exists():
    st = torch.load(h3_path, map_location="cpu")
    _safe_load_head(H3_MODEL.classifier_head, st, "")
    print(f"Trained H3 (Mixed) loaded: {h3_path}")
else:
    H3_MODEL = None; print("[Warning] H3 (Mixed) weights not found.")

# --- H2 (원본) ---
H2_MODEL = H2OutfitHead().to(VISION_DEVICE).eval()
h2_path = find_weight(["h2_best_*.pt","h2*.pt","h2*.pth"], start=CKPT_DIR)
if h2_path and h2_path.exists():
    st = torch.load(h2_path, map_location="cpu")
    _safe_load_head(H2_MODEL.refiner_head, st, "projection_head.")
    print(f"Trained H2 loaded: {h2_path}")
else:
    H2_MODEL = None; print("[Warning] H2 weights not found.")

# --- YOLO (새 모델) ---
YOLO_MODEL = None
yolo_weights = MIXED_CKPT_DIR / "yolo_mixed_run" / "weights" / "best.pt"
if not yolo_weights.exists():
    print(f"YOLO (Mixed) weights not found at: {yolo_weights}")
    raise FileNotFoundError(str(yolo_weights))
print(f"--- Using YOLO model: {yolo_weights} ---")
YOLO_MODEL = YOLO(yolo_weights)
print(f"Trained YOLO (Mixed) loaded: {yolo_weights}")


print(f"✅ [AI 모듈] 모든 모델 로드 완료.")

# --- 4. 헬퍼 함수 정의 ---

@dataclass
class DetectedItem:
    category: str
    bbox: List[float]
    score: float
    h3_category: Optional[str] = None
    crop_path: Optional[str] = None
    refined_feature_h2: Optional[torch.Tensor] = None
    colors_topk: Optional[List[Dict[str, float]]] = None
    materials_topk: Optional[List[Dict[str, float]]] = None

@dataclass
class AnalysisSummary:
    context_pred: str
    context_probs: Dict[str, float]
    main_material: str
    main_color: str
    items: List[DetectedItem]
    aesthetics_score_h1: Optional[float]
    compatibility_score_h2: Optional[float]
    main_colors_topk: Optional[List[Dict[str, float]]] = None
    main_materials_topk: Optional[List[Dict[str, float]]] = None

def _safe_mkdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

@torch.inference_mode()
def extract_feature(image: Image.Image, text_candidates: List[str]) -> str:
    image_input = preprocess(image).unsqueeze(0).to(DEVICE)
    text_inputs = open_clip.tokenize(text_candidates).to(DEVICE)
    img_f = clip_model.encode_image(image_input)
    txt_f = clip_model.encode_text(text_inputs)
    img_f /= img_f.norm(dim=-1, keepdim=True)
    txt_f /= txt_f.norm(dim=-1, keepdim=True)
    sim = (100.0 * img_f @ txt_f.T).softmax(dim=-1)
    return text_candidates[int(sim.argmax().item())]

@torch.inference_mode()
def extract_topk_features(image: Image.Image, text_candidates: List[str], top_k: int = 3, min_prob: float = 0.12):
    image_input = preprocess(image).unsqueeze(0).to(DEVICE)
    text_inputs = open_clip.tokenize(text_candidates).to(DEVICE)
    img_f = clip_model.encode_image(image_input)
    txt_f = clip_model.encode_text(text_inputs)
    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
    txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
    sim = (100.0 * img_f @ txt_f.T).softmax(dim=-1).squeeze(0)
    probs = sim.detach().float().cpu().tolist()
    ranked = sorted(list(zip(text_candidates, probs)), key=lambda x: x[1], reverse=True)
    picked = []
    for label, p in ranked[:max(top_k, 1)]:
        if p >= min_prob:
            picked.append((label, float(p)))
    return picked

@torch.inference_mode()
def _extract_item_attributes(item_image: Image.Image) -> (str, str):
    image_input = preprocess(item_image).unsqueeze(0).to(DEVICE)
    img_f = clip_model.encode_image(image_input)
    img_f /= img_f.norm(dim=-1, keepdim=True)
    
    text_colors = open_clip.tokenize(COLORS).to(DEVICE)
    txt_f_colors = clip_model.encode_text(text_colors)
    txt_f_colors /= txt_f_colors.norm(dim=-1, keepdim=True)
    sim_colors = (100.0 * img_f @ txt_f_colors.T).softmax(dim=-1)
    color_label = COLORS[int(sim_colors.argmax().item())]
    
    text_materials = open_clip.tokenize(MATERIALS).to(DEVICE)
    txt_f_materials = clip_model.encode_text(text_materials)
    txt_f_materials /= txt_f_materials.norm(dim=-1, keepdim=True)
    sim_materials = (100.0 * img_f @ txt_f_materials.T).softmax(dim=-1)
    material_label = MATERIALS[int(sim_materials.argmax().item())]
    
    return color_label, material_label

@torch.inference_mode()
def _infer_context_with_h4(image: Image.Image) -> Dict[str, Any]:
    H4_CLASS_NAMES = ["shop", "consumer"]
    x = preprocess(image).unsqueeze(0).to(DEVICE)
    logits = h4_model(x)
    if logits.shape[-1] == 1:
        p1 = torch.sigmoid(logits).item(); probs = [1 - p1, p1]
    else:
        probs = F.softmax(logits, dim=-1).squeeze(0).tolist()
    idx = int(torch.tensor(probs, device=DEVICE).argmax().item())
    return {"context": H4_CLASS_NAMES[idx], "probs": {H4_CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)}}

def _detect_yolo(image_path: str, save_dir: str) -> List[DetectedItem]:
    items: List[DetectedItem] = []
    if not YOLO_MODEL:
        print("YOLO model not loaded, skipping detection."); return items
    
    device_arg = 0 if USE_CUDA else "cpu"
    res = YOLO_MODEL.predict(image_path, device=device_arg, imgsz=896, conf=0.25, iou=0.5, half=False, verbose=False)

    img = Image.open(image_path).convert("RGB")
    _safe_mkdir(save_dir); img_basename = Path(image_path).stem
    for r in res:
        for b in r.boxes:
            cls = int(b.cls.item()); score = float(b.conf.item())
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            crop = img.crop((x1, y1, x2, y2)).convert("RGB")
            crop_name = f"{img_basename}_crop_{cls}_{int(x1)}_{int(y1)}.png"
            crop_path = str(Path(save_dir) / crop_name)
            crop.save(crop_path, format="PNG")
            items.append(DetectedItem(
                category=YOLO_LABELS[cls] if cls < len(YOLO_LABELS) else f"cls_{cls}",
                bbox=[x1,y1,x2,y2], score=score, crop_path=crop_path
            ))
    print(f"YOLO detected {len(items)} items ({'CUDA' if USE_CUDA else 'CPU'}).")
    return items

@torch.inference_mode()
def _classify_items_h3_and_refine_h2(items: List[DetectedItem]) -> None:
    if not items: return
    print(f"Running H3, H2, and Item-Attributes extraction for {len(items)} items...")
    for it in items:
        if not it.crop_path: continue
        try:
            crop_img = Image.open(it.crop_path).convert("RGB")
            x = preprocess(crop_img).unsqueeze(0).to(DEVICE)
            
            if H3_MODEL:
                logits = H3_MODEL(x); pred = int(logits.argmax().item())
                it.h3_category = H3_LABELS[pred] 
                
            if H2_MODEL:
                raw_clip_feature = clip_model.encode_image(x)
                it.refined_feature_h2 = H2_MODEL(raw_clip_feature)
                
            color_label, material_label = _extract_item_attributes(crop_img)
            it.colors_topk = [{"label": color_label, "prob": 1.0}]
            it.materials_topk = [{"label": material_label, "prob": 1.0}]

        except Exception as e: 
            print(f"[H3/H2/Attr] error on {it.crop_path}: {e}")

@torch.inference_mode()
def _score_aesthetic_h1(image: Image.Image) -> Optional[float]:
    if H1_MODEL is None:
        print("H1 model not loaded..."); return None
    try:
        x = preprocess(image).unsqueeze(0).to(DEVICE)
        score_1_to_10 = H1_MODEL(x)
        return float(score_1_to_10.squeeze().item())
    except Exception as e:
        print(f"[H1] error: {e}"); return None

@torch.inference_mode()
def _compute_compatibility_score_h2(items: List[DetectedItem]) -> Optional[float]:
    if H2_MODEL is None:
        print("H2 model not loaded..."); return None
    item_embeddings = [it.refined_feature_h2 for it in items if it.refined_feature_h2 is not None]
    if len(item_embeddings) < 2:
        print(f"H2 needs at least 2 items, found {len(item_embeddings)}. Skipping score."); return None
    pair_scores = []
    for i in range(len(item_embeddings)):
        for j in range(i+1, len(item_embeddings)):
            sim = torch.nn.functional.cosine_similarity(item_embeddings[i], item_embeddings[j]).item()
            pair_scores.append(sim)
    if not pair_scores: return None
    avg_similarity = sum(pair_scores)/len(pair_scores)
    score_1_to_10 = (avg_similarity + 1.0) * 4.5 + 1.0
    return float(score_1_to_10)

# --- 5. [핵심] E2E 분석 함수 (LLM 호출 제외) ---
# (이 함수가 BE #1 (/v1/analyze)의 핵심 로직이 됩니다)

def run_ai_analysis(image_path: str, workdir: str = "./_e2e_runs") -> Dict[str, Any]:
    """
    이미지 경로를 받아, 순수 AI 분석 결과(JSON)만 반환합니다.
    (LLM 호출은 이 함수에서 제외됩니다.)
    """
    print("="*50)
    print(f"--- [AI 모듈] 1. Analyzing Image: ...{image_path[-40:]} ---")
    _safe_mkdir(workdir)
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[Fatal Error] Failed to open image: {e}"); return {"error": f"Failed to open image: {e}"}

    # 1) YOLO 탐지
    print("--- 2. Running YOLOv8 Detection ---")
    img_basename = os.path.splitext(os.path.basename(image_path))[0]
    yolo_dir = os.path.join(workdir, f"{img_basename}_crops")
    items = _detect_yolo(image_path, yolo_dir)

    # 2) 재질/색 (전신 기준)
    print("--- 3. Running CLIP Feature Extraction (Material/Color) ---")
    main_material_global = extract_feature(img, MATERIALS) 
    main_color_global    = extract_feature(img, COLORS)    
    mat_topk = extract_topk_features(img, MATERIALS, top_k=3, min_prob=0.12)
    col_topk = extract_topk_features(img, COLORS,    top_k=3, min_prob=0.12)
    mat_topk_json = [{"label": l, "prob": round(p,4)} for (l,p) in mat_topk]
    col_topk_json = [{"label": l, "prob": round(p,4)} for (l,p) in col_topk]

    # 3) 컨텍스트(H4)
    print("--- 4. Running H4 Context Head ---")
    h4_result = _infer_context_with_h4(img)
    context = h4_result["context"]

    # 4) H3/H2/개별색상/개별재질 실행
    print("--- 5. Running H3 (Classification), H2 (Refinement) & Item-Attributes ---")
    _classify_items_h3_and_refine_h2(items)

    # 5) H1 점수
    print("--- 6. Running H1 (Aesthetic Score) ---")
    h1_score = _score_aesthetic_h1(img)

    # 6) H2 점수
    print("--- 7. Running H2 (Compatibility Score) ---")
    h2_score = _compute_compatibility_score_h2(items)

    # H2 텐서 제거 (JSON 직렬화)
    items_cleaned = []
    for it in items:
        items_cleaned.append(asdict(it, dict_factory=lambda x: {k: v for (k, v) in x if k != 'refined_feature_h2'}))

    # 요약 + Top-K 필드 포함
    summary = AnalysisSummary(
        context_pred=context, context_probs=h4_result["probs"],
        main_material=main_material_global, main_color=main_color_global,
        items=items_cleaned,
        aesthetics_score_h1=h1_score,
        compatibility_score_h2=h2_score,
        main_colors_topk=col_topk_json,
        main_materials_topk=mat_topk_json
    )

    # 7) LLM 호출 (제외)
    print("--- 8. AI Analysis Complete (Skipping LLM Call) ---")
    
    # 8) AI 분석 결과만 반환
    result = {
        "inputs": {"image_path": image_path},
        "analysis": asdict(summary),
        # "fashion_advice" 키가 없습니다.
    }

    print("="*50)
    print("--- E2E Analysis Function Defined (Top-K + Item-Color-Material-Aware) ---")
    return result

print("✅ 'Cell 5' (AI 분석 모듈) 정의 완료.")

# --- 6. (테스트용) 이 파일이 직접 실행될 때만 작동 ---
if __name__ == "__main__":
    print("\n--- [AI 모듈] fashion_model.py 테스트 실행 ---")
    # (Colab/Jupyter에서는 이 부분이 실행되지 않습니다)
    
    # 1. 테스트 이미지 경로
    test_img_path = PROJECT_ROOT / "test_images" / "Fashion_mosaic" / "mosaic_454724667_3051866654978525_3676099934339291567_n..jpg"
    test_img_str = str(test_img_path)
    
    # 2. (glob으로 비슷한 파일 찾기)
    if not test_img_path.exists():
        print(f"Warning: {test_img_path} not found. Trying glob...")
        base_name = "mosaic_454724667_3051866654978525_3676099934339291567_n"
        search_pattern = f"test_images/Fashion_mosaic/{base_name}*.jpg"
        possible_files = list(PROJECT_ROOT.glob(search_pattern))
        if possible_files:
            test_img_path = possible_files[0]
            test_img_str = str(test_img_path)
            print(f"Found: {test_img_str}")
        else:
            raise FileNotFoundError(f"Test image not found: {search_pattern}")

    print(f"Test image: {test_img_str}")

    # 3. AI 분석 실행
    ai_analysis_json = run_ai_analysis(test_img_str)
    
    # 4. 결과 출력
    print("\n--- ✨ 최종 AI 분석 JSON (LLM 제외) ✨ ---")
    print(json.dumps(ai_analysis_json, ensure_ascii=False, indent=2))