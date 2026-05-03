"""
cf_model.py
Train Collaborative Filtering model dùng ALS (implicit library).
Lưu model để dùng cho real-time scoring.
"""
import os
import pickle
import json
from datetime import datetime, timezone
import numpy as np
import implicit
from scipy.sparse import csr_matrix
from typing import Dict, Optional, List, Tuple

MODEL_PATH = os.getenv("CF_MODEL_PATH", "app/models/saved/cf_model.pkl")
INDEX_PATH = os.getenv("CF_INDEX_PATH", "app/models/saved/cf_index.pkl")
METADATA_PATH = os.getenv("CF_META_PATH", "app/models/saved/cf_model_meta.json")

# ALS hyperparameters
ALS_FACTORS = 50        # số latent factors
ALS_ITERATIONS = 20     # số vòng lặp train
ALS_REGULARIZATION = 0.1

_model = None
_user_index = {}
_item_index = {}
_item_index_reverse = {}  # {col_index: productId}


def train_model(
    user_item_matrix: csr_matrix,
    user_index: Dict[str, int],
    item_index: Dict[str, int]
) -> implicit.als.AlternatingLeastSquares:
    """
    Train ALS model từ user-item matrix.
    Lưu model và index vào disk.
    """
    print(f"[CF] Training ALS model: {user_item_matrix.shape[0]} users, "
          f"{user_item_matrix.shape[1]} items")

    model = implicit.als.AlternatingLeastSquares(
        factors=ALS_FACTORS,
        iterations=ALS_ITERATIONS,
        regularization=ALS_REGULARIZATION,
        use_gpu=False,  # CPU only cho capstone
    )

    # implicit expects item-user matrix (transpose)
    item_user_matrix = user_item_matrix.T.tocsr()
    model.fit(item_user_matrix)

    # Lưu model và index (atomic)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Atomic write model
    tmp_model = MODEL_PATH + ".tmp"
    with open(tmp_model, "wb") as f:
        pickle.dump(model, f)
    os.replace(tmp_model, MODEL_PATH)

    # Atomic write index
    tmp_index = INDEX_PATH + ".tmp"
    with open(tmp_index, "wb") as f:
        pickle.dump({
            "user_index": user_index,
            "item_index": item_index,
            "item_index_reverse": {v: k for k, v in item_index.items()}
        }, f)
    os.replace(tmp_index, INDEX_PATH)

    # Write metadata for easier inspection (atomic)
    meta = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "num_users": len(user_index),
        "num_items": len(item_index),
        "num_interactions": int(user_item_matrix.nnz),
        "model_path": MODEL_PATH,
        "index_path": INDEX_PATH,
        "source_db": os.getenv("MONGO_URI", ""),
    }
    tmp_meta = METADATA_PATH + ".tmp"
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp_meta, METADATA_PATH)

    print(f"[CF] Model saved to {MODEL_PATH} (users={len(user_index)}, items={len(item_index)})")
    return model


def load_model() -> bool:
    """Load model từ disk vào memory. Trả về True nếu thành công."""
    global _model, _user_index, _item_index, _item_index_reverse
    if not os.path.exists(MODEL_PATH) or not os.path.exists(INDEX_PATH):
        print("[CF] No saved model found — CF scoring disabled")
        return False
    try:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        with open(INDEX_PATH, "rb") as f:
            idx = pickle.load(f)
            _user_index = idx["user_index"]
            _item_index = idx["item_index"]
            _item_index_reverse = idx["item_index_reverse"]
        # Try to read metadata if present
        meta_info = None
        try:
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, "r", encoding="utf-8") as mf:
                    meta_info = json.load(mf)
        except Exception:
            meta_info = None

        if meta_info:
            print(f"[CF] Model loaded: {meta_info.get('num_users')} users, {meta_info.get('num_items')} items (saved_at={meta_info.get('saved_at')})")
        else:
            print(f"[CF] Model loaded: {len(_user_index)} users, {len(_item_index)} items")
        return True
    except Exception as e:
        print(f"[CF] Failed to load model: {e}")
        return False


def get_cf_scores(user_id: str, product_ids: List[str]) -> Dict[str, float]:
    """
    Tính CF score cho danh sách sản phẩm với user cụ thể.
    Trả về {productId: cf_score (0.0-1.0)}
    Nếu user chưa có trong model → trả về {} (CF score = 0 cho tất cả)
    """
    if _model is None or user_id not in _user_index:
        return {}

    user_row = _user_index[user_id]

    # Lấy scores cho tất cả items từ model
    # implicit trả về (item_ids, scores) sorted by score
    try:
        # Dùng model.recommend để lấy scores
        item_user_matrix_T = None  # cần lưu lại khi train
        # Thay thế: tính trực tiếp từ user/item factors
        user_factor = _model.user_factors[user_row]

        scores = {}
        max_score = 0.0
        for pid in product_ids:
            if pid in _item_index:
                item_row = _item_index[pid]
                item_factor = _model.item_factors[item_row]
                raw_score = float(np.dot(user_factor, item_factor))
                scores[pid] = raw_score
                max_score = max(max_score, raw_score)

        # Normalize về 0-1
        if max_score > 0:
            scores = {pid: s / max_score for pid, s in scores.items()}

        return scores
    except Exception as e:
        print(f"[CF] Error computing scores: {e}")
        return {}
