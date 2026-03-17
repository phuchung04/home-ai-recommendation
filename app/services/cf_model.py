"""
cf_model.py
Train Collaborative Filtering model dùng ALS (implicit library).
Lưu model để dùng cho real-time scoring.
"""
import os
import pickle
import numpy as np
import implicit
from scipy.sparse import csr_matrix
from typing import Dict, Optional, List, Tuple

MODEL_PATH = os.getenv("CF_MODEL_PATH", "app/models/saved/cf_model.pkl")
INDEX_PATH = os.getenv("CF_INDEX_PATH", "app/models/saved/cf_index.pkl")

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

    # Lưu model và index
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({
            "user_index": user_index,
            "item_index": item_index,
            "item_index_reverse": {v: k for k, v in item_index.items()}
        }, f)

    print(f"[CF] Model saved to {MODEL_PATH}")
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
