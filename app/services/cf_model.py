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
import asyncio

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
    # Check presence of files first
    model_exists = os.path.exists(MODEL_PATH)
    index_exists = os.path.exists(INDEX_PATH)
    if not model_exists and not index_exists:
        print("[CF] No saved model found — CF scoring disabled")
        return False

    # Try to load model and index separately to get precise errors
    load_ok = True
    # Load model
    if model_exists:
        try:
            with open(MODEL_PATH, "rb") as f:
                _model = pickle.load(f)
        except Exception as e:
            load_ok = False
            print(f"[CF] Failed to load model file '{MODEL_PATH}': {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[CF] Model file not found at {MODEL_PATH}")
        load_ok = False

    # Load index
    if index_exists:
        try:
            with open(INDEX_PATH, "rb") as f:
                idx = pickle.load(f)
                _user_index = idx.get("user_index", {})
                _item_index = idx.get("item_index", {})
                _item_index_reverse = idx.get("item_index_reverse", {})
        except Exception as e:
            load_ok = False
            print(f"[CF] Failed to load index file '{INDEX_PATH}': {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[CF] Index file not found at {INDEX_PATH}")
        load_ok = False

    # Validation: prefer metadata as source-of-truth (num_users, num_items)
    meta_info = None
    try:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r", encoding="utf-8") as mf:
                meta_info = json.load(mf)
    except Exception:
        meta_info = None

    if load_ok:
        actual_users = len(_user_index)
        actual_items = len(_item_index)

        # Use metadata if available for expected counts, otherwise fallback to model shapes
        if meta_info:
            expected_users = int(meta_info.get("num_users", 0))
            expected_items = int(meta_info.get("num_items", 0))
        elif _model is not None:
            expected_users = _model.user_factors.shape[0]
            expected_items = _model.item_factors.shape[0]
        else:
            expected_users = actual_users
            expected_items = actual_items

        # If actual matches expected, we're good
        if actual_users == expected_users and actual_items == expected_items:
            pass
        # If they are swapped, auto-correct by swapping mappings
        elif actual_users == expected_items and actual_items == expected_users:
            print("[CF] Detected swapped index/model dimensions relative to metadata. Auto-correcting indices.")
            _user_index, _item_index = _item_index, _user_index
            _item_index_reverse = {v: k for k, v in _item_index.items()}
            print(f"[CF] Swap complete: users={len(_user_index)}, items={len(_item_index)}")
        else:
            load_ok = False
            print(f"[CF] MISMATCH: Index has {actual_users} users (expected {expected_users}), {actual_items} items (expected {expected_items})")
            print("[CF] Index and model are out of sync. Recommend re-training.")
            _model = None
            _user_index = {}
            _item_index = {}
            _item_index_reverse = {}
        # Additional safety: if model factor matrices appear swapped relative to indices,
        # correct by swapping the factor arrays in-memory. This handles cases where a
        # previously saved model had user/item factors reversed while the index counts
        # remained correct (causing dimension mismatch at runtime).
        try:
            if _model is not None and len(_user_index) and len(_item_index):
                mu = _model.user_factors.shape[0]
                mi = _model.item_factors.shape[0]
                if mu == actual_items and mi == actual_users:
                    print("[CF] Detected model factor axes swapped relative to indices. Correcting by swapping factor matrices.")
                    # swap arrays
                    tmp_user = _model.user_factors.copy()
                    _model.user_factors = _model.item_factors.copy()
                    _model.item_factors = tmp_user
                    print(f"[CF] Swap complete: model.user_factors.shape={_model.user_factors.shape}, model.item_factors.shape={_model.item_factors.shape}")
        except Exception as e:
            print(f"[CF] Warning: failed to auto-correct swapped model factors: {e}")
        # Try to read metadata if present
        meta_info = None
        try:
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, "r", encoding="utf-8") as mf:
                    meta_info = json.load(mf)
        except Exception:
            meta_info = None

    # Report summary
    try:
        if meta_info:
            print(f"[CF] Model metadata: {meta_info.get('num_users')} users, {meta_info.get('num_items')} items (saved_at={meta_info.get('saved_at')})")
        else:
            print(f"[CF] Model metadata not available")
    except Exception:
        pass

    if load_ok:
        print(f"[CF] Model and index loaded: users={len(_user_index)}, items={len(_item_index)}")
    else:
        print("[CF] Model loading incomplete or failed — CF scoring disabled until fixed")

    return load_ok


def get_cf_scores(user_id: str, product_ids: List[str]) -> Dict[str, float]:
    """
    Tính CF score cho danh sách sản phẩm với user cụ thể.
    Trả về {productId: cf_score (0.0-1.0)}
    Nếu user chưa có trong model → trả về {} (CF score = 0 cho tất cả)
    """
    if _model is None or user_id not in _user_index:
        return {}

    user_row = _user_index[user_id]
    
    # Safety check: ensure user_row is within bounds
    if user_row >= _model.user_factors.shape[0]:
        print(f"[CF] Warning: user_row {user_row} out of bounds (model has {_model.user_factors.shape[0]} users)")
        return {}

    # Lấy scores cho tất cả items từ model
    try:
        # Determine whether model factors align with indices or are swapped
        num_user_factors = _model.user_factors.shape[0]
        num_item_factors = _model.item_factors.shape[0]
        len_user_index = len(_user_index)
        len_item_index = len(_item_index)

        swapped = False
        if num_user_factors == len_user_index and num_item_factors == len_item_index:
            swapped = False
        elif num_user_factors == len_item_index and num_item_factors == len_user_index:
            swapped = True
        else:
            # Unknown configuration — attempt best-effort but warn
            print(f"[CF] Warning: unexpected factor dimensions (user_factors={num_user_factors}, item_factors={num_item_factors}), index sizes (users={len_user_index}, items={len_item_index})")

        scores = {}
        max_score = 0.0

        for pid in product_ids:
            if pid in _item_index:
                item_row = _item_index[pid]
                # Ensure we don't index out of bounds
                try:
                    if not swapped:
                        user_factor = _model.user_factors[user_row]
                        item_factor = _model.item_factors[item_row]
                    else:
                        # model.user_factors are actually items, model.item_factors are users
                        user_factor = _model.item_factors[user_row]
                        item_factor = _model.user_factors[item_row]

                    raw_score = float(np.dot(user_factor, item_factor))
                    scores[pid] = raw_score
                    if raw_score > max_score:
                        max_score = raw_score
                except IndexError:
                    # Skip items that are out-of-bounds for the loaded model
                    print(f"[CF] Skipping pid={pid}: mapped row out of bounds (item_row={item_row}, user_row={user_row})")
                    continue

        # Normalize về 0-1
        if max_score > 0:
            scores = {pid: s / max_score for pid, s in scores.items()}

        return scores
    except Exception as e:
        print(f"[CF] Error computing scores: {e}")
        import traceback
        traceback.print_exc()
        return {}


def get_cf_raw_scores(user_id: str, product_ids: List[str]) -> Dict[str, float]:
    """Return raw (unnormalized) dot-product CF scores for given user and products.
    Handles swapped user/item factor axes similarly to `get_cf_scores`.
    """
    if _model is None or user_id not in _user_index:
        return {}

    user_row = _user_index[user_id]

    try:
        num_user_factors = _model.user_factors.shape[0]
        num_item_factors = _model.item_factors.shape[0]
        len_user_index = len(_user_index)
        len_item_index = len(_item_index)

        swapped = False
        if num_user_factors == len_user_index and num_item_factors == len_item_index:
            swapped = False
        elif num_user_factors == len_item_index and num_item_factors == len_user_index:
            swapped = True
        else:
            print(f"[CF] Warning: unexpected factor dimensions in raw score calc")

        raw_scores = {}
        for pid in product_ids:
            if pid in _item_index:
                item_row = _item_index[pid]
                try:
                    if not swapped:
                        user_factor = _model.user_factors[user_row]
                        item_factor = _model.item_factors[item_row]
                    else:
                        user_factor = _model.item_factors[user_row]
                        item_factor = _model.user_factors[item_row]

                    raw = float(np.dot(user_factor, item_factor))
                    raw_scores[pid] = raw
                except IndexError:
                    continue

        return raw_scores
    except Exception as e:
        print(f"[CF] Error computing raw scores: {e}")
        import traceback
        traceback.print_exc()
        return {}


async def auto_train_on_startup():
    """
    Auto-train CF model on application startup.
    - Load behavior matrix, train model in thread if data exists, then reload model.
    """
    try:
        print("[CF] Auto-training on startup...")
        from app.services.behavior_service import load_behavior_matrix

        matrix, user_index, item_index = await load_behavior_matrix()

        if matrix is None or matrix.nnz == 0:
            print("[CF] No behavior data available — skipping auto-train")
            return

        await asyncio.to_thread(train_model, matrix, user_index, item_index)
        load_model()
        print("[CF] Auto-training completed")
    except Exception as e:
        print(f"[CF] Auto-training failed: {e}")


async def get_user_behavior_count(user_id: str) -> int:
    """
    Helper to return the number of behavior events for a user.
    Delegates to behavior_service to avoid direct cyclic imports elsewhere.
    """
    if not user_id:
        return 0
    try:
        from app.services.behavior_service import get_user_event_count

        return await get_user_event_count(user_id)
    except Exception:
        return 0
