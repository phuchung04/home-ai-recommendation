"""
training.py
Endpoint để trigger CF model training thủ công (cho demo capstone).
"""
from fastapi import APIRouter
from app.services.behavior_service import load_behavior_matrix
from app.services.cf_model import train_model, load_model

router = APIRouter()

@router.post("/train")
async def trigger_training():
    """
    Trigger CF model training thủ công.
    Đọc behavior events từ MongoDB → train ALS → lưu model.
    """
    try:
        # Load behavior data
        matrix, user_index, item_index = await load_behavior_matrix()

        if matrix is None or matrix.nnz == 0:
            return {
                "status": "skipped",
                "message": "Không đủ dữ liệu behavior để train (cần ít nhất 1 event)"
            }

        # Train model
        model = train_model(matrix, user_index, item_index)

        # Reload model vào memory
        load_model()

        return {
            "status": "success",
            "message": "CF model trained successfully",
            "stats": {
                "num_users": len(user_index),
                "num_items": len(item_index),
                "num_interactions": matrix.nnz,
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/model-status")
def model_status():
    """Kiểm tra trạng thái model hiện tại."""
    import os
    from app.services.cf_model import MODEL_PATH, _model, _user_index, _item_index
    return {
        "model_loaded": _model is not None,
        "model_file_exists": os.path.exists(MODEL_PATH),
        "num_users": len(_user_index),
        "num_items": len(_item_index),
    }
