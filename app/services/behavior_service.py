"""
behavior_service.py
Đọc UserBehaviorEvent từ MongoDB và tạo user-item matrix cho CF training.
"""
import os
import numpy as np
from scipy.sparse import csr_matrix
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Tuple, Dict

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "cap2")
BEHAVIOR_COLLECTION = "user_behavior_events"

# Implicit score mapping
EVENT_SCORES = {
    "PRODUCT_VIEW": 1.0,
    "ADD_TO_CART": 3.0,
    "PURCHASE": 5.0,
    "RATING": None,  # dùng rating value trực tiếp
}

async def load_behavior_matrix() -> Tuple[csr_matrix, Dict, Dict]:
    """
    Đọc tất cả behavior events từ MongoDB.
    Trả về:
    - user_item_matrix: scipy sparse matrix (users x items)
    - user_index: {userId: row_index}
    - item_index: {productId: col_index}
    """
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    col = db[BEHAVIOR_COLLECTION]

    # Đọc tất cả events
    events = []
    async for doc in col.find({}):
        events.append(doc)

    if not events:
        return None, {}, {}

    # Tạo index mapping
    user_ids = list(set(e["userId"] for e in events))
    product_ids = list(set(e["productId"] for e in events))
    user_index = {uid: i for i, uid in enumerate(user_ids)}
    item_index = {pid: i for i, pid in enumerate(product_ids)}

    # Tính implicit score cho mỗi (user, product) pair
    # Nếu có nhiều events cùng pair → cộng dồn score
    score_dict = {}
    for e in events:
        uid = e["userId"]
        pid = e["productId"]
        event_type = e.get("eventType", "PRODUCT_VIEW")

        if event_type == "RATING":
            score = float(e.get("rating", 3))
        else:
            score = EVENT_SCORES.get(event_type, 1.0)

        key = (user_index[uid], item_index[pid])
        score_dict[key] = score_dict.get(key, 0) + score

    # Tạo sparse matrix
    rows, cols, data = [], [], []
    for (r, c), s in score_dict.items():
        rows.append(r)
        cols.append(c)
        data.append(s)

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(product_ids))
    )

    return matrix, user_index, item_index
