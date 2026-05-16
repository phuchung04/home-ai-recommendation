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

    # Đọc tất cả events và normalize id thành string để tránh type-mismatch
    events = []
    async for doc in col.find({}):
        # Normalize stored ids to string to match product `_id` stringification elsewhere
        norm = {
            "userId": str(doc.get("userId")),
            "productId": str(doc.get("productId")),
            "eventType": doc.get("eventType", "PRODUCT_VIEW"),
            "rating": doc.get("rating") if "rating" in doc else None,
        }
        events.append(norm)

    if not events:
        return None, {}, {}

    # Tạo index mapping (string ids)
    user_ids = list({e["userId"] for e in events})
    product_ids = list({e["productId"] for e in events})
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

        # Defensive: skip events whose ids are not in the computed indices
        if uid not in user_index or pid not in item_index:
            continue

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


async def get_user_event_count(user_id: str) -> int:
    """
    Return the total number of behavior events for a given user_id.
    """
    if not user_id:
        return 0
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    col = db[BEHAVIOR_COLLECTION]

    # Normalize to string for stored ids
    count = await col.count_documents({"userId": str(user_id)})
    return int(count)


async def get_popular_product_scores(product_ids: list) -> Dict[str, int]:
    """
    Aggregate global event counts for the provided product_ids.
    Returns a mapping {productId: count}.
    """
    if not product_ids:
        return {}
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    col = db[BEHAVIOR_COLLECTION]

    pipeline = [
        {"$match": {"productId": {"$in": [str(p) for p in product_ids]}}},
        {"$group": {"_id": "$productId", "count": {"$sum": 1}}},
    ]
    cursor = col.aggregate(pipeline)
    result = {}
    async for doc in cursor:
        pid = str(doc.get("_id"))
        result[pid] = int(doc.get("count", 0))

    # Ensure all requested ids present (missing => 0)
    for pid in product_ids:
        result.setdefault(str(pid), 0)

    return result
