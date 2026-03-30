import numpy as np
import io

_face_app = None
FACE_AVAILABLE = False

try:
    from insightface.app import FaceAnalysis
    import cv2

    _face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    _face_app.prepare(ctx_id=0, det_size=(640, 640))
    FACE_AVAILABLE = True
except Exception:
    pass

# InsightFace produces 512-d float32 embeddings (2048 bytes).
# Old dlib/face_recognition produced 128-d float64 (1024 bytes).
_ENCODING_BYTES = 512 * 4  # 2048


def encode_face(image_bytes):
    """Extract 512-d face embedding from image bytes using ArcFace."""
    if not FACE_AVAILABLE:
        return None
    try:
        img_array = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR
        )
        if img_array is None:
            return None
        faces = _face_app.get(img_array)
        if faces:
            return faces[0].embedding  # 512-d float32
        return None
    except Exception:
        return None


def match_face(unknown_encoding, face_data, threshold=0.4):
    """Match using cosine similarity. face_data: list of (child_id, encoding_bytes).
    Returns (child_id, similarity) or (None, None).
    Threshold 0.4 = standard ArcFace cutoff for same person."""
    if not FACE_AVAILABLE or unknown_encoding is None or not face_data:
        return None, None
    try:
        child_ids = []
        known = []
        for cid, blob in face_data:
            # Only use InsightFace-format encodings (2048 bytes), skip old dlib ones
            if len(blob) == _ENCODING_BYTES:
                child_ids.append(cid)
                known.append(np.frombuffer(blob, dtype=np.float32))

        if not known:
            return None, None

        unknown_norm = unknown_encoding / np.linalg.norm(unknown_encoding)
        similarities = []
        for k in known:
            k_norm = k / np.linalg.norm(k)
            similarities.append(float(np.dot(unknown_norm, k_norm)))

        idx = int(np.argmax(similarities))
        if similarities[idx] >= threshold:
            return child_ids[idx], similarities[idx]
    except Exception:
        pass
    return None, None
