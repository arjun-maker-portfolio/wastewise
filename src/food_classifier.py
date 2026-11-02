import os
import logging
from functools import lru_cache
from threading import Lock
from typing import Optional, Tuple

import numpy as np

_LOAD_LOCK = Lock()
_MODEL = None
_LABELS = None
_LOAD_FAILED = False
logger = logging.getLogger(__name__)


def _default_model_paths() -> Tuple[str, str]:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.getenv("FOOD_MODEL_PATH") or os.path.join(project_root, "keras_model.h5")
    labels_path = os.getenv("FOOD_LABELS_PATH") or os.path.join(project_root, "labels.txt")
    return model_path, labels_path


def _load_labels(labels_path: str):
    labels = []
    with open(labels_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            label = line.split(maxsplit=1)[1] if parts[0].isdigit() and len(parts) > 1 else line
            labels.append(label)
    return labels


def _ensure_model():
    global _MODEL, _LABELS
    if _MODEL is not None and _LABELS is not None:
        return

    with _LOAD_LOCK:
        if _MODEL is not None and _LABELS is not None:
            return

        model_path, labels_path = _default_model_paths()
        if not os.path.exists(model_path) or not os.path.isfile(model_path):
            raise FileNotFoundError(f"Keras model not found at {model_path}")
        if not os.path.exists(labels_path) or not os.path.isfile(labels_path):
            raise FileNotFoundError(f"Labels file not found at {labels_path}")

        try:
            from tensorflow.keras.models import load_model
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TensorFlow is required for food classification but is not installed. "
                "Install dependencies from requirements.txt."
            ) from exc

        model = load_model(model_path, compile=False)
        labels = _load_labels(labels_path)
        if model.input_shape[1:] != (224, 224, 3):
            raise ValueError(f"Expected 224x224 RGB input, got {model.input_shape}")
        if not labels or model.output_shape[-1] != len(labels):
            raise ValueError("Model output count does not match labels.txt")
        _MODEL, _LABELS = model, labels


def load_classifier():
    _ensure_model()
    return len(_LABELS)


def _preprocess(frame: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    rgb = frame[:, :, ::-1]
    tensor = tf.convert_to_tensor(rgb, dtype=tf.float32)
    tensor = tf.image.resize(tensor, (224, 224))
    tensor = tensor / 255.0
    tensor = tf.expand_dims(tensor, axis=0)
    return tensor.numpy()


@lru_cache(maxsize=128)
def _label_safe(idx: int) -> Optional[str]:
    if _LABELS is None:
        return None
    if 0 <= idx < len(_LABELS):
        return _LABELS[idx]
    return None


def classify_food(frame: np.ndarray) -> Optional[Tuple[str, float]]:

    global _LOAD_FAILED
    if _LOAD_FAILED:
        return None
    try:
        _ensure_model()
    except Exception:
        _LOAD_FAILED = True
        logger.exception("Food classifier could not load; classification is disabled")
        return None

    if frame is None or frame.size == 0:
        return None

    try:
        batch = _preprocess(frame)
        predictions = _MODEL(batch, training=False).numpy()
    except Exception:
        logger.exception("Food classification failed")
        return None

    if predictions.size == 0 or not np.all(np.isfinite(predictions)):
        logger.error("Food classifier returned empty or non-finite predictions")
        return None

    scores = predictions[0]
    idx = int(np.argmax(scores))
    confidence = float(scores[idx])
    label = _label_safe(idx)
    if label is None:
        return None
    return label, confidence
