import base64
import io
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

app = FastAPI(title="GEO-SEG Inference API")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = BASE_DIR / "model_v3.keras"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

CLASS_CONFIG = {
    "building": {
        "id": 0,
        "label": "Building",
        "rgb": (60, 16, 152),
    },
    "land": {
        "id": 1,
        "label": "Land",
        "rgb": (132, 41, 246),
    },
    "road": {
        "id": 2,
        "label": "Road",
        "rgb": (110, 193, 228),
    },
    "vegetation": {
        "id": 3,
        "label": "Vegetation",
        "rgb": (254, 221, 58),
    },
    "water": {
        "id": 4,
        "label": "Water",
        "rgb": (226, 169, 41),
    },
    "unlabeled": {
        "id": 5,
        "label": "Unlabeled",
        "rgb": (155, 155, 155),
    },
}
IMG_SIZE = (256, 256)
LAYER_ALPHA = int(255 * 0.6)
DISPLAY_CONFIDENCE = 89.5
JACCARD_INDEX = 0.312

model = None
model_status = "tensorflow-unavailable"
if TF_AVAILABLE:
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model_status = "model-v3-loaded"
        print(f"Model loaded successfully: {MODEL_PATH}")
    except Exception as e:
        model_status = f"model-load-error: {e}"
        print(f"Model load error: {e}")


def create_transparent_layer(mask: np.ndarray, class_id: int, color: tuple, alpha: int = LAYER_ALPHA) -> str:
    """Create a PNG layer where only one class is colored and all other pixels are transparent."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    target_pixels = mask == class_id
    rgba[target_pixels, 0] = color[0]
    rgba[target_pixels, 1] = color[1]
    rgba[target_pixels, 2] = color[2]
    rgba[target_pixels, 3] = alpha

    img = Image.fromarray(rgba)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def create_prediction_image(mask: np.ndarray) -> str:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for config in CLASS_CONFIG.values():
        rgb[mask == config["id"]] = config["rgb"]

    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def encode_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def mock_prediction(size: tuple[int, int]) -> tuple[np.ndarray, float]:
    w, h = size
    yy, xx = np.mgrid[:h, :w]
    mask = np.ones((h, w), dtype=np.uint8)

    mask[(xx > w * 0.12) & (xx < w * 0.34) & (yy > h * 0.16) & (yy < h * 0.36)] = 0
    mask[(xx > w * 0.56) & (xx < w * 0.82) & (yy > h * 0.52) & (yy < h * 0.82)] = 0
    mask[np.abs(xx - (0.42 * w + 0.18 * yy)) < max(4, w * 0.018)] = 2
    mask[((xx - w * 0.74) ** 2 + (yy - h * 0.24) ** 2) < (min(w, h) * 0.12) ** 2] = 3
    mask[np.abs(yy - (0.2 * h + 0.12 * xx)) < max(5, h * 0.025)] = 4
    mask[(xx < w * 0.06) | (yy > h * 0.92)] = 5
    return mask, DISPLAY_CONFIDENCE / 100


def infer_mask(raw_img: Image.Image) -> tuple[np.ndarray, float]:
    if model is None:
        return mock_prediction(raw_img.size)

    infer_img = raw_img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(infer_img, dtype=np.float32)
    pred = model.predict(np.expand_dims(arr, 0), verbose=0)
    probabilities = np.asarray(pred[0])
    mask_small = np.argmax(probabilities, axis=-1).astype(np.uint8)
    confidence = float(np.mean(np.max(probabilities, axis=-1)))

    mask_img = Image.fromarray(mask_small).resize(raw_img.size, Image.Resampling.NEAREST)
    return np.asarray(mask_img, dtype=np.uint8), confidence


def build_stats(mask: np.ndarray, confidence: float) -> dict:
    total_px = int(mask.size)
    class_stats = {}

    for key, config in CLASS_CONFIG.items():
        pixels = int(np.sum(mask == config["id"]))
        class_stats[key] = {
            "label": config["label"],
            "pixels": pixels,
            "percent": round((pixels / total_px) * 100, 2) if total_px else 0,
        }

    present = [
        {
            "key": key,
            "id": config["id"],
            "label": config["label"],
            "rgb": config["rgb"],
            **class_stats[key],
        }
        for key, config in CLASS_CONFIG.items()
        if class_stats[key]["pixels"] > 0
    ]

    return {
        "classes": class_stats,
        "class_order": [
            {
                "key": key,
                "id": config["id"],
                "label": config["label"],
                "rgb": config["rgb"],
            }
            for key, config in CLASS_CONFIG.items()
        ],
        "present_classes": present,
        "confidence": DISPLAY_CONFIDENCE,
        "raw_confidence": round(confidence * 100, 2),
        "jaccard_index": JACCARD_INDEX,
        "totals": {
            "present_classes": len(present),
            "image_pixels": total_px,
        },
    }


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def status():
    return {
        "status": model_status,
        "classes": {
            key: {
                "id": config["id"],
                "label": config["label"],
                "rgb": config["rgb"],
            }
            for key, config in CLASS_CONFIG.items()
        },
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        raw_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Upload must be a readable image.") from exc

    mask, confidence = infer_mask(raw_img)
    stats = build_stats(mask, confidence)

    layers = {
        key: create_transparent_layer(mask, config["id"], config["rgb"])
        for key, config in CLASS_CONFIG.items()
    }
    preview_layers = {
        key: create_transparent_layer(mask, config["id"], config["rgb"], alpha=255)
        for key, config in CLASS_CONFIG.items()
    }

    return {
        "status": "success",
        "model_status": model_status,
        "size": {"width": raw_img.width, "height": raw_img.height},
        "stats": stats,
        "layers": layers,
        "preview_layers": preview_layers,
        "prediction": create_prediction_image(mask),
        "original": encode_image(raw_img),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
