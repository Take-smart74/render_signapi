# main_cam.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import numpy as np
import cv2
import os
import torch
import uvicorn

app = FastAPI(title="Traffic Sign Detection API", version="1.0.0")

# CORS：本番は必要なドメインに絞る
ALLOWED = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- モデルは起動時に一度だけロード ---
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH).to(DEVICE)

# （任意）CUDA時は半精度を試す
if DEVICE == "cuda":
    try:
        model.model.half()
    except Exception:
        pass

class Detection(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    class_id: int
    class_name_en: str

class PredictResponse(BaseModel):
    detections: list[Detection]
    image_w: int
    image_h: int
    num_detections: int

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "names": model.names}

@app.post("/detect", response_model=PredictResponse)
async def detect(
    # ★ Streamlit側に合わせて "image" という名前で受ける
    image: UploadFile = File(...),
    conf: float = Form(0.4),
    imgsz: int = Form(640),
):
    try:
        file_bytes = await image.read()
        arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return PredictResponse(detections=[], image_w=0, image_h=0, num_detections=0)

        h, w = img.shape[:2]

        # 推論（no_grad で少しだけ効率化）
        with torch.no_grad():
            # Ultralytics v8: 1枚入力なら [0] でOK
            res = model.predict(
                img, conf=conf, imgsz=imgsz, verbose=False, device=DEVICE
            )[0]

        dets: list[Detection] = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy
            cls = res.boxes.cls
            confs = res.boxes.conf

            # CPUへ
            xyxy = xyxy.to("cpu").numpy()
            cls = cls.to("cpu").numpy()
            confs = confs.to("cpu").numpy()

            names = model.names  # {id: "name"}
            for (x1, y1, x2, y2), c, p in zip(xyxy, cls, confs):
                cid = int(c)
                dets.append(
                    Detection(
                        x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                        conf=float(p), class_id=cid,
                        class_name_en=names.get(cid, "unknown"),
                    )
                )

        return PredictResponse(
            detections=dets, image_w=w, image_h=h, num_detections=len(dets)
        )
    except Exception:
        # 何か起きても落とさず空で返す
        return PredictResponse(detections=[], image_w=0, image_h=0, num_detections=0)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
