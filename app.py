# app.py
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import eventlet
import os

eventlet.monkey_patch()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Load YOLO model
model = YOLO("yolov8n.pt")  # keep small model for free CPU hosting

@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("frame")
def handle_frame(data):
    try:
        header, encoded = data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            emit("error", {"msg": "Invalid image"})
            return

        h, w = img.shape[:2]

        # Run detection
        results = model.predict(img, imgsz=640, conf=0.25, verbose=False)

        detections = []
        r = results[0]
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]

            detections.append({
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "w": x2 - x1,
                "h": y2 - y1,
                "conf": conf,
                "name": name
            })

        emit("detections", {"detections": detections})

    except Exception as e:
        emit("error", {"msg": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)
