from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Load YOLOv8n COCO model once at startup
model = YOLO("static/yolov8n.pt")

# Open default camera (0); on Render you will later change this to a video/IP stream
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Run YOLOv8 inference
        results = model(frame, imgsz=640, conf=0.5)

        # results[0].boxes.xywh gives (cx, cy, w, h) for each detection in pixels
        boxes_xywh = results[0].boxes.xywh.cpu().tolist()
        classes = results[0].boxes.cls.cpu().tolist()
        scores = results[0].boxes.conf.cpu().tolist()

        for (cx, cy, w, h), cls_id, score in zip(boxes_xywh, classes, scores):
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            label = model.names[int(cls_id)]

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (34, 197, 94), 2)

            # Prepare text: name + dimensions in pixels
            dim_text = f"{label} {score:.2f} | {int(w)}x{int(h)} px"
            (tw, th), _ = cv2.getTextSize(dim_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (34, 197, 94), -1)
            cv2.putText(
                frame,
                dim_text,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Encode annotated frame as JPEG and stream
        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
