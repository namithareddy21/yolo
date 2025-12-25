from flask import Flask, render_template, Response
import cv2
import torch

app = Flask(__name__)

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def generate_frames():
    cap = cv2.VideoCapture(0)  # Webcam; replace with video path if needed
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)

        # Count objects by class
        class_counts = {}
        for result in results.xyxy[0]:
            cls_idx = int(result[5])
            class_name = results.names[cls_idx]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # Draw bounding box
            xmin, ymin, xmax, ymax = map(int, result[:4])
            confidence = float(result[4])
            label = f'{class_name} {confidence:.2f}'
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            cv2.putText(frame, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Prepare description string
        description = "Detected objects:\n"
        for obj, count in class_counts.items():
            description += f"{obj}: {count}  "

        # Overlay the object count/description at the top
        y0, dy = 30, 20
        for i, line in enumerate(description.split("\n")):
            y = y0 + i*dy
            cv2.putText(frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
