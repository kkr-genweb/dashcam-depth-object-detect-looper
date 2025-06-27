from quart import Quart, Response
import cv2
import threading
import time
import os
import numpy as np
from glob import glob

# ----------------- Flask Setup -----------------
app = Quart(__name__)
video_dir = "bdd100k/videos/train/"
#video_dir = "/mnt/bdd-tiny/kaggle/videos-sample-1k/bdd100k_videos_train_00/bdd100k/videos/train"
video_files = sorted(glob(os.path.join(video_dir, "*.mp4")) + glob(os.path.join(video_dir, "*.mov")))

video_index = 0
cap = None
frame_lock = threading.Lock()
latest_raw_frame = None
latest_processed_frame = None

# ----------------- DNN Setup -----------------
prototxt_path = "deploy.prototxt"
model_path = "mobilenet_iter_73000.caffemodel"
CONFIDENCE_THRESHOLD = 0.5

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

try:
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
except cv2.error as e:
    print(f"[ERROR] Could not load model: {e}")
    exit()

# ----------------- Streaming Loop -----------------
def video_stream_loop():
    global video_index, cap, latest_raw_frame, latest_processed_frame

    while True:
        if not video_files:
            print("No video files found.")
            time.sleep(5)
            continue

        cap = cv2.VideoCapture(video_files[video_index])
        if not cap.isOpened():
            print(f"Cannot open {video_files[video_index]}")
            video_index = (video_index + 1) % len(video_files)
            continue

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Processed frame copy
            processed = frame.copy()
            (h, w) = processed.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(processed, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            confidence_stats = []

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > CONFIDENCE_THRESHOLD:
                    idx = int(detections[0, 0, i, 1])
                    class_label = CLASSES[idx]
                    if class_label in ["person", "car"]:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        label = f"{class_label.capitalize()}: {confidence:.2%}"

                        confidence_stats.append(label)
                        cv2.rectangle(processed, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(processed, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Overlay summary in top-left
            summary_lines = []
            person_count = sum("Person" in s for s in confidence_stats)
            car_count = sum("Car" in s for s in confidence_stats)

            summary_lines.append(f"#Persons detected: {person_count}")
            summary_lines.append(f"#Cars detected: {car_count}")

            for i, label in enumerate(confidence_stats[:5]):
                summary_lines.append(f"{i+1}. {label}")

            for i, text in enumerate(summary_lines):
                y_pos = 30 + i * 25
                cv2.putText(processed, text, (15, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 255), 2)

            # Store both versions
            _, raw_buf = cv2.imencode('.jpg', frame)
            _, proc_buf = cv2.imencode('.jpg', processed)

            with frame_lock:
                latest_raw_frame = raw_buf.tobytes() if raw_buf is not None else None
                latest_processed_frame = proc_buf.tobytes() if proc_buf is not None else None

            time.sleep(1 / 24)

        cap.release()
        video_index = (video_index + 1) % len(video_files)

# ----------------- Stream Generators -----------------
def stream_generator(processed=False):
    while True:
        with frame_lock:
            frame = latest_processed_frame if processed else latest_raw_frame
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(1 / 24)

# ----------------- Routes -----------------
@app.route('/video')
def stream_raw():
    return Response(stream_generator(processed=False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video/processed')
def stream_processed():
    return Response(stream_generator(processed=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ----------------- Main -----------------
if __name__ == '__main__':
    if not video_files:
        print("No video files found in directory.")
    else:
        print(f"Starting stream server with {len(video_files)} video files.")
        t = threading.Thread(target=video_stream_loop, daemon=True)
        t.start()
        app.run(host='0.0.0.0', port=8085, debug=False, threaded=True)