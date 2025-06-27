import asyncio
from quart import Quart, Response
import cv2
import os
import numpy as np
from glob import glob
from transformers import pipeline
from PIL import Image

# ----------------- App Setup -----------------
app = Quart(__name__)

# --- CONFIGURATION (Use absolute paths for reliability) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

video_dir = os.path.join(BASE_DIR, "bdd100k/videos/train/")
prototxt_path = os.path.join(BASE_DIR, "deploy.prototxt")
model_path = os.path.join(BASE_DIR, "mobilenet_iter_73000.caffemodel")

# --- Global State ---
video_files = sorted(glob(os.path.join(video_dir, "*.mp4")) + glob(os.path.join(video_dir, "*.mov")))
video_index = 0
frame_lock = asyncio.Lock() # Use asyncio.Lock for async compatibility
latest_raw_frame = None
latest_processed_frame = None
latest_depth_frame = None # New frame for the depth map

# ----------------- AI Model Setup -----------------
# --- Object Detector (Caffe) ---
CONFIDENCE_THRESHOLD = 0.5
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
try:
    if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Object detection model files not found.")
    object_detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    print("[INFO] Object detection model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Could not initialize object detector: {e}")
    exit()

# --- Depth Estimator (Transformers) ---
try:
    print("[INFO] Initializing depth estimation pipeline... (This may take a moment)")
    # Using the small, faster model. Other options: V2-Base, V2-Large
    depth_estimator = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    print("[INFO] Depth estimation pipeline initialized successfully.")
except Exception as e:
    print(f"[ERROR] Could not initialize depth estimator pipeline: {e}")
    print("[INFO] Please ensure you have PyTorch, transformers, and accelerate installed: pip install torch transformers accelerate")
    exit()


# ----------------- Main Processing Loop (Async) -----------------
async def video_processing_loop():
    global video_index, latest_raw_frame, latest_processed_frame, latest_depth_frame

    while True:
        if not video_files:
            print("No video files found. Retrying in 5 seconds.")
            await asyncio.sleep(5)
            continue

        video_path = video_files[video_index]
        print(f"\n[INFO] Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open {video_path}")
            video_index = (video_index + 1) % len(video_files)
            await asyncio.sleep(2)
            continue

        while True:
            success, frame = cap.read()
            if not success:
                print(f"[INFO] End of video {video_path}. Moving to next.")
                break
            
            # --- 1. Pre-process Frame ---
            raw_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # --- 2. Object Detection ---
            processed_frame = raw_frame.copy()
            (h, w) = processed_frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(processed_frame, (300, 300)), 0.007843, (300, 300), 127.5)
            object_detector.setInput(blob)
            detections = object_detector.forward()
            
            # Collect all valid detections first before drawing
            detected_objects = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > CONFIDENCE_THRESHOLD:
                    idx = int(detections[0, 0, i, 1])
                    class_label = CLASSES[idx]
                    if class_label in ["person", "car", "bus", "motorbike"]:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        detected_objects.append({
                            "box": box.astype("int"),
                            "label": class_label,
                            "confidence": confidence
                        })
            
            # --- 3. Depth Estimation (non-blocking) ---
            # Convert frame for the transformers model
            rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Run the blocking model in a separate thread to not freeze the server
            depth_result = await asyncio.to_thread(depth_estimator, pil_image)
            depth_pil = depth_result["depth"]
            
            # Convert depth map for visualization with OpenCV
            depth_numpy = np.array(depth_pil)
            depth_normalized = cv2.normalize(depth_numpy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

            # --- 4. Draw Detections on Both Frames ---
            person_count = sum(1 for obj in detected_objects if obj["label"] == "person")
            car_count = sum(1 for obj in detected_objects if obj["label"] in ["car", "bus"])

            for obj in detected_objects:
                (startX, startY, endX, endY) = obj["box"]
                label_text = f"{obj['label'].capitalize()}: {obj['confidence']:.2%}"
                
                # Draw on the color frame
                cv2.rectangle(processed_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(processed_frame, label_text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw on the depth map
                cv2.rectangle(depth_colormap, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(depth_colormap, label_text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Overlay summary text on both frames
            summary_text = [f"# Persons: {person_count}", f"# Cars/Buses: {car_count}"]
            for i, text in enumerate(summary_text):
                pos = (15, 30 + i * 25)
                cv2.putText(processed_frame, text, pos, cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(depth_colormap, text, pos, cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 255), 2)


            # --- 5. Encode and Store Frames ---
            _, raw_buf = cv2.imencode('.jpg', raw_frame)
            _, proc_buf = cv2.imencode('.jpg', processed_frame)
            _, depth_buf = cv2.imencode('.jpg', depth_colormap)

            async with frame_lock:
                latest_raw_frame = raw_buf.tobytes()
                latest_processed_frame = proc_buf.tobytes()
                latest_depth_frame = depth_buf.tobytes()

            await asyncio.sleep(1 / 30) # Target ~30 FPS

        cap.release()
        video_index = (video_index + 1) % len(video_files)

# ----------------- Stream Generators (Async) -----------------
async def stream_generator(stream_type="raw"):
    while True:
        frame_to_send = None
        async with frame_lock:
            if stream_type == "processed":
                frame_to_send = latest_processed_frame
            elif stream_type == "depth":
                frame_to_send = latest_depth_frame
            else: # "raw"
                frame_to_send = latest_raw_frame
        
        if frame_to_send:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')
        
        await asyncio.sleep(1 / 30) # Match FPS

# ----------------- Routes -----------------
@app.route('/video')
async def stream_raw():
    return Response(stream_generator(stream_type="raw"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video/processed')
async def stream_processed():
    """Streams video with object detection overlays."""
    return Response(stream_generator(stream_type="processed"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video/depth')
async def stream_depth():
    """Streams a depth map with object detection overlays."""
    return Response(stream_generator(stream_type="depth"), mimetype='multipart/x-mixed-replace; boundary=frame')

# ----------------- App Lifecycle -----------------
@app.before_serving
async def startup():
    if not video_files:
        print("[CRITICAL] No video files found. The stream will not start.")
        return
    print(f"[INFO] Found {len(video_files)} videos. Starting background processing task.")
    app.add_background_task(video_processing_loop)

# ----------------- Main -----------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8085, debug=False)