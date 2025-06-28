An async Python app for live object detection and depth estimation on dashcam videos.  
Streams raw frames, detected objects, and depth maps via a Quart web server.

---
## Features
✅ Processes dashcam videos in a loop  
✅ Runs object detection with OpenCV DNN (Caffe model)  
✅ Runs depth estimation with `transformers` (Depth-Anything)  
✅ Streams raw, processed, and depth views via MJPEG endpoints  
✅ Async + `uv` compatible for fast dev workflows

---
## Requirements
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (fast Python package manager & runner)

## **Usage**
1. Clone the repo
```
git clone https://github.com/kkr-genweb/dashcam-depth-object-detect-looper.git
cd dashcam-depth-object-detect-looper
```
2. Install dependencies:
```bash
uv sync
````
3. Place your dashcam videos
Add .mp4 or .mov files to:
```
bdd100k/videos/train/
```
4. Make sure your deploy.prototxt and mobilenet_iter_73000.caffemodel are in the project root.
5. **Run the looper**
```
uv run detect-depth-looper.py
```

6. **Open your browser**
    - Raw video: [http://localhost:8085/video](http://localhost:8085/video)
    - With object detection: [http://localhost:8085/video/processed](http://localhost:8085/video/processed)
    - With depth map: [http://localhost:8085/video/depth](http://localhost:8085/video/depth)
---

## **Notes**
- Adjust the CONFIDENCE_THRESHOLD or CLASSES list as needed.
- The depth estimation uses the depth-anything model, torch and transformers are needed as dependencies.
