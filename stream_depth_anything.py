import cv2
from transformers import pipeline
from PIL import Image
import numpy as np
from pathlib import Path # Import the Path object for modern file/path handling

# --- This is the function you want to call in a loop ---
# (Using the fully corrected version from our previous conversations)
def extract_frames(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate) if frame_rate > 0 else 1
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if count % interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # This is the slowest step, as it runs the AI model
            depth_pil = depth_estimator(pil_image)["depth"]
            
            depth_numpy = np.array(depth_pil)

            cv2.imshow("Original Frame", frame)
            cv2.imshow("Depth Map", depth_numpy)
        
        count += 1
        
        # This is crucial for video display and allows you to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Set a flag or break out completely if 'q' is pressed
            # For simplicity, we'll just close this specific video
            break
    
    cap.release()
    # We move destroyAllWindows to the main loop to prevent it from closing
    # the windows after every single video.
    # cv2.destroyAllWindows()


# --- Main script execution ---
if __name__ == "__main__":
    # Initialize the pipeline once, so it's not reloaded for every video
    print("Initializing the depth estimation pipeline...")
    depth_estimator = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    print("Pipeline initialized.")

    # 1. Define the path to the directory containing the videos
    video_directory = Path("bdd100k/videos/train/")

    # 2. Check if the directory exists to provide a helpful error
    if not video_directory.is_dir():
        print(f"Error: Directory not found at '{video_directory}'")
        print("Please make sure the 'bdd100k' folder is in the correct location.")
    else:
        # 3. Find all files ending with .mov using .glob()
        print(f"Searching for .mov files in {video_directory}...")
        mov_files = list(video_directory.glob("*.mov"))
        
        if not mov_files:
            print("No .mov files found in the specified directory.")
        else:
            print(f"Found {len(mov_files)} videos to process.")
            
            # 4. Loop through each video file found
            for video_file in mov_files:
                # The video_file is a Path object. We use str() to pass its path as a string.
                print("-" * 60)
                print(f"Processing video: {video_file.name}")

                try:
                    # 5. Call your function for the current video file
                    extract_frames(str(video_file), frame_rate=2)
                except Exception as e:
                    print(f"An error occurred while processing {video_file.name}: {e}")
                    # continue to the next video even if one fails
                    continue

    # Clean up all OpenCV windows at the very end of the script
    cv2.destroyAllWindows()
    print("-" * 60)
    print("All videos processed.")