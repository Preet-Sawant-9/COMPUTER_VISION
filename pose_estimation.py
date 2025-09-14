import cv2
from ultralytics import YOLO

# Load YOLOv8 pose estimation model (downloaded automatically if missing)
model = YOLO('yolov8n-pose.pt')  # or 'yolov8s-pose.pt' for more accuracy

# Open a video file (replace 'Walk.mp4' with your video filename)
cap = cv2.VideoCapture('walk.mp4')
if not cap.isOpened():
    raise RuntimeError('Failed to open video file.')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 expects BGR numpy arrays as used by OpenCV
    results = model.predict(frame, conf=0.5)

    # Draw the pose landmarks on the frame
    annotated_frame = results[0].plot()  # overlay keypoints and skeleton

    cv2.imshow('YOLOv8 Pose Detection', annotated_frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





#Same but with better results

import cv2
from ultralytics import YOLO

# ------------------------------------------------------------------
# Load the YOLOv8-Pose model (downloads if not present)
model = YOLO('yolov8n-pose.pt')  # Or yolov8s-pose.pt for higher accuracy

# ------------------------------------------------------------------
# Open the video file
cap = cv2.VideoCapture('walk2.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for consistent processing (optional)
    frame = cv2.resize(frame, (960, 720))

    # --------------------------------------------------------------
    # Run YOLOv8 Pose detection on the entire frame
    results = model.predict(frame, conf=0.5)

    # --------------------------------------------------------------
    # Draw detected poses (keypoints and skeleton) directly on frame
    annotated_frame = results[0].plot()  # YOLO draws all people & poses

    # --------------------------------------------------------------
    # Display the processed frame
    cv2.imshow('YOLOv8 Multi-Person Pose Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit if 'q' is pressed

# ------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
