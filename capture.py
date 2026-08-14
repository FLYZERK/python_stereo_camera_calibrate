import cv2

pipeline = (
    "nvarguscamerasrc sensor-id=1 ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink drop=1"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ Failed to open pipeline.")
else:
    print("✅ Pipeline opened successfully!")
    ret, frame = cap.read()
    if ret:
        print(f"Captured frame shape: {frame.shape}")
    cap.release()