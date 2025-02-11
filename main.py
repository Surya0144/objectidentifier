import torch
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import numpy
# Check if GPU is available and use it if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# For Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width to 640 for faster processing
cap.set(4, 480)  # Set height to 480 for faster processing

# For Video
# cap = cv2.VideoCapture("../Videos/motorbikes.mp4")

# Load YOLO model with the specified device
model = YOLO("yolov8m.pt").to(device)  # Use smaller model for faster processing

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        break

    # Resize the image to be compatible with the model's expected input size
    img_resized = cv2.resize(img, (640,640))

    # Convert image to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # Normalize the image to be in range [0, 1]
    img_tensor /= 255.0

    results = model(img_tensor, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=2)

    fps = 1 / (new_frame_time - prev_frame_time)  # Correct FPS calculation
    prev_frame_time = new_frame_time
    print(f"FPS: {fps:.2f}")

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
