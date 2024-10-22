'''from ultralytics import YOLO
import cv2
import cvzone
import math
import time

confidence = 0.8
 
cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 640)
cap.set(4, 720)
 
model = YOLO("../models/best.pt")
 
classNames = ["fake", "real"]
 
prev_frame_time = 0
new_frame_time = 0
 
while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        print("Failed to read frame from camera.")
        break
    
    results = model(img, stream=True, verbose = False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            if conf>confidence:
                
                if classNames[cls] =='real':
                    color=(0,255,0)
                else:
                    color = (0,0,255)
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color,colorR=color)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%', (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color, colorB=color)
 
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)
 
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
'''


from ultralytics import YOLO
import cv2
import math
import time

# Configuration
confidence = 0.5  # Lowered threshold for faster processing
frame_width = 320  # Reduced frame width for quicker recognition
frame_height = 240  # Reduced frame height

# Open video source
cap = cv2.VideoCapture(0) #5cv2.VideoCapture("meet.mp4")  # Change to your video source
cap.set(3, frame_width)  # Set frame width
cap.set(4, frame_height)  # Set frame height

# Load YOLO model
model = YOLO("models/best.pt")  # Change to your model path

# Class names used in your model
classNames = ["fake", "real"]

# Time tracking for FPS calculation
prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    new_frame_time = time.time()  # Current time for FPS calculation
    
    success, img = cap.read()  # Read a frame from the video source
    if not success:
        print("End of video or failed to read frame.")
        break
    
    # Perform object detection
    results = model(img, stream=True, verbose=False)

    # Process detection results
    for r in results:
        for box in r.boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Calculate confidence and class
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            
            if conf > confidence:
                # Determine bounding box color based on class
                color = (0, 255, 0) if classNames[cls] == "real" else (0, 0, 255)
                
                # Draw a simple rectangle for the bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Display class name and confidence
                cv2.putText(
                    img,
                    f'{classNames[cls].upper()} {int(conf * 100)}%', 
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

    # Calculate and display FPS
    fps = 1 / (new_frame_time - prev_frame_time)  # Efficient FPS calculation
    prev_frame_time = new_frame_time
    
    cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Video", img)
    
    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()  # Release the video source
cv2.destroyAllWindows()  # Close OpenCV windows

