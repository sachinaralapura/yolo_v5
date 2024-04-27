import cv2
import torch
from torchvision import transforms

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", trust_repo=True)

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Transformation to convert image to tensor
# Transformation pipeline
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# Load webcam


# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*"XVID")
# out = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 480))


cap = cv2.VideoCapture(0)


def generate_frames():
    while True:
        print("running")
        success, frame = cap.read()
        if not success:
            break

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )  # concat frame one by one and show result)


# Release resources
cap.release()
cv2.destroyAllWindows()
