import cv2
import cv2.data
import torch
from torchvision import transforms
from threading import *
from thread import process_image


camera = cv2.VideoCapture(0)

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


def draw_rectange(results, frame):
    # Draw rectangles around detected persons
    for detection in results.xyxy[0]:
        if detection[5] == 0:  # 0 corresponds to 'person' class
            xmin, ymin, xmax, ymax, confidence = detection[:5].cpu().numpy()
            cv2.rectangle(
                frame,
                (int(xmin), int(ymin)),
                (int(xmax), int(ymax)),
                (0, 165, 255),
                2,
            )  # Orange color
            cv2.putText(
                frame,
                "Person",
                (int(xmin), int(ymin) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 165, 255),
                2,
            )

            # Convert the person area to grayscale for face detection
            person_area_gray = cv2.cvtColor(
                frame[int(ymin) : int(ymax), int(xmin) : int(xmax)],
                cv2.COLOR_BGR2GRAY,
            )

            # Perform face detection within the person area
            faces = face_cascade.detectMultiScale(
                person_area_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            # If face detected, draw a green rectangle around the face
            for fx, fy, fw, fh in faces:
                cv2.rectangle(
                    frame,
                    (int(xmin) + fx, int(ymin) + fy),
                    (int(xmin) + fx + fw, int(ymin) + fy + fh),
                    (0, 255, 0),
                    2,
                )  # Green color
            return faces


fps = 30


def generate_frames():
    run_func = fps
    while camera.isOpened():

        # read the camera frame
        success, frame = camera.read()
        if not success:
            break

        # Perform person detection using YOLOv5
        results = model(frame)
        faces = draw_rectange(results, frame)

        # ------------------- process the image in different thread ------------
        run_func -= 1
        if run_func == 0:
            run_func = fps
            Thread(target=process_image, args=(results, frame, faces)).start()
        # ------------------------------------------------

        try:
            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
        except cv2.error:
            continue
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")