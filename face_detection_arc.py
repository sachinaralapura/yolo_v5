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
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*"XVID")
# out = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform person detection using YOLOv5
    results = model(frame)

    # Draw rectangles around detected persons
    for detection in results.xyxy[0]:
        if detection[5] == 0:  # 0 corresponds to 'person' class
            xmin, ymin, xmax, ymax, confidence = detection[:5].cpu().numpy()
            cv2.rectangle(
                frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 165, 255), 2
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
                frame[int(ymin) : int(ymax), int(xmin) : int(xmax)], cv2.COLOR_BGR2GRAY
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

                # Draw blue rectangle around the whole person
                cv2.rectangle(
                    frame,
                    (int(xmin), int(ymin)),
                    (int(xmax), int(ymax)),
                    (255, 0, 0),
                    2,
                )  # Blue color
                cv2.putText(
                    frame,
                    "Verifiable",
                    (int(xmin), int(ymin) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2,
                )

    # output the frame
    # out.write(frame)

    # Show frame
    cv2.imshow("Person and Face Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
