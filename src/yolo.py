import cv2
import copy
import configparser
from threading import *
from src.thread import process_image
from our_models.initialize_model import model, face_cascade


# read config.ini file to get the default parameters
config = configparser.ConfigParser()
config.read("config.ini")
fps = int(config["DEFAULT"]["fps"]) or 30
# initial the camera
camera = cv2.VideoCapture(0)


# Draw rectangles around detected persons
def draw_rectangle(results, frame):
    """
    draw_rectangle
    results : the result of running the model with frame
    frame   : the current image from the camera
    ----------------------------------
    The main task of this function is to draw the rectangle around the body , face
    how :
        for this frame "results" contains  the coordinates of all the objects detected , we need only the person
        object
    """
    for detection in results.xyxy[0]:  # for each object detected
        if detection[5] == 0:  # 0 corresponds to 'person' class
            try:
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
                face = face_cascade.detectMultiScale(
                    person_area_gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10)
                )

                # If face detected, draw a green rectangle around the face
                for fx, fy, fw, fh in face:
                    cv2.rectangle(
                        frame,
                        (int(xmin) + fx, int(ymin) + fy),
                        (int(xmin) + fx + fw, int(ymin) + fy + fh),
                        (0, 255, 0),
                        2,
                    )  # Green color
            except cv2.error:
                continue


def generate_frames():
    """
    this is function  main task is to yield the boxed image of frame , so that it can be consumed by the
    web api in flask application

    this function also send the results of the yolov5 and frame captured to a separate thread
    """
    run_func = fps
    while camera.isOpened():

        # read the camera frame
        success, frame = camera.read()
        if not success:
            break

        # Perform person detection using YOLOv5
        results = model(frame)

        # ------------------- process the image in different thread ------------
        run_func -= 1
        if run_func == 0:
            # copy the frame to send to the new thread
            framecopy = copy.copy(frame)

            run_func = fps
            new_thread = Thread(target=process_image, args=(results, framecopy))
            new_thread.start()
        # ------------------------------------------------

        draw_rectangle(results, frame)

        try:
            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
        except cv2.error:
            continue
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
