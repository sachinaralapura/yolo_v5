from our_models.initialize_model import *
from threading import *
from src.utils import Utils
from our_models.initialize_model import face_cascade

# create a utils object
# --------------------------------------------------------------------------------


def get_age_gender(blob):
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = gender_list[genderPreds[0].argmax()]
    print(f"Gender: {gender}")

    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]
    print(f"Age: {age} years")
    return (gender, age)


# ---------------------------------------------------------------------------------


def process_image(result, frame):
    """
    process_image( result , frame)
    arguments
    result : result of running the frame in yolov5 model
    frame  : the image fromt the camera

    we need the result attribute because when croping the face we are doing it
    with respect to the xmin  and ymin  which are the cordinated

    the detection of person and face is similar to the draw_rectangle() function
    we are doing the same here , i think this is best option as of now  because
    we are running this function once in a while ( can configure fps in config.ini file )

    i suggest not to set the fps less then 30 , this function is the bottleneck
    """
    print("-----processing frame in new thread---------")

    for detection in result.xyxy[0]:  # for each object detected
        if detection[5] == 0:  # 0 corresponds to 'person' class
            xmin, ymin, xmax, ymax = detection[:4].cpu().numpy()

            # Convert the person area to grayscale for face detection
            person_area_gray = cv2.cvtColor(
                frame[int(ymin) : int(ymax), int(xmin) : int(xmax)],
                cv2.COLOR_BGR2GRAY,
            )

            # the try and expect block is for the face detection which may leads to cv2 error
            # because of the minSize parameter ( i don't know the exact minSize to give)
            try:
                # Perform face detection within the person area
                face = face_cascade.detectMultiScale(
                    person_area_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                for fx, fy, fw, fh in face:
                    face_crop = frame[
                        int(ymin) + fy : int(ymin) + fy + fh,
                        int(xmin) + fx : int(xmin) + fx + fw,
                    ]

                    cv2.imwrite("Data/faces/face.jpg", face_crop)

                    # creating blob for age and gender detection
                    blob = cv2.dnn.blobFromImage(
                        face_crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False
                    )

                    # print the age and gender of all the faces in the frames
                    (gender, age) = get_age_gender(blob)
            except cv2.error:
                continue
