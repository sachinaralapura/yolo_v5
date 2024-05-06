from models.initialize_model import *


def process_image(result, frame, faces):
    for detection in result.xyxy[0]:
        if detection[5] == 0:  # 0 corresponds to 'person' class
            xmin, ymin = detection[:2].cpu().numpy()
            # crop_image = frame[int(ymin) : int(ymax), int(xmin) : int(xmax)]

            # crop only the face and save in a file

            try:
                no_of_faces = 0
                for fx, fy, fw, fh in faces:
                    no_of_faces += 1
                    filename = "faces/face" + str(no_of_faces) + ".jpg"
                    face_crop = frame[
                        int(ymin) + fy : int(ymin) + fy + fh,
                        int(xmin) + fx : int(xmin) + fx + fw,
                    ]
                    cv2.imwrite(filename, face_crop)
                    blob = cv2.dnn.blobFromImage(
                        face_crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False
                    )

                    genderNet.setInput(blob)
                    genderPreds = genderNet.forward()
                    print(f"Gender: {gender_list[genderPreds[0].argmax()]}")

                    ageNet.setInput(blob)
                    agePreds = ageNet.forward()
                    print(f"Age: {ageList[ agePreds[0].argmax()]} years")

            except cv2.error:
                continue
