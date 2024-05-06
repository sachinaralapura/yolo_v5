import cv2

ageProto = "models/age_deploy.prototxt"
ageModel = "models/age_net.caffemodel"
genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


gender_list = ["male", "female"]
ageList = [
    "(0-5)",
    "(5-15)",
    "(15-20)",
    "(21-30)",
    "(30-40)",
    "(40-55)",
    "(56-70)",
    "(70-100)",
]

ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
