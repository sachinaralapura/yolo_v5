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

ageProto = "our_models/age_deploy.prototxt"
ageModel = "our_models/age_net.caffemodel"
genderProto = "our_models/gender_deploy.prototxt"
genderModel = "our_models/gender_net.caffemodel"
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
