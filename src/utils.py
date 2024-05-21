import face_recognition, os, configparser, csv
from datetime import datetime
from sklearn import svm
from joblib import dump

config = configparser.ConfigParser()
config.read("config.ini")
dataset_path = config["PATH"]["dataset"]


class Utils:
    def __init__(self):
        # read config.ini file to get the default parameters
        self.main_path = config["PATH"]["main"]
        self.detected_face_path = config["PATH"]["detected_face"]
        self.csv_path = config["PATH"]["csv_path"]

    # get the number of detected faces
    def get_detected_faces(self):
        pass

    # append the given row to csv file in csv_path
    def append_row_to_csv(self, row):
        with open(self.csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)


    def update_csv(self, gender="male", age="23"):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")  # Format: MM:HH:SS
        current_date = now.strftime("%d-%m-%Y")  # Format: DD-MM-YYYY
        current_day = now.strftime("%A")  # Full weekday name, e.g., Monday
        ID = self.generate_id()

        # self.append_row_to_csv("path", "row")

    def generate_id():
        # todo!
        pass


def train_model():

    print("----start model train------")

    folders = []
    known_face_names = []
    known_face_encodings = []

    # get the directory name in Dataset directory
    # r=root, d=directories, f = files
    for r, d, f in os.walk(dataset_path):
        for folder in d:
            folders.append(folder)

    # for all first image
    for f in folders:
        first_image_path = dataset_path + "/" + f + "/0.jpg"
        ru_image = face_recognition.load_image_file(first_image_path)
        ru_face_encoding = face_recognition.face_encodings(ru_image)[0]
        known_face_names.append(f)
        known_face_encodings.append(ru_face_encoding)

    print("---extration done------")

    clf = svm.SVC(gamma="scale")
    clf.fit(known_face_encodings, known_face_names)
    dump(clf, dataset_path + "/SVM.Model")

    print("----done training-----")
