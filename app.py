import configparser
from flask import Flask, render_template, Response

from src.yolo import generate_frames
from src.utils import train_model

app = Flask(__name__)
config = configparser.ConfigParser()
config.read("config.ini")
train_model_first = int(config["OPERATION"]["train"])


@app.route("/")
def index():
    if train_model_first:
        train_model()
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True)
