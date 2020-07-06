from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import os
model = load_model("/Users/gautammanocha/python projects/`DL/Classifier")


def pred2(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (150, 150))
    img = img/255
    img = img.reshape(1, 150, 150, 3)
    y = model.predict(img)
    y = y.round()
    dic1 = {1: "dog", 0: "cat"}
    b = dic1[y[0][0]]
    return b

app = Flask(__name__)
upload = "static"
app.config["upload"] = upload
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/prediction', methods=["POST"])
def pred():
    a = request.files["img"]
    if not a:
        return render_template("success.html")
    images = os.listdir(app.config["upload"])
    if len(images) <= 2:
        a.save(os.path.join(app.config["upload"], a.filename))
    else:
        image = images[0]
        os.remove(app.config["upload"]+"/"+image)
        a.save(os.path.join(app.config["upload"], a.filename))
    predict = pred2(os.path.join(app.config["upload"], a.filename))
    return render_template("layout.html", result=predict, file=a.filename)

if __name__ == "__main__":
    app.run(debug=True)


