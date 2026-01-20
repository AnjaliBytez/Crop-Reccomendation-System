import numpy as np 
from flask import Flask, request, render_template 
import pickle

flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def home():
    return render_template("index.html", show="home")

@flask_app.route("/predict", methods=["POST"])
def predict():
    float_feature = [float(x) for x in request.form.values()]
    features = [np.array(float_feature)]
    prediction = model.predict(features)[0]

    # This renders index.html but tells it to show the result-page
    return render_template("index.html", 
                           show="result", 
                           prediction_text=prediction)

if __name__ == "__main__":
    flask_app.run(debug=True)

#python "C:\Anjali\crop recc\app.py"

