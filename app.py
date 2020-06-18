# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'diabetes_predictor_model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('model.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        age = int(request.form['age'])

        data = np.array([[preg, glucose, bp, insulin,bmi, age]])
        my_prediction = classifier.predict(data)

        return render_template('prediction.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)