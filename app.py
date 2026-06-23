from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('main.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert all inputs to correct data types
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Correct feature order (VERY IMPORTANT)
        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])

        # Apply scaling
        data = scaler.transform(data)

        # Prediction
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]
        risk_percent = round(probability * 100, 2)

        # Risk classification
        if probability < 0.4:
            risk_level = "Low Risk"
        elif probability < 0.7:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        return render_template('result.html',
                               prediction=prediction,
                               risk=risk_percent,
                               risk_level=risk_level)

    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(debug=True)