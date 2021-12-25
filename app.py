from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('BreastCancerClassifier.pkl')
scaler = joblib.load('BreastCancerScaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_feature = [float(x) for x in request.form.values()]
    features = [np.array(input_feature)]
    pred = model.predict(scaler.transform(features))
    if(pred==0):
        text = "Sorry! You have Cancer."
    elif(pred==1):
        text = "You do not have Cancer."
    else:
        text = "Something went wrong! Please try again Later."

    print(features)
    print(pred)
    return render_template('index.html', predicted_text=text)


if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080)