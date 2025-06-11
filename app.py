from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"❌ Could not load model: {e}")

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

#predict Page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            sl = float(request.form['sepal_length'])
            sw = float(request.form['sepal_width'])
            pl = float(request.form['petal_length'])
            pw = float(request.form['petal_width'])

            data = np.array([[sl, sw, pl, pw]])
            prediction_index = model.predict(data)[0]

            # 🌸 Map class index to flower name
            label_map = {
                0: "Iris-setosa",
                1: "Iris-versicolor",
                2: "Iris-virginica"
            }

            prediction = label_map.get(prediction_index, "Unknown Iris Flower")

            return render_template('result.html', prediction=prediction)
        except Exception as e:
            return f"Prediction Error: {e}"
    return render_template('predict.html')


# About Page
@app.route('/about')
def about():
    return render_template('about.html')

# Iris Types Page
@app.route('/iris-types')
def iris_types():
    return render_template('iris-types.html')

# Optional: Contact Page (if you want)
@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
