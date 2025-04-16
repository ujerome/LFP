from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the trained model
model = joblib.load('occupation_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        data = request.form.to_dict()

        # Prepare data for prediction
        input_data = pd.DataFrame([data])

        # Predict the occupation
        prediction = model.predict(input_data)
        predicted_occupation = prediction[0]  # Get the first prediction from the array

        # Return the result as JSON response
        return jsonify({'Occupation': predicted_occupation})

    except Exception as e:
        return jsonify({'Error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
