from flask import Flask, request, make_response, jsonify, render_template
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

@app.route('/ussd', methods=['POST'])
def ussd():
    session_id = request.values.get("sessionId", None)
    service_code = request.values.get("serviceCode", None)
    phone_number = request.values.get("phoneNumber", None)
    text = request.values.get("text", "")

    # Split the text input to determine the user's response
    user_response = text.strip().split("*")

    if text == "":
        # This is the first request. Respond with the main menu
        response = "CON Welcome to the Occupation Predictor\n"
        response += "1. Predict Occupation"
    elif text == "1":
        # Prompt user for input data
        response = "CON Enter your details separated by commas (e.g., age,gender,education):"
    elif len(user_response) == 2:
        # User has entered their details
        try:
            # Parse user input
            user_input = user_response[1].split(",")
            # Assuming the model expects age, gender, education
            input_data = pd.DataFrame([{
                "age": int(user_input[0]),
                "gender": user_input[1],
                "education": user_input[2]
            }])
            # Predict occupation
            prediction = model.predict(input_data)
            predicted_occupation = prediction[0]
            response = f"END Predicted Occupation: {predicted_occupation}"
        except Exception as e:
            response = f"END Error: {str(e)}"
    else:
        response = "END Invalid input. Please try again."

    return make_response(response, 200, {'Content-Type': 'text/plain'})

if __name__ == '__main__':
    app.run(debug=True)
