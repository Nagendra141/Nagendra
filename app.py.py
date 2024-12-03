from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the saved model
model = joblib.load('model_campus_placement.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Campus Placement Prediction Model is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json()

    # Convert JSON data to DataFrame
    input_data = pd.DataFrame(data, index=[0])

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    # Return result as JSON
    result = {
        'placement_status': 'Placed' if prediction == 1 else 'Not Placed',
        'placement_probability': probability
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
