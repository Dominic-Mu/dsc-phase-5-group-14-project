from flask import Flask, render_template, request, jsonify
import json
import numpy as np
from tensorflow.keras.models import model_from_json

app = Flask(__name__)

# Load the model architecture from JSON file
with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# Load the model weights from the HDF5 file
loaded_model.load_weights("model_weights.h5")

# Load the feature scaling parameters
with open("scaling_params.json", "r") as json_file:
    scaling_params = json.load(json_file)

# Function to preprocess input data
def preprocess_input(data):
    # Assuming data is a dictionary with feature names as keys
    numerical_features = ["number_of_people", "number_of_children", "sleep_under_net"]
    categorical_features = ["county"]

    # Select numerical features
    numerical_values = [float(data[feature]) for feature in numerical_features]

    # Select categorical features and convert to one-hot encoding
    categorical_values = [data["county"]]

    # Combine numerical and categorical values
    input_data = np.array(numerical_values + categorical_values).reshape(1, -1)

    # Apply feature scaling
    input_data_scaled = (input_data - np.array(scaling_params["mean"])) / np.array(scaling_params["scale"])

    return input_data_scaled

# Route to render the form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data
        form_data = request.form.to_dict()

        # Preprocess the input data
        input_data = preprocess_input(form_data)

        # Make predictions
        predictions = loaded_model.predict(input_data)

        # Threshold for binary classification
        threshold = 0.5
        binary_prediction = (predictions > threshold).astype(int)[0]

        return render_template('result.html', prediction=binary_prediction)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)

