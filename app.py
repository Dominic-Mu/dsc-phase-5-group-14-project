import os
import warnings
os.environ['TF_DISABLE_RESOURCE_VARIABLES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from flask import Flask, render_template, request
import json
import numpy as np
from tensorflow.keras.models import model_from_json

app = Flask(__name__)

# Load the model architecture from JSON file
try:
    with open("model.json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    # Load the model weights from the HDF5 file
    loaded_model.load_weights("model_weights.h5")

    # Load the feature scaling parameters
    with open("scaling_params.json", "r") as json_file:
        scaling_params = json.load(json_file)

    # Load the county options from the JSON file
    with open('county_options.json', 'r') as json_file:
        county_options = json.load(json_file)

except Exception as e:
    raise RuntimeError("Error loading the model: {}".format(str(e)))

# Function to preprocess input data
def preprocess_input(data):
    # Assuming data is a dictionary with feature names as keys
    selected_numerical_features = ["feature1", "feature2", "feature3", "feature4"]
    binary_features = ["falciparum_present"]

    # Select numerical features
    numerical_values = [float(data[feature]) for feature in selected_numerical_features]

    # Select categorical features and convert to one-hot encoding
    county_code = int(data["county"])
    num_counties = len(county_options)

    # Initialize an array of zeros for one-hot encoding
    categorical_values = np.zeros(num_counties)

    # Find the index of the county in the one-hot encoded array
    county_index = list(county_options.keys()).index(str(county_code))

    # Set the corresponding entry to 1 for the selected county
    categorical_values[county_index] = 1

    # Combine numerical and categorical values
    input_data = np.concatenate([numerical_values, categorical_values], dtype=float)

    # Add binary features
    for feature in binary_features:
        input_data = np.append(input_data, [1 if data[feature] == "yes" else 0])

    input_data = input_data.reshape(1, -1)

    # Apply feature scaling
    input_data_scaled = (input_data - np.array(scaling_params["mean"])) / np.array(scaling_params["scale"])

    return input_data_scaled

# Route to render the form
@app.route('/')
def home():
    return render_template('index.html', county_options=county_options)

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

        # Extract probability from predictions
        probability = predictions[0][0]

        # Threshold for binary classification
        threshold = 0.5
        binary_prediction = (predictions > threshold).astype(int)[0]

        return render_template('result.html', prediction=binary_prediction, probability=probability)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
