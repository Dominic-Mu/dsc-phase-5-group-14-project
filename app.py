import os
import warnings
import json
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.layers import Input
import tensorflow as tf

# Disable resource variables and filter warnings (optional)
os.environ['TF_DISABLE_RESOURCE_VARIABLES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")

# Disable eager execution (ensure graph mode)
tf.compat.v1.disable_eager_execution()
app = Flask(__name__)

# Global model and graph (share across requests)
loaded_model = None
graph = None
new_model = None  # Initialize new_model globally

# Load the model and graph once at startup
def load_model_and_graph():
    global loaded_model, graph, new_model

    # Load model architecture
    with open("model.json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    # Load model weights
    loaded_model.load_weights("model_weights.h5")

    # Set the global graph
    graph = tf.compat.v1.get_default_graph()

    # Initialize the model within the graph context
    with graph.as_default():
        loaded_model.predict(np.zeros((1, 115)))

    # Create the subset model after loading the original model
    selected_features = ["county", "number_of_household_members", "falciparum_present", "number_of_children_resident"]  # Adjust features as needed
    input_subset = Input(shape=(len(selected_features),))

    # Extract relevant layers (assuming they are the first 2 layers)
    shared_layers = loaded_model.layers[0:2]

    # Connect the subset input to the extracted layers
    output = shared_layers(input_subset)
    new_model = Model(inputs=input_subset, outputs=output)

    # Set subset model's weights
    new_model.set_weights([layer.get_weights() for layer in shared_layers])

# Load model and graph at startup
load_model_and_graph()
# Loading the model architecture from JSON file
try:
    with open("model.json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    # Loading the model weights from the HDF5 file
    loaded_model.load_weights("model_weights.h5")

    # Loading the feature scaling parameters
    with open("scaling_params.json", "r") as json_file:
        scaling_params = json.load(json_file)

    # Loading the county options from the JSON file
    with open('county_options.json', 'r') as json_file:
        county_options = json.load(json_file)

    # Explicitly set TensorFlow session and graph for the loaded model
    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        loaded_model.predict(np.zeros((1, 4)))  # Initialize the model within the graph context

except Exception as e:
    raise RuntimeError("Error loading the model: {}".format(str(e)))

# Function to get county name from numerical value
def get_county_name(value):
    for numerical_value, county_name in county_options.items():
        if numerical_value == str(value):
            return county_name
    return 0  # Return 0 if the value is not found

# Function to preprocess input data
def preprocess_input(data):
    selected_features = ["county", "number_of_household_members", "falciparum_present", "number_of_children_resident"]
    binary_features = ["falciparum_present"]

    # Extracting numerical features
    numerical_values = [float(data[feature]) if feature != "county" and feature != "falciparum_present" else 1 if feature == "falciparum_present" and data[feature] == "yes" else 0 for feature in selected_features[1:]]  # Exclude "county" from numerical features

    # Get the numerical value for "Nairobi" from county_options
    county_value = county_options.get(str(data["county"]), 0)  # Default to 0 if the county is not found

    # Get the county name based on the numerical value or return 0 if not found
    county_name = get_county_name(county_value)
    numerical_values.append(float(county_name) if county_name != 0 else 0)

    # Pad with zeros to match the expected input shape
    missing_features = np.zeros(111)  # Assuming the total number of features is 115
    input_data = np.concatenate([missing_features, numerical_values])

    # Reshaping input_data for model compatibility
    input_data = input_data.reshape(1, -1)

    # Applying feature scaling if necessary
    if "scaling_params" in globals() and "mean" in scaling_params and "scale" in scaling_params:
        input_data_scaled = (input_data - np.array(scaling_params["mean"])) / np.array(scaling_params["scale"])
        return input_data_scaled
    else:
        return input_data

# Route to render the form
@app.route('/')
def home():
    return render_template('index.html', county_options=county_options)

# Route to handle the form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Getting the form data
        form_data = request.form.to_dict()

        # Preprocessing the input data
        input_data = preprocess_input(form_data)

        # Making predictions using the explicitly set TensorFlow session and graph
        with graph.as_default():
            predictions = new_model.predict(input_data)

        # Extracting probability from predictions
        probability = predictions[0][0]

        # Threshold for binary classification
        threshold = 0.5
        binary_prediction = (predictions > threshold).astype(int)[0]

        return render_template('result.html', prediction=binary_prediction, probability=probability)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
