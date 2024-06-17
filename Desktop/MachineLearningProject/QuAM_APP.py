from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import time
import warnings

app = Flask(__name__)

# Load the scaler and model
scaler = joblib.load('scaler.pkl')
stacking_clf_model = joblib.load('stacking_clf_model.pkl')

# Function to perform prediction
def prediction_function(product_quality_type, air_temperature, process_temperature,
                        rotational_speed, torque, tool_wear):
    try:
        # Calculate Temperature Differential and Power Consumption
        temperature_differential = process_temperature - air_temperature
        power_consumption = rotational_speed * torque

        # Encode Product Quality Type
        product_quality_encoding = {'M': 2, 'L': 1, 'H': 0}
        product_quality_encoded = product_quality_encoding.get(product_quality_type.upper(), -1)

        # Check if Product Quality Type is valid
        if product_quality_encoded == -1:
            return {'error': 'Invalid Product Quality Type!'}

        # Define feature names
        feature_names = ['Product Quality Type', 'Air Temperature', 'Process Temperature',
                         'Rotational Speed', 'Torque', 'Tool Wear', 'Temperature Differential', 'Power Consumption']

        # Create a dictionary with input data
        input_data = np.array([[product_quality_encoded, air_temperature, process_temperature,
                                rotational_speed, torque, tool_wear, temperature_differential, power_consumption]])

        # Scale the input features
        scaled_values = scaler.transform(input_data)

        # Start time for prediction
        start_pred_time = time.time()

        # Predict using the model
        predicted_value = stacking_clf_model.predict(scaled_values)

        # End time for prediction
        end_pred_time = time.time()

        # Calculate prediction time
        prediction_time = end_pred_time - start_pred_time

        # Get probabilities
        probabilities = stacking_clf_model.predict_proba(scaled_values)

        # Decode the numeric predicted value
        predicted_labels = {0: 'Heat Dissipation Failure', 1: 'No Failure', 2: 'Overstrain Failure',
                            3: 'Power Failure', 4: 'Random Failures', 5: 'Tool Wear Failure'}
        predicted_label = predicted_labels.get(predicted_value[0])

        # Get the probability for the predicted label
        predicted_prob = probabilities[0][predicted_value[0]]

        return {'predicted_label': predicted_label, 'predicted_prob': predicted_prob, 'prediction_time': prediction_time}

    except ValueError:
        return {'error': 'Please enter numeric values for all input fields.'}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve input values from the form
        product_quality_type = request.form.get('product_quality')
        air_temperature = float(request.form.get('air_temperature'))
        process_temperature = float(request.form.get('process_temperature'))
        rotational_speed = float(request.form.get('rotational_speed'))
        torque = float(request.form.get('torque'))
        tool_wear = float(request.form.get('tool_wear'))

        prediction_result = prediction_function(product_quality_type, air_temperature, process_temperature,
                                                 rotational_speed, torque, tool_wear)
        return jsonify(prediction_result)

    return render_template('index.html')

if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    app.run(debug=True)
