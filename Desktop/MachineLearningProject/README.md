# Machine Failure Prediction App

**Overview**

This Flask application predicts machine failure types based on user input. It utilizes a trained Random Forest model to make predictions.

**Installation**

1- Clone or download the repository.

2- Install Python dependencies using the following command:


```python
pip install -r requirements.txt
```

**Usage**

1- Navigate to the directory where the application is located.

2- Run the Flask application by executing the following command:


```python
python app_name.py
```

3- Open a web browser and go to http://localhost:5000.

4- Fill out the input form with the required information (product quality type, air temperature, process temperature, etc.).

5- Submit the form to see the predicted machine failure type and probability.

# File Structure

- app.py: Main Flask application file containing the route definitions.
    
- templates/index.html: HTML template for the web interface.
    
- scaler.pkl: Scaler object used to preprocess input data.
    
- random_forest_model.pkl: Trained Random Forest model for prediction

# Credits

- Flask: https://flask.palletsprojects.com/
- scikit-learn: https://scikit-learn.org/
