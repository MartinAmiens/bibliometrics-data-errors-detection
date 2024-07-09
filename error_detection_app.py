# app.py

from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load your trained model
model = load_model('path_to_your_trained_model.h5')

# Load necessary encoders or preprocessors
encoder = OneHotEncoder()  # Load or initialize your encoder

# Example: Load feature names if needed
feature_names = ['Full_Name_John Doe', 'Full_Name_Jane Smith']  # Replace with actual feature names

# Define a function to process input and make predictions
def predict_error(id, full_name):
    # Process input (e.g., encode features)
    X = pd.DataFrame({'id': [id], 'Full_Name': [full_name]})
    X_encoded = encoder.transform(X[['Full_Name']])
    X_encoded = pd.concat([X[['id']], pd.DataFrame(X_encoded.toarray(), columns=feature_names)], axis=1)
    
    # Make predictions
    predictions = model.predict(X_encoded)
    binary_prediction = np.round(predictions).astype(int)[0]  # Convert to binary prediction (0 or 1)
    
    return binary_prediction

# Define routes for your web application
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    id = request.form['id']
    full_name = request.form['full_name']
    
    # Perform prediction
    prediction = predict_error(id, full_name)
    
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
