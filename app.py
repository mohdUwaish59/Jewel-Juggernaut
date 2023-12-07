# Import necessary libraries
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

# Load the trained model
# Load the preprocessing pipeline from the pickle file
with open('artifacts/preprocessor.pkl', 'rb') as preprocess_file:
    preprocess_pipeline = pickle.load(preprocess_file)

# Load the diamond price prediction model from the pickle file
with open('artifacts/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Combine the preprocessing pipeline and the model into a single pipeline
pipeline = Pipeline([
    ('preprocess', preprocess_pipeline),
    ('model', model)
])

# Function to make predictions# Function to make predictions
def predict_price(carat, cut, color, clarity, depth, table, x, y, z):
    input_data = pd.DataFrame([[carat, cut, color, clarity, depth, table, x, y, z]],
                              columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z'])
    price_prediction = pipeline.predict(input_data)
    return price_prediction[0]

# Streamlit app
def main():
    st.title("Diamond Price Prediction App")

    # Input fields
    carat = st.number_input("Carat", min_value=0.2, max_value=5.0, step=0.1, value=1.0)
    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
    color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
    clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
    depth = st.number_input("Depth (%)", min_value=40, max_value=80, step=1, value=60)
    table = st.number_input("Table (%)", min_value=50, max_value=80, step=1, value=60)
    x = st.number_input(label="X",step=1.,format="%.2f") 
    y = st.number_input(label="Y",step=1.,format="%.2f")
    z = st.number_input(label="Z",step=1.,format="%.2f")

    # Button to make predictions
    if st.button("Predict Price"):
        price_prediction = predict_price(carat, cut, color, clarity, depth, table, x, y, z)
        st.success(f"Predicted Price: ${price_prediction:.2f}")

if __name__ == "__main__":
    main()
