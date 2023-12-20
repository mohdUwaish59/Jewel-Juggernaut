# Import necessary libraries
import streamlit as st
import pandas as pd
from src.pipelines.prediction_pipeline import PredictPipeline, CustomData
from src.logger import logging
from dataclasses import dataclass
import os

# Function to make predictions

@dataclass
class Config:
    model_path: str = os.path.join("artifacts", "model.pkl")
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")
    
def predict_price(prediction_pipeline, carat, cut, color, clarity, depth, table, x, y, z):
    try:
        # Create CustomData instance
        custom_data = CustomData(carat, cut, color, clarity, depth, table, x, y, z)

        # Convert CustomData to a DataFrame
        input_data = custom_data.get_data_as_data_frame()

        # Make predictions using PredictPipeline
        price_prediction = prediction_pipeline.predict(input_data)
        return price_prediction[0]

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

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
    x = st.number_input(label="X", step=1., format="%.2f")
    y = st.number_input(label="Y", step=1., format="%.2f")
    z = st.number_input(label="Z", step=1., format="%.2f")

    # Button to make predictions
    if st.button("Predict Price"):
        config = Config()
        prediction_pipeline = PredictPipeline(config)
        price_prediction = predict_price(prediction_pipeline,carat, cut, color, clarity, depth, table, x, y, z)

        if price_prediction is not None:
            st.success(f"Predicted Price: ${price_prediction:.2f}")

if __name__ == "__main__":
    main()
