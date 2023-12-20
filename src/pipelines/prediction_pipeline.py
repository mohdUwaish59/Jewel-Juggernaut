import os
import sys
import pandas as pd
from src.logger import logging
from src.utils import load_object
from dataclasses import dataclass

@dataclass
class Config:
    model_path: str
    preprocessor_path: str

class PredictPipeline:
    def __init__(self, config):
        self.config = config
        self.model = self.load_model()
        self.preprocessor = self.load_preprocessor()

    def load_model(self):
        try:
            model = load_object(file_path=self.config.model_path)
            return model
        except Exception as e:
            raise Exception(f"Error loading the model: {e}", sys)

    def load_preprocessor(self):
        try:
            preprocessor = load_object(file_path=self.config.preprocessor_path)
            return preprocessor
        except Exception as e:
            raise Exception(f"Error loading the preprocessor: {e}", sys)

    def predict(self, features):
        try:
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            return preds

        except Exception as e:
            raise Exception(f"Error during prediction: {e}", sys)

@dataclass
class CustomData:
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "carat": [self.carat],
                "cut": [self.cut],
                "color": [self.color],
                "clarity": [self.clarity],
                "depth": [self.depth],
                "table": [self.table],
                "x": [self.x],
                "y": [self.y],
                "z": [self.z],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise Exception(f"Error creating DataFrame: {e}", sys)


if __name__ == '__main__':
    try:
        # Set your configuration here
        config = Config(
            model_path=os.path.join("artifacts", "model.pkl"),
            preprocessor_path=os.path.join("artifacts", "preprocessor.pkl")
        )

        # Example data for prediction
        custom_data = CustomData(
            carat=1.0,
            cut="Ideal",
            color="G",
            clarity="VS1",
            depth=60.0,
            table=60.0,
            x=5.67,
            y=5.68,
            z=3.56
        )

        # Create a DataFrame from custom data
        input_data = custom_data.get_data_as_data_frame()

        # Create prediction pipeline
        predict_pipeline = PredictPipeline(config)

        # Make predictions
        predictions = predict_pipeline.predict(input_data)
        logging.info(f"Predictions: {predictions}")

    except Exception as ce:
        logging.error(f"Custom Exception: {ce}")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
