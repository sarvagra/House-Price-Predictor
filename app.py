from fastapi import FastAPI, Form
import joblib
import numpy as np
from pydantic import BaseModel

# Load the trained model
model = joblib.load("xgboost_model.pkl")

# Initialize FastAPI app
app = FastAPI()

class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/")
def home():
    return {"message": "Welcome to the House Price Prediction API!"}

@app.post("/predict")
def predict_price(features: HouseFeatures):
    try:
        # Convert input to model-compatible format
        input_data = np.array([
            features.MedInc, features.HouseAge, features.AveRooms, 
            features.AveBedrms, features.Population, features.AveOccup, 
            features.Latitude, features.Longitude
        ]).reshape(1, -1)

        # Predict house price
        prediction = model.predict(input_data)

        # Convert to Python float to avoid serialization issues
        return {"predicted_price": float(prediction[0])}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

