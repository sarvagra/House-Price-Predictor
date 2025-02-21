# 🚀 House Price Prediction API – Deployed & Live! 🏡💰

Excited to share my latest project – a Machine Learning-powered House Price Prediction API built with FastAPI and deployed on Render! 🎯

## 🔹 Project Overview
This API predicts house prices based on features like median income, house age, number of rooms, population, and location. The model is trained using the **California Housing Dataset** and the **XGBoost algorithm** for high accuracy.

## 🌐 Try it Live
**Swagger UI:** [Click here to test the API](https://lnkd.in/eyynxNUW)

1. Go to the Swagger UI and click on `POST /predict`
2. Enter the following example input:
   ```json
   {
     "MedInc": 3.5,
     "HouseAge": 25,
     "AveRooms": 5.2,
     "AveBedrms": 1.1,
     "Population": 1500,
     "AveOccup": 3.0,
     "Latitude": 37.7,
     "Longitude": -122.4
   }
   ```
3. Click **Execute**, and the model will return the predicted house price!

## 📡 Use cURL or Postman
Run this cURL command in your terminal:
```bash
curl -X 'POST' 'https://lnkd.in/eFE5QHX6' \
 -H 'Content-Type: application/json' \
 -d '{
   "MedInc": 3.5,
   "HouseAge": 25,
   "AveRooms": 5.2,
   "AveBedrms": 1.1,
   "Population": 1500,
   "AveOccup": 3.0,
   "Latitude": 37.7,
   "Longitude": -122.4
 }'
```

## 🔧 Tech Stack Used
- **FastAPI** ⚡ - for the backend
- **XGBoost** 📈 - for the predictive model
- **Joblib** 🛠️ - for model serialization
- **Render** 🚀 - for deployment

## 📂 Project Repository
GitHub URL: [Click here](https://lnkd.in/e3XGiy7s)

## 💬 Feedback & Suggestions
I would love to hear your thoughts! Drop a dm or open an issue if you have feedback or suggestions. 😊
