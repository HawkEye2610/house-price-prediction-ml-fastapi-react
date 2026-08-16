import io
import joblib 
import json
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pathlib import Path
from app.schemas import HouseFeatures


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "house_model.joblib")
features = joblib.load(MODEL_DIR / "house_features.joblib")

with open(MODEL_DIR / "model_metrics.json", "r") as file:
    metrics = json.load(file)



#home
@app.get("/")
def home():
    return {
        "message": "California Housing Price Prediction API",
        "status": "running",
        "endpoints": "send POST request to predict"
    }

@app.get("/health")
def health():
    return {
        "status": "running",
        "model": metrics["model"],
        "features": features,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"]
    }

#prediction
@app.post("/predict")
def predict(house: HouseFeatures):
    try:
        input_data = pd.DataFrame([{
            "MedInc": house.MedInc,
            "HouseAge": house.HouseAge,
            "AveRooms": house.AveRooms,
            "AveBedrms": house.AveBedrms,
            "Population": house.Population,
            "AveOccup": house.AveOccup,
            "Latitude": house.Latitude,
            "Longitude": house.Longitude
        }])

        predicted = model.predict(input_data)[0]

        # The California Housing target is measured in hundreds of thousands of dollars.
        price_usd = predicted * 100000

        mae_usd = metrics["mae"] * 100000

        lower_bound = max(0, price_usd - mae_usd)
        upper_bound = price_usd + mae_usd

        return {
            "predicted_price_usd": round(price_usd, 2),
            "estimated_error_usd": round(mae_usd, 2),
            "prediction_range": {
                "lower": round(lower_bound, 2),
                "upper": round(upper_bound, 2)
            },
            "model": metrics["model"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )
    

@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):

    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a CSV file."
        )

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    required_column = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
        'Population', 'AveOccup', 'Latitude', 'Longitude'
    ]

    missing_columns = [
        col for col in required_column
        if col not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing_columns}"
        )

    if len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV file is empty."
        )

    try:
        predictions = model.predict(df[required_column])
        df['predicted_price_usd'] = predictions * 100000
        df['predicted_price_usd'] = df['predicted_price_usd'].apply(lambda x: f"${x:,.2f}")

        output = df.to_csv(index=False)

        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=predictions.csv"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )