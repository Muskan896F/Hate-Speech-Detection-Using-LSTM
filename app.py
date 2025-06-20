from fastapi import FastAPI, Request
from fastapi.responses import Response, RedirectResponse
from pydantic import BaseModel
import uvicorn
import sys

from hate.pipeline.train_pipeline import TrainPipeline
from hate.pipeline.prediction_pipeline import PredictionPipeline
from hate.exception import CustomException
from hate.constants import *

app = FastAPI(
    title="Hate Speech Detection API",
    description="API to train model and make predictions on text for hate speech detection",
    version="1.0"
)

class TextInput(BaseModel):
    text: str

@app.get("/", tags=["Root"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train", tags=["Training"])
async def training():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response(content="✅ Training completed successfully!", media_type="text/plain")
    except Exception as e:
        print(f"❌ Training error: {e}")
        return Response(content=f"❌ Error during training: {e}", media_type="text/plain")

@app.post("/predict", tags=["Prediction"])
async def predict_route(input: TextInput):
    try:
        text = input.text
        prediction_pipeline = PredictionPipeline()
        prediction = prediction_pipeline.run_pipeline(text)
        return {"prediction": prediction}
    except Exception as e:
        print(f"❌ Prediction error: {e}")  # 👈 LOG ERROR
        return Response(content=f"❌ Internal server error: {e}", media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
