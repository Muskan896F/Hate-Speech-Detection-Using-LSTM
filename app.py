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

# 👇 Data model for POST input
class TextInput(BaseModel):
    text: str


@app.get("/", tags=["Root"])
async def index():
    """
    Redirect to Swagger UI.
    """
    return RedirectResponse(url="/docs")


@app.get("/train", tags=["Training"])
async def training():
    """
    Triggers the model training pipeline.
    """
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response(content="✅ Training completed successfully!", media_type="text/plain")
    except Exception as e:
        return Response(content=f"❌ Error during training: {e}", media_type="text/plain")


@app.post("/predict", tags=["Prediction"])
async def predict_route(input: TextInput):
    """
    Predicts if the input text is hate speech or not.
    """
    try:
        text = input.text
        prediction_pipeline = PredictionPipeline()
        prediction = prediction_pipeline.run_pipeline(text)
        return {"prediction": prediction}
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    # ✅ Fallback defaults for local run
    host = APP_HOST if 'APP_HOST' in globals() else "127.0.0.1"
    port = APP_PORT if 'APP_PORT' in globals() else 8000
    uvicorn.run("app:app", host=host, port=port, reload=True)
