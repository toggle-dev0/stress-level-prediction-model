from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ARTIFACT_PATH = Path("artifacts/stress_level_prediction_model.joblib")

app = FastAPI(
    title="Stress Level Prediction API",
    version="1.0.0",
    description="A FastAPI server for serving a scikit-learn stress level prediction model",
)


class PredictionRequest(BaseModel):
    work_screen_hours: float
    leisure_screen_hours: float
    sleep_hours: float
    sleep_quality_1_5: int
    exercise_minutes_per_week: int


@app.on_event("startup")
def load_model():
    if not ARTIFACT_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {ARTIFACT_PATH}. Run `python train.py` first."
        )

    artifact = joblib.load(ARTIFACT_PATH)
    app.state.model = artifact["model"]
    app.state.target_names = artifact["target_names"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        features = np.array(
            [
                [
                    request.work_screen_hours,
                    request.leisure_screen_hours,
                    request.sleep_hours,
                    request.sleep_quality_1_5,
                    request.exercise_minutes_per_week,
                ]
            ]
        )
        model = app.state.model
        target_names = app.state.target_names

        prediction_id = model.predict(features)[0]

        # return {
        #     "prediction_id": prediction_id,
        #     "prediction_label": target_names[prediction_id],
        #     "probabilities": {
        #         target_names[i]: float(round(probabilities[i], 6))
        #         for i in range(len(target_names))
        #     },
        # }
        return {"Predicted Stress Level": prediction_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
