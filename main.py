
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import joblib, json, numpy as np, pandas as pd

import os

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "models", "stockout_model.joblib")
SCALER_PATH   = os.path.join(BASE_DIR, "models", "stockout_scaler.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "models", "stockout_metadata.json")

print(f"Looking for model at: {MODEL_PATH}")
print(f"File exists: {os.path.exists(MODEL_PATH)}")

FEATURE_COLUMNS = [
    "Quantity_Current", "Weekly_Avg", "Monthly_Avg_This_Month",
    "Lead_Time_Days", "Days_Until_Expiry", "Batch_Count", "Cost_Per_Unit"
]

ml = {}

@asynccontextmanager
async def lifespan(app):
    ml["model"]  = joblib.load(MODEL_PATH)
    ml["scaler"] = joblib.load(SCALER_PATH)
    with open(METADATA_PATH) as f:
        ml["metadata"] = json.load(f)
    print("Model loaded")
    yield
    ml.clear()

app = FastAPI(title="PharmGuard API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["YOUR_FRONTEND_URL"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

class StockRequest(BaseModel):
    quantity_current:       float = Field(..., gt=0)
    weekly_avg:             float = Field(..., ge=0)
    monthly_avg_this_month: float = Field(..., ge=0)
    lead_time_days:         float = Field(..., gt=0)
    days_until_expiry:      float = Field(..., ge=0)
    batch_count:            int   = Field(..., ge=1)
    cost_per_unit:          float = Field(..., gt=0)
    medicine_name:          Optional[str] = "Unknown"
    medicine_id:            Optional[int] = None

class StockResponse(BaseModel):
    medicine_name:         str
    medicine_id:           Optional[int]
    days_until_stockout:   float
    alert_level:           str
    alert_description:     str
    recommendation:        str

def _get_alert(days):
    if days < 7:    return "CRITICAL", "Order immediately"
    elif days < 14: return "WARNING",  "Order within 3 days"
    elif days < 30: return "CAUTION",  "Monitor closely"
    else:           return "OK",       "Adequate stock"

RECOMMENDATIONS = {
    "CRITICAL": "URGENT: Place emergency order now.",
    "WARNING":  "Place standard reorder, expedite if possible.",
    "CAUTION":  "Schedule routine reorder.",
    "OK":       "Continue normal operations."
}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": bool(ml), "version": "2.0.0"}

@app.get("/model-info")
def model_info():
    return ml.get("metadata", {})

@app.post("/predict", response_model=StockResponse)
def predict(req: StockRequest):
    try:
        df = pd.DataFrame([{
            "Quantity_Current": req.quantity_current,
            "Weekly_Avg": req.weekly_avg,
            "Monthly_Avg_This_Month": req.monthly_avg_this_month,
            "Lead_Time_Days": req.lead_time_days,
            "Days_Until_Expiry": req.days_until_expiry,
            "Batch_Count": req.batch_count,
            "Cost_Per_Unit": req.cost_per_unit
        }])
        scaled = ml["scaler"].transform(df)
        days   = float(np.clip(ml["model"].predict(scaled)[0], 0, 365))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    level, desc = _get_alert(days)
    return StockResponse(
        medicine_name=req.medicine_name, medicine_id=req.medicine_id,
        days_until_stockout=round(days, 2), alert_level=level,
        alert_description=desc, recommendation=RECOMMENDATIONS[level]
    )

@app.post("/batch-predict", response_model=List[StockResponse])
def batch_predict(requests: List[StockRequest]):
    return [predict(r) for r in requests]

@app.get("/")
def root():
    return {"name": "PharmGuard API", "version": "2.0.0", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=2, reload=False)
