from fastapi import FastAPI
from starlette.requests import Request
from mangum import Mangum

from app.data import FeatureStoreDataRequest
from app.predictor import PredictorService

MODELS_PATH = "models"
app = FastAPI()
handler = Mangum(app)


@app.on_event("startup")
def load_predictor():
    global predictor
    predictor = PredictorService(models_path=MODELS_PATH)


@app.get("/")
def home():
    return "Home"


@app.post("/predict")
async def predict(request: Request, features: FeatureStoreDataRequest):
    if request.method == "POST":
        loan_id = features.id
        member_id = features.member_id
        funded_amnt = features.funded_amnt
        cleaned_features = features.__dict__
        del cleaned_features["id"]
        del cleaned_features["member_id"]
        predictions = predictor.predict(cleaned_features)
        return {
            "id": loan_id,
            "member_id": member_id,
            "funded_amnt": funded_amnt,
            **predictions,
        }
    return "No POST request found!"
