from fastapi import FastAPI

app = FastAPI(
    title="LightShift-PatchCore API",
    version="0.1.0",
    description="API de démonstration pour la détection d'anomalies industrielles.",
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "lightshift-patchcore"}


@app.get("/metrics")
def metrics():
    return {"status": "not_implemented", "message": "Brancher les métriques MLflow ici."}


@app.post("/predict")
def predict():
    return {"status": "not_implemented", "message": "Brancher le modèle PatchCore ici."}


@app.post("/calibrate")
def calibrate():
    return {"status": "not_implemented", "message": "Brancher B3 ici."}
