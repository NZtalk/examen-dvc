import pandas as pd
import pickle
import json
import os
from sklearn.metrics import r2_score, mean_squared_error

X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv")

with open("models/gbr_model.pkl", "rb") as f:
    model = pickle.load(f)

# Prédiction
y_pred = model.predict(X_test)

# Sauvegarde prédictions
pd.DataFrame(y_pred, columns=["prediction"]).to_csv("data/prediction.csv", index=False)

# Calcul des métriques
scores = {
    "r2_score": r2_score(y_test, y_pred),
    "mse": mean_squared_error(y_test, y_pred)
}

# Sauvegarde des scores
with open("metrics/scores.json", "w") as f:
    json.dump(scores, f, indent=4)
