import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor
import os

X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

# Chargement des meilleurs paramètres
with open("models/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

# Entraînement du modèle
model = GradientBoostingRegressor(**best_params)
model.fit(X_train, y_train)

# Sauvegarde le modèle
with open("models/gbr_model.pkl", "wb") as f:
    pickle.dump(model, f)
