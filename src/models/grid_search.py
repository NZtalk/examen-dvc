import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import os

X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel() #permet d'applatir l'array en 1D essentiel pour scikit learn

# Définition hyperparamètres à tester
param_grid = {
    "n_estimators": [50, 100],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5]
}

# Model + GridSearch
model = GradientBoostingRegressor()

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)


# Sauvegarde des meilleurs paramètres
with open("models/best_params.pkl", "wb") as f:
    pickle.dump(grid_search.best_params_, f)
