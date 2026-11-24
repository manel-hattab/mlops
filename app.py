
# ===== 1. Importer les bibliothèques =====
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ===== 2. Créer l'application =====
app = FastAPI(title="API Churn")


# ===== 3. Charger le champion UNE SEULE FOIS, au démarrage =====
# Le recharger à chaque requête rendrait l'API très lente.
mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = mlflow.sklearn.load_model(
    "models:/churn-model@champion"
)


# ===== 4. Décrire les données attendues =====
# Les données sont validées automatiquement par Pydantic.
class Client(BaseModel):
    age: int  # Nombre entier
    salaire: float  # Nombre décimal
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float


# ===== 5. Route /health : vérifier que l'API est en vie =====
@app.get("/health")
def health():
    # GET : on lit, on n'envoie rien.
    return {"status": "ok"}


# ===== 6. Route /predict : recevoir un client et renvoyer la prédiction =====
@app.post("/predict")
def predict(client: Client):
    # a) Transformer le client reçu en tableau d'une ligne
    ligne = pd.DataFrame([client.model_dump()])

    # b) Transformer le texte « abonnement » en colonnes 0/1
    # comme lors de l'entraînement.
    ligne = pd.get_dummies(ligne)

    # c) Garder les colonnes du modèle dans le bon ordre.
    # Les colonnes absentes prennent la valeur 0.
    ligne = ligne.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # d) Prédire : 1 = le client part, 0 = le client reste
    resultat = int(model.predict(ligne)[0])

    return {
        "churn": resultat,
        "message": (
            "Le client va partir"
            if resultat == 1
            else "Le client reste"
        )
    }
