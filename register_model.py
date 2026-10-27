
# ===== 1. Importer les bibliothèques =====
import mlflow
from mlflow import MlflowClient

# ===== 2. Se connecter à la même base que le TP3 =====
# Le fichier mlflow.db du dossier courant contient tous nos runs.
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ===== 3. Retrouver le meilleur run =====
# Trier les runs par score F1 décroissant.
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

# Le premier run est le meilleur.
best = runs.iloc[0]
run_id = best["run_id"]

print(
    "Meilleur run :",
    best["tags.mlflow.runName"],
    "| id :",
    run_id[:8],
    "..."
)

# ===== 4. Enregistrer le modèle dans le Registry =====
# "runs:/<id>/model" correspond au modèle enregistré dans ce run.
# "churn-model" est le nom du modèle dans le Registry.
result = mlflow.register_model(
    f"runs:/{run_id}/model",
    "churn-model"
)

print(
    "Modèle enregistré : churn-model | version",
    result.version
)

# ===== 5. Définir l'alias « champion » =====
# L'alias champion pointe vers la version enregistrée.
client = MlflowClient()

client.set_registered_model_alias(
    "churn-model",
    "champion",
    result.version
)

print(
    "Alias 'champion' -> version",
    result.version
)
