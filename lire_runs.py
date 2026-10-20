# ===== 1. Importer la bibliothèque =====
import mlflow

# ===== 2. Récupérer les runs et les trier par score F1 décroissant =====
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

# ===== 3. Afficher les résultats sous forme de tableau =====
colonnes = [
    "tags.mlflow.runName",
    "params.n_estimators",
    "params.max_depth",
    "metrics.accuracy",
    "metrics.f1"
]

print(runs[colonnes].to_string(index=False))

# ===== 4. Identifier et afficher le meilleur run =====
if not runs.empty:
    meilleur = runs.iloc[0]

    print()
    print(
        "Meilleur run :",
        meilleur["tags.mlflow.runName"],
        "| id :",
        meilleur["run_id"][:8],
        "..."
    )
else:
    print("Aucun run trouvé dans l'expérience churn-prediction.")