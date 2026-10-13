import numpy as np
import pandas as pd

# pandas sert à construire et manipuler notre tableau de données.
# NumPy permet de générer des valeurs numériques aléatoires.

# Cette graine permet de recréer les mêmes données à chaque exécution.
np.random.seed(42)

n = 500  # Nombre de clients fictifs à générer.

# Chaque colonne ci-dessous est une caractéristique d'entrée du modèle.
data = pd.DataFrame(
    {
        "age": np.random.randint(18, 70, n),  # Âge entre 18 et 69 ans.
        "salaire": np.random.randint(1200, 8000, n),  # Salaire mensuel simulé.
        "abonnement": np.random.choice(
            ["Basique", "Standard", "Premium"], n
        ),  # Type d'offre.
        "anciennete": np.random.randint(0, 15, n),  # Ancienneté en années.
        "consommation": np.random.randint(
            0, 100, n
        ),  # Score d'usage du service, sur 100.
    }
)

# La cible « churn » vaut 1 si le client est peu actif et récent.
data["churn"] = (
    (data["consommation"] < 30) & (data["anciennete"] < 2)
).astype(int)

# Enregistre le tableau au format CSV dans le répertoire courant.
data.to_csv("churn_data.csv", index=False)

# Affiche quelques lignes et le taux de churn pour vérifier le résultat.
print(data.head())
print(f"Taux de churn : {data['churn'].mean():.1%}")