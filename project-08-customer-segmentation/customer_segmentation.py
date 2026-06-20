import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv(
    "data/customers.csv"
)

features = [
    "AnnualIncome",
    "SpendingScore",
    "TotalPurchases"
]

X = df[features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=5,
    random_state=42
)

df["Cluster"] = kmeans.fit_predict(
    X_scaled
)

print(
    df[[
        "CustomerID",
        "Cluster"
    ]].head()
)

joblib.dump(
    kmeans,
    "models/kmeans_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)
