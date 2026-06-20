import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

properties = pd.read_csv(
    "data/properties.csv"
)

features = [
    "Price",
    "Bedrooms",
    "Bathrooms",
    "Area"
]

X = properties[features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

similarity_matrix = cosine_similarity(
    X_scaled
)

joblib.dump(
    similarity_matrix,
    "models/recommendation_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Property recommendation model created successfully.")
