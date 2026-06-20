import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv(
    "data/structured_data.csv"
)

X = df.drop(
    "Target",
    axis=1
)

y = df["Target"]

model = RandomForestClassifier()

model.fit(
    X,
    y
)

joblib.dump(
    model,
    "models/enterprise_model.pkl"
)
