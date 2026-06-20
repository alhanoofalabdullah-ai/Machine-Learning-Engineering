import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv(
    "data/equipment_sensors.csv"
)

X = df.drop(
    ["EquipmentID", "FailureStatus"],
    axis=1
)

y = df["FailureStatus"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

print(
    classification_report(
        y_test,
        predictions
    )
)

joblib.dump(
    model,
    "models/maintenance_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)
