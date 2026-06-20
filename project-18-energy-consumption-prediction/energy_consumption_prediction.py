import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

df = pd.read_csv(
    "data/energy_usage.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Year"] = df["Date"].dt.year

X = df.drop(
    ["Date", "EnergyConsumption"],
    axis=1
)

y = df["EnergyConsumption"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor(
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

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

print("MAE:", mae)
print("R2 Score:", r2)

joblib.dump(
    model,
    "models/energy_prediction_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)
