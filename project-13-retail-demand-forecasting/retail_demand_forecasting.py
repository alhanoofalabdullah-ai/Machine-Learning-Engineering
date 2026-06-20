import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

df = pd.read_csv(
    "data/retail_sales.csv"
)

df["SalesDate"] = pd.to_datetime(
    df["SalesDate"]
)

df["Year"] = df["SalesDate"].dt.year
df["Month"] = df["SalesDate"].dt.month
df["Day"] = df["SalesDate"].dt.day

for col in [
    "ProductCategory",
    "StoreLocation"
]:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(
        df[col]
    )

X = df.drop(
    ["SalesDate", "DemandQuantity"],
    axis=1
)

y = df["DemandQuantity"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

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
print("R2:", r2)

joblib.dump(
    model,
    "models/demand_forecasting_model.pkl"
)
