import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

df = pd.read_csv(
    "data/supply_chain_data.csv"
)

for col in [
    "WarehouseID",
    "SupplierID",
    "ProductCategory"
]:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(
        df[col]
    )

X = df.drop(
    ["DeliveryTime"],
    axis=1
)

y = df["DeliveryTime"]

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
print("R2 Score:", r2)

joblib.dump(
    model,
    "models/supply_chain_model.pkl"
)
