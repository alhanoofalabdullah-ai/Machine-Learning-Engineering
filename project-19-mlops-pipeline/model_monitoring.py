import pandas as pd

predictions = pd.read_csv(
    "predictions.csv"
)

print(
    predictions.describe()
)
