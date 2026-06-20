import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse
import joblib

ratings = pd.read_csv(
    "data/user_ratings.csv"
)

reader = Reader(
    rating_scale=(1, 5)
)

data = Dataset.load_from_df(
    ratings[["UserID", "ProductID", "Rating"]],
    reader
)

trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

model = SVD()

model.fit(trainset)

predictions = model.test(testset)

rmse(predictions)

joblib.dump(
    model,
    "models/recommendation_model.pkl"
)
