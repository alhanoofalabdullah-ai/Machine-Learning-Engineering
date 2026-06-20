import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv(
    "data/resumes.csv"
)

X = df["ResumeText"]
y = df["HiringDecision"]

vectorizer = TfidfVectorizer(
    max_features=5000
)

X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
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
    "models/resume_classifier.pkl"
)

joblib.dump(
    vectorizer,
    "models/vectorizer.pkl"
)
