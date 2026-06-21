import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("matches.csv")
df = df.dropna(subset=["match_won_by"])

X = df[
    [
        "batting_team",
        "bowling_team",
        "toss_winner",
        "toss_decision",
        "venue",
        "season"
    ]
]

y = df["match_won_by"]

for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y.astype(str))

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "ipl_model.pkl")

print("Model saved successfully!")