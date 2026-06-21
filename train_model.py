import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("matches.csv")

# Remove missing winners
df = df.dropna(subset=["match_won_by"])

# Features
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

# Target
y = df["match_won_by"]

# Encode categorical columns
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y.astype(str))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, pred)

print("Accuracy =", round(acc * 100, 2), "%")