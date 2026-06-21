import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("matches.csv")
df = df.dropna(subset=["match_won_by"])

# Create encoders
encoders = {}

for col in [
    "batting_team",
    "bowling_team",
    "toss_winner",
    "toss_decision",
    "venue",
    "season"
]:
    le = LabelEncoder()
    le.fit(df[col].astype(str))
    encoders[col] = le

# Target encoder
target_encoder = LabelEncoder()
target_encoder.fit(df["match_won_by"].astype(str))

# Load trained model
model = joblib.load("ipl_model.pkl")

# UI
st.title("🏏 IPL Match Winner Prediction")

teams = sorted(df["batting_team"].unique())

team1 = st.selectbox(
    "Batting Team",
    teams
)

available_teams = [t for t in teams if t != team1]

team2 = st.selectbox(
    "Bowling Team",
    available_teams
)

toss_winner = st.selectbox(
    "Toss Winner",
    teams
)

toss_decision = st.selectbox(
    "Toss Decision",
    ["bat", "field"]
)

venue = st.selectbox(
    "Venue",
    sorted(df["venue"].astype(str).unique())
)

season = st.selectbox(
    "Season",
    sorted(df["season"].astype(str).unique())
)

# Prediction
if st.button("Predict Winner"):

    data = pd.DataFrame(
        [[
            team1,
            team2,
            toss_winner,
            toss_decision,
            venue,
            season
        ]],
        columns=[
            "batting_team",
            "bowling_team",
            "toss_winner",
            "toss_decision",
            "venue",
            "season"
        ]
    )

    for col in data.columns:
        data[col] = encoders[col].transform(
            data[col].astype(str)
        )

    pred = model.predict(data)
    proba = model.predict_proba(data)[0]

    winner = target_encoder.inverse_transform(pred)[0]
    confidence = max(proba) * 100

    st.success(
        f"🏆 Predicted Winner: {winner}"
    )

    st.info(
        f"📊 Confidence: {confidence:.2f}%"
    )
