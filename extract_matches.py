import pandas as pd
df = pd.read_csv(r"C:\Users\BIT\Downloads\IPL.csv")

matches = df[
    [
        "match_id",
        "batting_team",
        "bowling_team",
        "toss_winner",
        "toss_decision",
        "venue",
        "season",
        "match_won_by"
    ]
]

matches = matches.drop_duplicates(subset="match_id")

matches.to_csv("matches.csv", index=False)

print(matches.head())
print("Total Matches =", len(matches))