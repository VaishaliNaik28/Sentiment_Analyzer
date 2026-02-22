import pandas as pd

# Sentiment140 format: target, id, date, flag, user, text
df = pd.read_csv(
    "data/sentiment140.csv",
    encoding="latin-1",
    header=None
)

df = df[[0, 5]]
df.columns = ["label", "text"]

# Keep only 0 and 4
df = df[df["label"].isin([0, 4])].copy()

# Map: 0->0 (neg), 4->2 (pos)
df["label"] = df["label"].map({0: 0, 4: 2})

# Save clean binary dataset
df.to_csv("data/sentiment140_binary.csv", index=False)

print("✅ Saved: data/sentiment140_binary.csv")
print(df["label"].value_counts())