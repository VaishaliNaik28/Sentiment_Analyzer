import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# confidence band → neutral
LOW = 0.40
HIGH = 0.60

# load binary dataset
df = pd.read_csv("data/sentiment140_binary.csv")
df["text"] = df["text"].apply(clean_text)

X = df["text"].values
y = df["label"].values  # 0 (neg) or 2 (pos)

# train a quick confidence model
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=50000)),
    ("clf", LogisticRegression(max_iter=2000))
])

pipe.fit(X_train, y_train)

# predict probabilities
probs = pipe.predict_proba(X)
pos_index = list(pipe.named_steps["clf"].classes_).index(2)
p_pos = probs[:, pos_index]

# create 3-class labels
y3 = np.array(y)
neutral_mask = (p_pos >= LOW) & (p_pos <= HIGH)
y3[neutral_mask] = 1   # Neutral

# save 3-class dataset
df_3 = pd.DataFrame({
    "text": df["text"],
    "label": y3
})

df_3.to_csv("data/sentiment140_3class.csv", index=False)

print("✅ 3-class dataset created")
print(df_3["label"].value_counts())