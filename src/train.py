import re
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import classification_report, confusion_matrix, f1_score


LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}


def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"http\S+|www\S+", " ", s)   # URLs
    s = re.sub(r"@\w+", " ", s)            # mentions
    s = re.sub(r"#[\w-]+", " ", s)         # hashtags
    s = re.sub(r"[^a-z\s]", " ", s)        # keep letters only
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    # 1) Load your FINAL 3-class dataset
    df = pd.read_csv("data/sentiment140_3class.csv")
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected columns: text, label in data/sentiment140_3class.csv")

    # Optional: extra cleaning (safe)
    df["text"] = df["text"].apply(clean_text)

    X = df["text"].values
    y = df["label"].astype(int).values

    # 2) Train/Test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3) Models to compare
    models = {
        "logreg": LogisticRegression(max_iter=2000),
        "naive_bayes": MultinomialNB(),
        "linear_svm": LinearSVC()
    }

    results = []
    best_name = None
    best_f1 = -1
    best_pipe = None

    for name, clf in models.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000)),
            ("clf", clf)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        macro_f1 = f1_score(y_test, y_pred, average="macro")
        results.append((name, macro_f1))

        print("\n" + "=" * 70)
        print(f"MODEL: {name}")
        print(f"Macro F1: {macro_f1:.4f}")
        print("Confusion Matrix (rows=true, cols=pred):")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            labels=[0, 1, 2],
            target_names=[LABEL_NAMES[0], LABEL_NAMES[1], LABEL_NAMES[2]],
            digits=4
        ))

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_name = name
            best_pipe = pipe

    # 4) Print summary
    results.sort(key=lambda x: x[1], reverse=True)
    print("\n" + "=" * 70)
    print("SUMMARY (best → worst by Macro F1):")
    for n, f in results:
        print(f"{n:12s}  Macro-F1={f:.4f}")

    # 5) Save best model pipeline (vectorizer + model together)
    joblib.dump(best_pipe, "models/best_model.joblib")
    print("\n" + "=" * 70)
    print(f"✅ Saved best model: {best_name} (Macro-F1={best_f1:.4f})")
    print("✅ File saved at: models/best_model.joblib")


if __name__ == "__main__":
    main()