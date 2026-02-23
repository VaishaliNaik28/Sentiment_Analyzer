from flask import Flask, render_template, request
import joblib
import re
import os
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

template_path = os.path.join(PROJECT_ROOT, "templates")
app = Flask(__name__, template_folder=template_path)

# Load model
model_path = os.path.join(PROJECT_ROOT, "models", "best_model.joblib")
model = joblib.load(model_path)

# Extract vectorizer & classifier
vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["clf"]

LABELS = {
    0: "Negative 😡",
    1: "Neutral 😐",
    2: "Positive 😄"
}

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"#[\w-]+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    words = None
    user_input = ""  # 👈 store text so it doesn't disappear

    if request.method == "POST":
        user_input = request.form.get("text")

        if user_input:
            cleaned = clean_text(user_input)

            X = vectorizer.transform([cleaned])
            result = classifier.predict(X)[0]
            prediction = LABELS[int(result)]

            # ===== Explainability =====
            feature_names = vectorizer.get_feature_names_out()
            X_dense = X.toarray()[0]

            if len(classifier.coef_.shape) > 1:
                class_coef = classifier.coef_[int(result)]
            else:
                class_coef = classifier.coef_[0]

            contributions = X_dense * class_coef

            word_contributions = []

            for word, value in zip(feature_names, contributions):
                if value != 0:
                    word_contributions.append((word, round(float(value), 4)))

            word_contributions = sorted(
                word_contributions,
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]

            words = word_contributions

    return render_template(
        "index.html",
        prediction=prediction,
        words=words,
        user_input=user_input   # 👈 send back to UI
    )


if __name__ == "__main__":
    app.run(debug=True)
