from flask import Flask, render_template, request
import joblib
import re
import os

# Get current directory (app folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go one level up to project root
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Tell Flask where templates are (outside app folder)
template_path = os.path.join(PROJECT_ROOT, "templates")

app = Flask(__name__, template_folder=template_path)

# Load model (outside app folder)
model_path = os.path.join(PROJECT_ROOT, "models", "best_model.joblib")
model = joblib.load(model_path)

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

    if request.method == "POST":
        user_input = request.form.get("text")
        if user_input:
            cleaned = clean_text(user_input)
            result = model.predict([cleaned])[0]
            prediction = LABELS[int(result)]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)