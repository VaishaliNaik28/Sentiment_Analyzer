import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("models/best_model.joblib")

# Extract TF-IDF and classifier
vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["clf"]

# Example text
text = ["This product is absolutely amazing and works perfectly"]

# Transform text
X = vectorizer.transform(text)

# Predict class
predicted_class = classifier.predict(X)[0]
print("Predicted Class:", predicted_class)

# Get feature names
feature_names = vectorizer.get_feature_names_out()

# Convert sparse matrix to dense
X_dense = X.toarray()[0]

# Get coefficients for predicted class
if len(classifier.coef_.shape) > 1:
    class_coef = classifier.coef_[predicted_class]
else:
    class_coef = classifier.coef_[0]

# Calculate word contributions
contributions = X_dense * class_coef

# Create dataframe
df = pd.DataFrame({
    "word": feature_names,
    "contribution": contributions
})

# Remove zero values
df = df[df["contribution"] != 0]

# Sort by absolute contribution
df["abs_contribution"] = np.abs(df["contribution"])
top_words = df.sort_values("abs_contribution", ascending=False).head(10)

print("\nTop words influencing prediction:\n")
print(top_words[["word", "contribution"]])
