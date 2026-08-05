import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

print("Loading model and test data...")

# -----------------------------
# 1️⃣ Load NEW Trained Model (.keras)
# -----------------------------
model = load_model("model/ids_lstm_model.keras")

# -----------------------------
# 2️⃣ Load Test Data
# -----------------------------
X_test = np.load("model/X_test.npy")
y_test = np.load("model/y_test.npy")

# -----------------------------
# 3️⃣ Load Label Encoder
# -----------------------------
label_encoder = joblib.load("model/label_encoder.pkl")

# -----------------------------
# 4️⃣ Make Predictions
# -----------------------------
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# -----------------------------
# 5️⃣ Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# -----------------------------
# 6️⃣ Classification Report
# -----------------------------
print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

# -----------------------------
# 7️⃣ Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm,
                     index=label_encoder.classes_,
                     columns=label_encoder.classes_)

print("\nConfusion Matrix:\n")
print(cm_df)