import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

print("Loading dataset...")

# 🔹 Change this to your dataset path
df = pd.read_csv("dataset/cicids2017_cleaned.csv")

print("Dataset Loaded:", df.shape)

# -----------------------------
# 1️⃣ Separate Features & Target
# -----------------------------

# IMPORTANT: Your target column is "Attack Type"
X = df.drop("Attack Type", axis=1)
y = df["Attack Type"]

# -----------------------------
# 2️⃣ Encode Target Labels
# -----------------------------

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder
joblib.dump(label_encoder, "model/label_encoder.pkl")

# -----------------------------
# 3️⃣ Scale Features
# -----------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "model/scaler.pkl")

# Save feature column names (VERY IMPORTANT)
joblib.dump(X.columns.tolist(), "model/feature_columns.pkl")

# -----------------------------
# 4️⃣ Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# -----------------------------
# 5️⃣ Reshape for LSTM
# -----------------------------

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# -----------------------------
# 6️⃣ Save Processed Data
# -----------------------------

np.save("model/X_train.npy", X_train)
np.save("model/X_test.npy", X_test)
np.save("model/y_train.npy", y_train)
np.save("model/y_test.npy", y_test)

print("Data preparation completed successfully!")