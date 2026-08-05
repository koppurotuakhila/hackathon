import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

print("Prepare data script started...")

# -----------------------------
# Ensure model folder exists
# -----------------------------
if not os.path.exists("model"):
    os.makedirs("model")
    print("Model folder created!")

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
print("Loading dataset...")
df = pd.read_csv("dataset/cicids2017_cleaned.csv")

# -----------------------------
# 2️⃣ Reduce Size (for faster training)
# -----------------------------
df = df.sample(n=200000, random_state=42)
print("Dataset Shape After Sampling:", df.shape)

# -----------------------------
# 3️⃣ Separate Features & Target
# -----------------------------
X = df.drop("Attack Type", axis=1)
y = df["Attack Type"]

# -----------------------------
# 4️⃣ Encode Labels
# -----------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Attack Classes:")
print(label_encoder.classes_)

joblib.dump(label_encoder, "model/label_encoder.pkl")

# -----------------------------
# 5️⃣ Scale Features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "model/scaler.pkl")

# -----------------------------
# 6️⃣ Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# -----------------------------
# 7️⃣ Reshape for LSTM
# -----------------------------
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# -----------------------------
# 8️⃣ Save Processed Data
# -----------------------------
np.save("model/X_train.npy", X_train)
np.save("model/X_test.npy", X_test)
np.save("model/y_train.npy", y_train)
np.save("model/y_test.npy", y_test)

print("Files saved successfully inside model folder!")
print("Data preparation completed successfully!")