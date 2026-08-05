import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

print("Loading processed data...")

# -----------------------------
# 1️⃣ Load Processed Data
# -----------------------------
X_train = np.load("model/X_train.npy")
X_test = np.load("model/X_test.npy")
y_train = np.load("model/y_train.npy")
y_test = np.load("model/y_test.npy")

print("Training Shape:", X_train.shape)

# -----------------------------
# 2️⃣ Compute Class Weights (IMPORTANT)
# -----------------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights_dict = dict(enumerate(class_weights))

print("Class Weights:")
print(class_weights_dict)

# -----------------------------
# 3️⃣ Convert Labels to Categorical
# -----------------------------
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_train.shape[1]
print("Number of Classes:", num_classes)

# -----------------------------
# 4️⃣ Build LSTM Model
# -----------------------------
model = Sequential()

model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# 5️⃣ Train Model with Class Weights
# -----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    class_weight=class_weights_dict   # 🔥 This fixes imbalance
)

# -----------------------------
# 6️⃣ Save Model (Modern Format)
# -----------------------------
model.save("model/ids_lstm_model.keras")

print("Model training completed and saved successfully!")