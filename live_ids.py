import numpy as np
import pandas as pd
import joblib
import json
import os
import time
from collections import defaultdict
from scapy.all import sniff, IP, TCP
from tensorflow.keras.models import load_model

print("Loading IDS model...")

model = load_model("model/ids_lstm_model.keras")
scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

print("Model loaded successfully\n")

# =============================
# Network Interface (your WiFi)
# =============================
IFACE = "\\Device\\NPF_{4EC6CC30-9AA5-432F-9611-D3CC1767C08D}"

# =============================
# Save predictions
# =============================

def save_prediction(data):
    file = "ids_predictions.json"

    if os.path.exists(file):
        with open(file, "r") as f:
            try:
                predictions = json.load(f)
            except json.JSONDecodeError:
                predictions = []
    else:
        predictions = []

    predictions.append(data)
    predictions = predictions[-200:]  # keep last 200

    with open(file, "w") as f:
        json.dump(predictions, f, indent=4)

# =============================
# Flow storage
# =============================

flows = defaultdict(lambda: {
    "start_time": None,
    "last_time": None,
    "fwd_packets": 0,
    "bwd_packets": 0,
    "fwd_bytes": 0,
    "bwd_bytes": 0,
    "fwd_lengths": [],
    "bwd_lengths": [],
    "timestamps": []
})

FLOW_PACKET_THRESHOLD = 10
FLOW_DURATION_THRESHOLD = 2

# =============================
# Packet processing
# =============================

def process_packet(packet):

    if not packet.haslayer(IP) or not packet.haslayer(TCP):
        return

    src = packet[IP].src
    dst = packet[IP].dst
    sport = packet[TCP].sport
    dport = packet[TCP].dport
    length = len(packet)
    timestamp = time.time()

    flow_key = tuple(sorted([(src, sport), (dst, dport)]))
    flow = flows[flow_key]

    if flow["start_time"] is None:
        flow["start_time"] = timestamp

    flow["last_time"] = timestamp
    flow["timestamps"].append(timestamp)

    if (src, sport) == flow_key[0]:
        flow["fwd_packets"] += 1
        flow["fwd_bytes"] += length
        flow["fwd_lengths"].append(length)
    else:
        flow["bwd_packets"] += 1
        flow["bwd_bytes"] += length
        flow["bwd_lengths"].append(length)

    duration = flow["last_time"] - flow["start_time"]
    total_packets = flow["fwd_packets"] + flow["bwd_packets"]

    if duration < FLOW_DURATION_THRESHOLD or total_packets < FLOW_PACKET_THRESHOLD:
        return

    total_bytes = flow["fwd_bytes"] + flow["bwd_bytes"]
    bytes_per_sec = total_bytes / duration if duration > 0 else 0
    packets_per_sec = total_packets / duration if duration > 0 else 0

    def stats(arr):
        if len(arr) == 0:
            return 0, 0, 0, 0
        return np.mean(arr), np.std(arr), np.max(arr), np.min(arr)

    fwd_mean, fwd_std, fwd_max, fwd_min = stats(flow["fwd_lengths"])
    bwd_mean, bwd_std, bwd_max, bwd_min = stats(flow["bwd_lengths"])

    flow_iat = np.diff(flow["timestamps"])
    flow_iat_mean = np.mean(flow_iat) if len(flow_iat) > 0 else 0
    flow_iat_std  = np.std(flow_iat)  if len(flow_iat) > 0 else 0

    packet_variance = np.var(flow["fwd_lengths"] + flow["bwd_lengths"])

    feature_dict = {
        "Flow Duration":                  duration,
        "Total Fwd Packets":              flow["fwd_packets"],
        "Total Backward Packets":         flow["bwd_packets"],
        "Total Length of Fwd Packets":    flow["fwd_bytes"],
        "Total Length of Bwd Packets":    flow["bwd_bytes"],
        "Flow Bytes/s":                   bytes_per_sec,
        "Flow Packets/s":                 packets_per_sec,
        "Fwd Packet Length Mean":         fwd_mean,
        "Fwd Packet Length Std":          fwd_std,
        "Fwd Packet Length Max":          fwd_max,
        "Fwd Packet Length Min":          fwd_min,
        "Bwd Packet Length Mean":         bwd_mean,
        "Bwd Packet Length Std":          bwd_std,
        "Bwd Packet Length Max":          bwd_max,
        "Bwd Packet Length Min":          bwd_min,
        "Flow IAT Mean":                  flow_iat_mean,
        "Flow IAT Std":                   flow_iat_std,
        "Packet Length Variance":         packet_variance
    }

    df = pd.DataFrame([feature_dict])
    df = df.reindex(columns=feature_columns, fill_value=0)

    X_scaled = scaler.transform(df)
    X_lstm   = X_scaled.reshape(1, 1, X_scaled.shape[1])

    prediction     = model.predict(X_lstm, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)
    label          = label_encoder.inverse_transform(predicted_class)[0]
    confidence     = float(np.max(prediction) * 100)

    data = {
        "time":        time.strftime("%H:%M:%S"),
        "source":      src,
        "destination": dst,
        "prediction":  label,
        "confidence":  round(confidence, 2)
    }

    print(data)
    save_prediction(data)

    flows.pop(flow_key)


print("Real-Time IDS Started on WiFi interface...\n")

sniff(
    prn=process_packet,
    store=0,
    iface=IFACE        # locked to Intel Wireless-AC 9560 (192.168.1.80)
)