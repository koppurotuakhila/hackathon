from flask import Flask, jsonify
from flask_cors import CORS
import subprocess
import sys
import os
import json

app = Flask(__name__)
CORS(app)

ids_process = None

IDS_SCRIPT = os.path.join(os.getcwd(), "live_ids.py")
PREDICTIONS_FILE = os.path.join(os.getcwd(), "ids_predictions.json")


@app.route("/start")
def start_ids():
    global ids_process

    if ids_process is None or ids_process.poll() is not None:
        print("Starting IDS...")
        ids_process = subprocess.Popen(
            [sys.executable, IDS_SCRIPT],
            cwd=os.getcwd()
        )
        print("IDS PID:", ids_process.pid)
        return jsonify({"status": "IDS started", "pid": ids_process.pid})

    return jsonify({"status": "IDS already running", "pid": ids_process.pid})


@app.route("/stop")
def stop_ids():
    global ids_process

    if ids_process and ids_process.poll() is None:
        print("Stopping IDS...")
        ids_process.terminate()
        ids_process.wait()
        ids_process = None
        return jsonify({"status": "IDS stopped"})

    return jsonify({"status": "IDS not running"})


@app.route("/status")
def status():
    global ids_process
    running = ids_process is not None and ids_process.poll() is None
    return jsonify({
        "running": running,
        "pid": ids_process.pid if running else None
    })


# ✅ NEW: Endpoint the frontend polls every few seconds
@app.route("/predictions")
def get_predictions():
    if not os.path.exists(PREDICTIONS_FILE):
        return jsonify([])

    try:
        with open(PREDICTIONS_FILE, "r") as f:
            data = json.load(f)
        # Return latest 100 entries, newest first
        return jsonify(list(reversed(data[-100:])))
    except (json.JSONDecodeError, IOError) as e:
        print("Error reading predictions:", e)
        return jsonify([])


# ✅ NEW: Clear predictions log
@app.route("/clear")
def clear_predictions():
    try:
        with open(PREDICTIONS_FILE, "w") as f:
            json.dump([], f)
        return jsonify({"status": "cleared"})
    except IOError as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=False)