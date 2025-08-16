# ===== stop_simulator.py =====
import os, signal

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
PID_FILE = os.path.join(DATA_DIR, "simulator.pid")

if os.path.exists(PID_FILE):
    with open(PID_FILE) as f:
        pid = int(f.read().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print("Simulator stopped.")
    except Exception as e:
        print(f"Error stopping simulator: {e}")
    os.remove(PID_FILE)
else:
    print("No running simulator found.")
