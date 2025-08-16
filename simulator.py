# ===== simulator.py =====
import os
import time
import random
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
LIVE_PATH = os.path.join(DATA_DIR, "live.csv")

os.makedirs(DATA_DIR, exist_ok=True)

MACHINE_STATES = ["Running", "Idle", "Fault"]

if not os.path.exists(LIVE_PATH):
    pd.DataFrame(columns=[
        "timestamp", "machine_status", "laser_power_w",
        "gas_pressure_bar", "head_height_mm", "machine_age"
    ]).to_csv(LIVE_PATH, index=False)

print("Simulator started. Generating live data... (Ctrl+C to stop)")

while True:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    machine_status = random.choices(MACHINE_STATES, weights=[0.75, 0.20, 0.05])[0]
    laser_power = random.uniform(900, 1500) if machine_status == "Running" else random.uniform(500, 900)
    gas_pressure = random.uniform(4.5, 6.5) if machine_status == "Running" else random.uniform(3, 5)
    head_height = random.uniform(0.8, 1.5)
    machine_age = round(random.uniform(0.5, 5), 2)

    new_data = pd.DataFrame([{
        "timestamp": timestamp,
        "machine_status": machine_status,
        "laser_power_w": laser_power,
        "gas_pressure_bar": gas_pressure,
        "head_height_mm": head_height,
        "machine_age": machine_age,
    }])

    with open(LIVE_PATH, "a", newline="") as f:
        new_data.to_csv(f, header=f.tell() == 0, index=False)
        f.flush()

    time.sleep(1)
