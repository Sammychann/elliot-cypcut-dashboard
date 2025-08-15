import os
import time
import random
import pandas as pd
from datetime import datetime

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LIVE_PATH = os.path.join(DATA_DIR, "live.csv")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Machine states
MACHINE_STATES = ["Running", "Idle", "Fault"]

# If file doesn't exist, create it with headers
if not os.path.exists(LIVE_PATH):
    df_init = pd.DataFrame(columns=[
        "timestamp",
        "machine_status",
        "laser_power_w",
        "gas_pressure_bar",
        "head_height_mm",
        "machine_age"
    ])
    df_init.to_csv(LIVE_PATH, index=False)

print("Simulator started. Generating live data...")

while True:
    # Generate realistic values
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    machine_status = random.choices(MACHINE_STATES, weights=[0.75, 0.20, 0.05])[0]  # mostly running
    laser_power = random.uniform(900, 1500) if machine_status == "Running" else random.uniform(500, 900)
    gas_pressure = random.uniform(4.5, 6.5) if machine_status == "Running" else random.uniform(3, 5)
    head_height = random.uniform(0.8, 1.5)
    machine_age = round(random.uniform(0.5, 5), 2)  # years

    # Append to CSV
    new_data = pd.DataFrame([{
        "timestamp": timestamp,
        "machine_status": machine_status,
        "laser_power_w": laser_power,
        "gas_pressure_bar": gas_pressure,
        "head_height_mm": head_height,
        "machine_age": machine_age
    }])
    new_data.to_csv(LIVE_PATH, mode="a", header=not os.path.exists(LIVE_PATH) or os.stat(LIVE_PATH).st_size == 0, index=False)

    # Wait before next reading
    time.sleep(1)
