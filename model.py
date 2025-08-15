import os
import pandas as pd
import joblib
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_PATH = os.path.join(DATA_DIR, "fault_model.pkl")

def train_model():
    # Generate dummy training data
    data = []
    for _ in range(500):
        laser_power = random.uniform(500, 1600)
        gas_pressure = random.uniform(2, 8)
        head_height = random.uniform(0, 2)
        machine_age = random.uniform(0.5, 5.0)
        is_running = random.choice([0, 1])
        fault = 1 if (laser_power > 1500 and gas_pressure < 3) else 0
        data.append([laser_power, gas_pressure, head_height, machine_age, is_running, fault])

    df = pd.DataFrame(data, columns=["laser_power_w", "gas_pressure_bar", "head_height_mm", "machine_age", "is_running", "fault"])

    X = df[["laser_power_w", "gas_pressure_bar", "head_height_mm", "machine_age", "is_running"]]
    y = df["fault"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
