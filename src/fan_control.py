
import os
import time
import joblib
import pandas as pd
import numpy as np
import config
from sensor_fusion import SensorFusion
class FanController:

    def __init__(self, model_dir=config.SYNTH_MODEL_DIR):
        self.model_dir = model_dir
        self.aqi_model = None
        self.fan_model = None
        self.fuser = SensorFusion()
        self.metadata = None
        self._loaded = False

    
    def load(self):
        """Load trained models and metadata from disk."""
        try:
            self.aqi_model = joblib.load(
                os.path.join(self.model_dir, "aqi_model.pkl")
            )
            self.fan_model = joblib.load(
                os.path.join(self.model_dir, "fan_speed_model.pkl")
            )
            self.metadata = joblib.load(
                os.path.join(self.model_dir, "model_metadata.pkl")
            )

            if self.metadata.get("fusion_params"):
                self.fuser.set_norm_params(self.metadata["fusion_params"])

            self._loaded = True
            print("Fan controller loaded successfully.\n")
            return True
        except Exception as e:
            print(f"Error loading fan controller: {e}")
            return False


    def predict_and_command(self, sensor_readings):
        
        if not self._loaded:
            raise RuntimeError("Call load() first.")

        # Build a single-row DataFrame with fused features
        row_df = pd.DataFrame([sensor_readings])
        row_df = self.fuser.fuse(row_df, fit_normalization=False)

        # Predict
        features = row_df[config.FUSED_FEATURES]
        aqi_class = self.aqi_model.predict(features)[0]
        fan_percent = int(self.fan_model.predict(features)[0])

        # Map to full command
        fan_info = config.FAN_SPEED_MAP.get(
            aqi_class, config.FAN_SPEED_MAP["Unknown"]
        )

        return {
            "aqi_class": aqi_class,
            "fan_speed_percent": fan_info["percent"],
            "fan_speed_label": fan_info["label"],
            "sensor_readings": sensor_readings,
        }

    @staticmethod
    def format_serial_command(command):
        
        return f"FAN:{command['fan_speed_percent']}"

    @staticmethod
    def format_mqtt_json(command):
        
        import json
        return json.dumps({
            "aqi": command["aqi_class"],
            "fan": command["fan_speed_percent"],
        })

    @staticmethod
    def get_fan_command(fan_speed_label):
       
        for aqi_class, info in config.FAN_SPEED_MAP.items():
            if info["label"] == fan_speed_label:
                return {
                    "speed_percent": info["percent"],
                }
        return {"speed_percent": 0}

   
    def simulate_live(self, csv_path, delay=0.5, max_rows=20):
        
        if not self._loaded:
            self.load()

        data = pd.read_csv(csv_path)
        if max_rows > 0:
            data = data.head(max_rows)

        print("\n" + "=" * 70)
        print("FAN CONTROL SIMULATION - Live Sensor Feed")
        print("=" * 70)
        print(
            f"{'#':>3} | {'Temp':>5} | {'Hum':>5} | {'MQ2':>5} | {'MQ3':>5} | "
            f"{'AQI Class':<23} | {'Fan':>4}% | {'CMD'}"
        )
        print("-" * 70)

        for idx, row in data.iterrows():
            readings = {
                "Temperature": row["Temperature"],
                "Humidity": row["Humidity"],
                "MQ2_Raw": row["MQ2_Raw"],
                "MQ3_Raw": row["MQ3_Raw"],
            }

            cmd = self.predict_and_command(readings)
            serial = self.format_serial_command(cmd)

            print(
                f"{idx+1:>3} | {readings['Temperature']:>5.1f} | "
                f"{readings['Humidity']:>5.1f} | {readings['MQ2_Raw']:>5} | "
                f"{readings['MQ3_Raw']:>5} | {cmd['aqi_class']:<23} | "
                f"{cmd['fan_speed_percent']:>4}% | "
                f"{serial}"
            )

            time.sleep(delay)

        print("=" * 70)
        print("Simulation complete.\n")


if __name__ == "__main__":
    controller = FanController()
    controller.load()

    print("--- Single Prediction Demo ---")
    test_reading = {"Temperature": 35.0, "Humidity": 38.0, "MQ2_Raw": 750, "MQ3_Raw": 180}
    cmd = controller.predict_and_command(test_reading)
    print(f"Reading:  {test_reading}")
    print(f"AQI:      {cmd['aqi_class']}")
    print(f"Fan:      {cmd['fan_speed_percent']}% ({cmd['fan_speed_label']})")
    print(f"Serial:   {FanController.format_serial_command(cmd)}")
    print(f"MQTT:     {FanController.format_mqtt_json(cmd)}")

    controller.simulate_live(config.RAW_CSV, delay=0.1, max_rows=10)
