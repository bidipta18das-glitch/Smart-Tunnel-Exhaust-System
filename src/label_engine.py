
import numpy as np
import pandas as pd
import config
class LabelEngine:
    def __init__(self, tolerance=0.10):
        self.thresholds = config.AQI_THRESHOLDS
        self.priority = config.AQI_PRIORITY
        self.fan_map = config.FAN_SPEED_MAP
        self.tolerance = tolerance

    def _in_range(self, value, lo, hi):
        margin = (hi - lo) * self.tolerance
        return (lo - margin) <= value <= (hi + margin)

    def classify_row(self, row):
       
        for cls in self.priority:
            ranges = self.thresholds[cls]
            match = all(
                self._in_range(row[col], lo, hi)
                for col, (lo, hi) in ranges.items()
            )
            if match:
                return cls
        return self._nearest_class(row)

    def _nearest_class(self, row):
        best_class = "Normal"
        best_dist = float("inf")

        for cls in self.priority:
            ranges = self.thresholds[cls]
            dist = 0
            for col, (lo, hi) in ranges.items():
                mid = (lo + hi) / 2
                span = (hi - lo) if (hi - lo) > 0 else 1
                dist += ((row[col] - mid) / span) ** 2
            dist = np.sqrt(dist)
            if dist < best_dist:
                best_dist = dist
                best_class = cls

        return best_class

    def get_fan_speed(self, aqi_class):
        
        return self.fan_map.get(aqi_class, self.fan_map["Unknown"])["percent"]

    def apply_labels(self, df):
       
        print("\n" + "=" * 50)
        print("LABEL ENGINE - Applying engineering rules")
        print("=" * 50)

        df = df.copy()
        df["Rule_AQI_Class"] = df.apply(self.classify_row, axis=1)
        df["Rule_Fan_Speed"] = df["Rule_AQI_Class"].map(self.get_fan_speed)

        print("\nRule-based AQI class distribution:")
        print(df["Rule_AQI_Class"].value_counts().to_string())
        print("\nRule-based fan speed distribution:")
        print(df["Rule_Fan_Speed"].value_counts().to_string())
        print("=" * 50 + "\n")

        return df

    def validate_labels(self, df):
        
        print("\n" + "=" * 50)
        print("LABEL VALIDATION - CSV vs. Engineering Rules")
        print("=" * 50)

        report = {"aqi_mismatches": 0, "fan_mismatches": 0, "details": []}

        if config.TARGET_AQI not in df.columns:
            print("No existing AQI labels to validate against.")
            return report

        # AQI class comparison
        aqi_match = df[config.TARGET_AQI] == df["Rule_AQI_Class"]
        aqi_mismatches = (~aqi_match).sum()
        report["aqi_mismatches"] = int(aqi_mismatches)

        # Fan speed comparison
        if config.TARGET_FAN in df.columns:
            fan_match = df[config.TARGET_FAN] == df["Rule_Fan_Speed"]
            fan_mismatches = (~fan_match).sum()
            report["fan_mismatches"] = int(fan_mismatches)
        else:
            fan_mismatches = "N/A"

        print(f"\nTotal rows:           {len(df)}")
        print(f"AQI class mismatches: {aqi_mismatches}")
        print(f"Fan speed mismatches: {fan_mismatches}")

        if aqi_mismatches > 0:
            mismatch_df = df[~aqi_match][
                config.SENSOR_COLUMNS
                + [config.TARGET_AQI, "Rule_AQI_Class"]
            ]
            print(f"\nFirst 10 AQI mismatches:")
            print(mismatch_df.head(10).to_string(index=False))
            report["details"] = mismatch_df.head(20).to_dict("records")
        else:
            print("\n[OK] All labels match the engineering rules perfectly.")

        print("=" * 50 + "\n")
        return report


if __name__ == "__main__":
    data = pd.read_csv(config.RAW_CSV)
    engine = LabelEngine()
    data = engine.apply_labels(data)
    report = engine.validate_labels(data)
    print(f"Mismatches: {report['aqi_mismatches']}")
