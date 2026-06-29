import os
import pandas as pd
import config
class Preprocessing:

    def __init__(self, input_path, output_path=None):
        self.input_path = input_path
        self.output_path = output_path
        self.data = None
        self.cleaned_data = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_data(self):
        try:
            self.data = pd.read_csv(self.input_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}\n")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def check_data_quality(self):
        """Print a quality report: missing values, duplicates, dtypes."""
        print("Data Quality Report:")
        print("-" * 50)

        missing = self.data.isnull().sum()
        if missing.sum() > 0:
            print(f"Missing values:\n{missing[missing > 0]}\n")
        else:
            print("No missing values\n")

        duplicates = self.data.duplicated().sum()
        print(f"Duplicate rows: {duplicates}\n")

        print(f"Data shape: {self.data.shape}")
        print(f"Data types:\n{self.data.dtypes}\n")

    def validate_sensor_ranges(self):
       
        print("Sensor Range Validation:")
        print("-" * 50)

        df = self.cleaned_data
        total_invalid = 0

        for col, (lo, hi) in config.SENSOR_LIMITS.items():
            if col not in df.columns:
                continue
            out_of_range = (df[col] < lo) | (df[col] > hi)
            count = out_of_range.sum()
            if count > 0:
                print(f"  [!] {col}: {count} values outside [{lo}, {hi}]")
                total_invalid += count
            else:
                print(f"  [OK] {col}: all values within [{lo}, {hi}]")

        if total_invalid > 0:
            # Remove physically impossible readings
            for col, (lo, hi) in config.SENSOR_LIMITS.items():
                if col in df.columns:
                    self.cleaned_data = self.cleaned_data[
                        (self.cleaned_data[col] >= lo)
                        & (self.cleaned_data[col] <= hi)
                    ]
            self.cleaned_data = self.cleaned_data.reset_index(drop=True)
            removed = len(df) - len(self.cleaned_data)
            print(f"\n  Removed {removed} rows with out-of-range sensor values")
        else:
            print("\n  All sensor readings are within physical limits.")
        print()

    def remove_missing_values(self):
        """Drop rows containing any NaN."""
        initial_shape = self.cleaned_data.shape
        self.cleaned_data = self.cleaned_data.dropna()
        removed = initial_shape[0] - self.cleaned_data.shape[0]
        print(f"Removed {removed} rows with missing values")
        print(f"New shape: {self.cleaned_data.shape}\n")

    def remove_duplicates(self):
        """Remove duplicate rows."""
        initial_shape = self.cleaned_data.shape
        self.cleaned_data = self.cleaned_data.drop_duplicates().reset_index(
            drop=True
        )
        removed = initial_shape[0] - self.cleaned_data.shape[0]
        print(f"Removed {removed} duplicate rows")
        print(f"New shape: {self.cleaned_data.shape}\n")

   
    def save_cleaned_data(self):
        """Save the cleaned DataFrame to CSV."""
        if self.output_path is None:
            print("No output path specified, skipping save.")
            return True
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self.cleaned_data.to_csv(self.output_path, index=False)
            print(f"Cleaned data saved successfully")
            print(f"Output file: {self.output_path}")
            print(f"Final shape: {self.cleaned_data.shape}\n")
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False

   
    def get_statistics(self):
        """Return per-column descriptive statistics."""
        return self.cleaned_data[config.SENSOR_COLUMNS].describe()

    def get_summary(self):
        print("Preprocessing Summary:")
        print("=" * 50)
        print(f"Input file:  {self.input_path}")
        if self.output_path:
            print(f"Output file: {self.output_path}")
        print(f"Final shape: {self.cleaned_data.shape}")
        print("=" * 50 + "\n")

    def preprocess(self):
        
        print("\n" + "=" * 50)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("=" * 50 + "\n")

        if self.load_data() is None:
            return None

        self.check_data_quality()
        self.cleaned_data = self.data.copy()

        self.remove_missing_values()
        self.remove_duplicates()
        self.validate_sensor_ranges()

        if config.TARGET_AQI in self.cleaned_data.columns:
            print("Class Distribution:")
            print(self.cleaned_data[config.TARGET_AQI].value_counts())
            print()

        self.save_cleaned_data()
        self.get_summary()

        return self.cleaned_data

def main():
    preprocessor = Preprocessing(config.RAW_CSV, config.CLEANED_CSV)
    cleaned_data = preprocessor.preprocess()
    if cleaned_data is not None:
        print("First few rows of cleaned data:")
        print(cleaned_data.head())
        print("\nData statistics:")
        print(preprocessor.get_statistics())


if __name__ == "__main__":
    main()