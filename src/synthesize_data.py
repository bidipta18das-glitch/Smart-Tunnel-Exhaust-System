
import os
import sys
import numpy as np
import pandas as pd

# Ensure the src directory is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

def generate_class_samples(class_name, num_samples, seed=42):
    """
    Generate synthetic sensor readings within the threshold boundaries for a specific class.
    """
    np.random.seed(seed)
    ranges = config.AQI_THRESHOLDS[class_name]
    
    # Generate sensor columns uniformly within their boundaries
    temp = np.random.uniform(ranges["Temperature"][0], ranges["Temperature"][1], num_samples)
    hum = np.random.uniform(ranges["Humidity"][0], ranges["Humidity"][1], num_samples)
    mq2 = np.random.randint(ranges["MQ2_Raw"][0], ranges["MQ2_Raw"][1] + 1, num_samples)
    mq3 = np.random.randint(ranges["MQ3_Raw"][0], ranges["MQ3_Raw"][1] + 1, num_samples)
    
    # Format according to hardware sensor precision (1 decimal place for DHT11, integers for ADC)
    temp = np.round(temp, 1)
    hum = np.round(hum, 1)
    
    # Create DataFrame
    df = pd.DataFrame({
        "Temperature": temp,
        "Humidity": hum,
        "MQ2_Raw": mq2,
        "MQ3_Raw": mq3
    })
    
    # Add targets
    df["Future_AQI_Class"] = class_name
    df["Fan_Speed_Percent"] = config.FAN_SPEED_MAP[class_name]["percent"]
    
    return df

def main():
    print("=" * 60)
    print("STARTING DATA SYNTHESIS")
    print("=" * 60)
    
    # Target counts for each class to get 50,000 total rows
    class_targets = {
        "Normal": 15000,
        "Increased Ventilation": 12500,
        "Hazardous Smoke": 12500,
        "Chemical Vapor": 10000
    }
    
    dfs = []
    # Seed per class to ensure variety and reproducibility
    seeds = [42, 100, 2026, 999]
    
    for (class_name, count), seed in zip(class_targets.items(), seeds):
        print(f"Generating {count} samples for class: '{class_name}'...")
        class_df = generate_class_samples(class_name, count, seed=seed)
        dfs.append(class_df)
        
    # Combine all classes
    dataset = pd.concat(dfs, ignore_index=True)
    
    # Shuffle the dataset to simulate real sequential or randomized monitoring stream
    dataset = dataset.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Print statistics
    print("\nSynthesized Dataset Info:")
    print(f"  Total Rows: {len(dataset)}")
    print(f"  Columns:    {list(dataset.columns)}")
    print("\nClass Distribution:")
    print(dataset["Future_AQI_Class"].value_counts())
    
    # Ensure raw output directory exists
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    
    # Save to RAW_CSV path
    dataset.to_csv(config.RAW_CSV, index=False)
    print(f"\n[SUCCESS] Saved synthesized dataset to: {config.RAW_CSV}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
