import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import config
class TrainModel:

    def __init__(self, data, model_dir=config.SYNTH_MODEL_DIR):
       
        self.data = data
        self.model_dir = model_dir

        # Models
        self.aqi_model = None
        self.fan_model = None

        # Data splits
        self.X = None
        self.y_aqi = None
        self.y_fan = None
        self.X_train = None
        self.X_test = None
        self.y_aqi_train = None
        self.y_aqi_test = None
        self.y_fan_train = None
        self.y_fan_test = None

        # Results
        self.cv_scores_aqi = None
        self.cv_scores_fan = None

    # ------------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------------
    def prepare_data(self):
        
        print("Preparing features and targets...")

        self.X = self.data[config.FUSED_FEATURES]

        # Use rule-generated labels if available, otherwise fall back to CSV
        if "Rule_AQI_Class" in self.data.columns:
            self.y_aqi = self.data["Rule_AQI_Class"]
            print("  Using Rule_AQI_Class (rule-generated) as AQI target")
        else:
            self.y_aqi = self.data[config.TARGET_AQI]
            print(f"  Using {config.TARGET_AQI} (CSV) as AQI target")

        if "Rule_Fan_Speed" in self.data.columns:
            self.y_fan = self.data["Rule_Fan_Speed"]
            print("  Using Rule_Fan_Speed (rule-generated) as fan target")
        elif config.TARGET_FAN in self.data.columns:
            self.y_fan = self.data[config.TARGET_FAN]
            print(f"  Using {config.TARGET_FAN} (CSV) as fan target")
        else:
            # Derive from AQI class
            self.y_fan = self.y_aqi.map(
                lambda c: config.FAN_SPEED_MAP.get(c, config.FAN_SPEED_MAP["Unknown"])["percent"]
            )
            print("  Deriving fan speed from AQI class mapping")

        print(f"\n  Feature matrix shape: {self.X.shape}")
        print(f"  Features: {list(self.X.columns)}")
        print(f"\n  AQI class distribution:\n{self.y_aqi.value_counts().to_string()}")
        print(f"\n  Fan speed distribution:\n{self.y_fan.value_counts().to_string()}\n")
        return True

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    def split_data(self):
        """Stratified train/test split on AQI labels with robust fallbacks."""
        # Check if the dataset is extremely small
        if len(self.X) < 5:
            print("WARNING: Dataset is too small for validation split. Using entire set for training and testing.")
            self.X_train, self.X_test = self.X, self.X
            self.y_aqi_train, self.y_aqi_test = self.y_aqi, self.y_aqi
            self.y_fan_train, self.y_fan_test = self.y_fan, self.y_fan
            return True

        try:
            # First, try standard stratified split
            (
                self.X_train,
                self.X_test,
                self.y_aqi_train,
                self.y_aqi_test,
                self.y_fan_train,
                self.y_fan_test,
            ) = train_test_split(
                self.X,
                self.y_aqi,
                self.y_fan,
                test_size=config.TEST_SIZE,
                random_state=config.SPLIT_RANDOM_STATE,
                stratify=self.y_aqi,
            )
            print("Data split successfully using stratified sampling!")
        except Exception as stratified_err:
            print(f"Stratified split failed: {stratified_err}. Falling back to standard split.")
            try:
                # Fall back to non-stratified split
                (
                    self.X_train,
                    self.X_test,
                    self.y_aqi_train,
                    self.y_aqi_test,
                    self.y_fan_train,
                    self.y_fan_test,
                ) = train_test_split(
                    self.X,
                    self.y_aqi,
                    self.y_fan,
                    test_size=config.TEST_SIZE,
                    random_state=config.SPLIT_RANDOM_STATE,
                    stratify=None,
                )
                print("Data split successfully using standard (non-stratified) sampling.")
            except Exception as std_err:
                print(f"Standard split failed: {std_err}. Falling back to using full dataset.")
                self.X_train, self.X_test = self.X, self.X
                self.y_aqi_train, self.y_aqi_test = self.y_aqi, self.y_aqi
                self.y_fan_train, self.y_fan_test = self.y_fan, self.y_fan

        print(f"  Training set: {self.X_train.shape[0]} samples")
        print(f"  Testing set:  {self.X_test.shape[0]} samples\n")
        return True

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    def _create_rf(self):
        """Create a Random Forest classifier with config hyperparameters."""
        return RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
            random_state=config.RF_RANDOM_STATE,
            n_jobs=config.RF_N_JOBS,
        )

    def train(self):
        """Train both AQI and fan-speed models."""
        print("=" * 50)
        print("TRAINING MODELS")
        print("=" * 50)

        # --- AQI Model ---
        print("\n[1/2] Training AQI classification model...")
        self.aqi_model = self._create_rf()
        self.aqi_model.fit(self.X_train, self.y_aqi_train)

        train_acc = accuracy_score(self.y_aqi_train, self.aqi_model.predict(self.X_train))
        test_acc = accuracy_score(self.y_aqi_test, self.aqi_model.predict(self.X_test))
        print(f"  Train accuracy: {train_acc:.4f}")
        print(f"  Test accuracy:  {test_acc:.4f}")

        # --- Fan Speed Model ---
        print("\n[2/2] Training fan speed classification model...")
        self.fan_model = self._create_rf()
        self.fan_model.fit(self.X_train, self.y_fan_train)

        train_acc_fan = accuracy_score(self.y_fan_train, self.fan_model.predict(self.X_train))
        test_acc_fan = accuracy_score(self.y_fan_test, self.fan_model.predict(self.X_test))
        print(f"  Train accuracy: {train_acc_fan:.4f}")
        print(f"  Test accuracy:  {test_acc_fan:.4f}")

        print("\nBoth models trained successfully!\n")
        return True

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------
    def cross_validate(self):
        """Run stratified/standard k-fold CV on both models with robust fallbacks."""
        num_samples = len(self.X)
        if num_samples < 5:
            print("WARNING: Skipping cross-validation (dataset too small).")
            return

        folds = min(config.CV_FOLDS, num_samples)
        print("=" * 50)
        print(f"CROSS-VALIDATION ({folds}-fold)")
        print("=" * 50)

        # Try StratifiedKFold first
        try:
            skf = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=config.SPLIT_RANDOM_STATE,
            )
            # Verify we have enough samples per class for StratifiedKFold
            class_counts = self.y_aqi.value_counts()
            if class_counts.min() < folds:
                print(f"  [Notice] Least populated class has {class_counts.min()} samples (< {folds} folds).")
                print("  Falling back to standard KFold cross-validation.")
                from sklearn.model_selection import KFold
                skf = KFold(n_splits=folds, shuffle=True, random_state=config.SPLIT_RANDOM_STATE)
            
            # AQI model CV
            aqi_rf = self._create_rf()
            self.cv_scores_aqi = cross_val_score(
                aqi_rf, self.X, self.y_aqi, cv=skf, scoring="accuracy"
            )
            print(f"\nAQI Model CV Scores:  {self.cv_scores_aqi.round(4)}")
            print(f"  Mean: {self.cv_scores_aqi.mean():.4f} +/- {self.cv_scores_aqi.std():.4f}")

            # Fan model CV
            fan_rf = self._create_rf()
            self.cv_scores_fan = cross_val_score(
                fan_rf, self.X, self.y_fan, cv=skf, scoring="accuracy"
            )
            print(f"\nFan Speed Model CV Scores: {self.cv_scores_fan.round(4)}")
            print(f"  Mean: {self.cv_scores_fan.mean():.4f} +/- {self.cv_scores_fan.std():.4f}")
            
        except Exception as cv_err:
            print(f"Cross-validation skipped due to error: {cv_err}")
            
        print("=" * 50 + "\n")

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    def get_feature_importance(self):
        """Print and return feature importances for both models."""
        print("=" * 50)
        print("FEATURE IMPORTANCE")
        print("=" * 50)

        for name, model in [("AQI", self.aqi_model), ("Fan Speed", self.fan_model)]:
            imp_df = pd.DataFrame({
                "Feature": self.X.columns,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=False)

            print(f"\n{name} Model:")
            print(imp_df.to_string(index=False))

        print("=" * 50 + "\n")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save_models(self, fusion_params=None):
        """Save both models + metadata to disk."""
        os.makedirs(self.model_dir, exist_ok=True)

        # Save models
        aqi_path = os.path.join(self.model_dir, "aqi_model.pkl")
        fan_path = os.path.join(self.model_dir, "fan_speed_model.pkl")
        joblib.dump(self.aqi_model, aqi_path)
        joblib.dump(self.fan_model, fan_path)

        # Save metadata
        metadata = {
            "features": list(self.X.columns),
            "aqi_classes": list(self.y_aqi.unique()),
            "fan_speeds": sorted(list(self.y_fan.unique())),
            "cv_scores_aqi": (
                self.cv_scores_aqi.tolist() if self.cv_scores_aqi is not None else None
            ),
            "cv_scores_fan": (
                self.cv_scores_fan.tolist() if self.cv_scores_fan is not None else None
            ),
            "fusion_params": fusion_params,
        }
        meta_path = os.path.join(self.model_dir, "model_metadata.pkl")
        joblib.dump(metadata, meta_path)

        print(f"Models saved to {self.model_dir}:")
        print(f"  AQI model:     {aqi_path}")
        print(f"  Fan model:     {fan_path}")
        print(f"  Metadata:      {meta_path}\n")
        return True

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def train_pipeline(self, fusion_params=None):
        """Execute the complete training pipeline."""
        print("\n" + "=" * 50)
        print("STARTING MODEL TRAINING PIPELINE")
        print("=" * 50 + "\n")

        if not self.prepare_data():
            return False
        if not self.split_data():
            return False
        if not self.train():
            return False

        self.cross_validate()
        self.get_feature_importance()

        if not self.save_models(fusion_params):
            return False

        print("Training pipeline completed successfully!")
        return True


# ======================================================================
# Standalone entry point
# ======================================================================
if __name__ == "__main__":
    from sensor_fusion import SensorFusion

    data = pd.read_csv(config.CLEANED_CSV)
    fuser = SensorFusion()
    data = fuser.fuse(data)

    trainer = TrainModel(data)
    trainer.train_pipeline(fusion_params=fuser.get_norm_params())
