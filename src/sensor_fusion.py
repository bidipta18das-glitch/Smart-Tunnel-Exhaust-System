import config
class SensorFusion:

    def __init__(
        self,
        mq2_weight=config.GAS_INDEX_MQ2_WEIGHT,
        mq3_weight=config.GAS_INDEX_MQ3_WEIGHT,
    ):
        self.mq2_weight = mq2_weight
        self.mq3_weight = mq3_weight
        self.sensor_cols = config.SENSOR_COLUMNS

        # We'll compute normalization stats from the training data
        self._mq2_min = None
        self._mq2_max = None
        self._mq3_min = None
        self._mq3_max = None

    # ------------------------------------------------------------------
    # Core fusion
    # ------------------------------------------------------------------
    def fuse(self, df, fit_normalization=True):
       
        print("\n" + "=" * 50)
        print("SENSOR FUSION - Feature Engineering")
        print("=" * 50)

        df = df.copy()

        # --- 1. Gas Index (weighted normalized combination) ---
        if fit_normalization:
            self._mq2_min = df["MQ2_Raw"].min()
            self._mq2_max = df["MQ2_Raw"].max()
            self._mq3_min = df["MQ3_Raw"].min()
            self._mq3_max = df["MQ3_Raw"].max()

        mq2_range = self._mq2_max - self._mq2_min
        mq3_range = self._mq3_max - self._mq3_min

        # Avoid division by zero if all values are identical
        mq2_norm = (
            (df["MQ2_Raw"] - self._mq2_min) / mq2_range
            if mq2_range > 0
            else 0.0
        )
        mq3_norm = (
            (df["MQ3_Raw"] - self._mq3_min) / mq3_range
            if mq3_range > 0
            else 0.0
        )

        df["Gas_Index"] = (
            self.mq2_weight * mq2_norm + self.mq3_weight * mq3_norm
        )

        # --- 2. Heat Stress (convective heat indicator) ---
        df["Heat_Stress"] = df["Temperature"] * (1 - df["Humidity"] / 100)

        # --- 3. MQ2/MQ3 Ratio (smoke vs. vapor discriminator) ---
        df["MQ2_MQ3_Ratio"] = df["MQ2_Raw"] / (df["MQ3_Raw"] + 1)

        # Print summary
        print(f"\nNew features added: Gas_Index, Heat_Stress, MQ2_MQ3_Ratio")
        print(f"Total feature count: {len(config.FUSED_FEATURES)}")
        print(f"\nFused feature statistics:")
        print(
            df[["Gas_Index", "Heat_Stress", "MQ2_MQ3_Ratio"]]
            .describe()
            .round(3)
            .to_string()
        )
        print("=" * 50 + "\n")

        return df

    def get_norm_params(self):
       
        return {
            "mq2_min": self._mq2_min,
            "mq2_max": self._mq2_max,
            "mq3_min": self._mq3_min,
            "mq3_max": self._mq3_max,
        }

    def set_norm_params(self, params):
     
        self._mq2_min = params["mq2_min"]
        self._mq2_max = params["mq2_max"]
        self._mq3_min = params["mq3_min"]
        self._mq3_max = params["mq3_max"]



if __name__ == "__main__":
    data = pd.read_csv(config.RAW_CSV)
    fuser = SensorFusion()
    fused = fuser.fuse(data)
    print(fused[config.FUSED_FEATURES].head(10))
