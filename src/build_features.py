import pandas as pd, os

RAW = ["motor_current", "rated_motor_current", "vibration_level", "belt_age_months",
       "parcels_per_min", "dust_index", "days_since_maintenance", "humidity"]
FEATURES = ["motor_load_ratio", "belt_wear_pressure_index", "vibration_level",
            "dust_index", "days_since_maintenance", "humidity",
            "parcels_per_min", "belt_age_months"]

def preprocess(df):
    df = df.drop_duplicates("shift_id").copy()
    for c in RAW:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    df = df[df["rated_motor_current"] > 0]
    df["dust_index"] = df["dust_index"].clip(0, 100)
    df["humidity"] = df["humidity"].clip(0, 100)
    return df

def engineer(df):
    df = df.copy()
    df["motor_load_ratio"] = df["motor_current"] / df["rated_motor_current"]
    df["belt_wear_pressure_index"] = (df["belt_age_months"] * df["parcels_per_min"]
                                      * (1 + df["dust_index"] / 100)) / 100
    return df

def main():
    frames = [preprocess(pd.read_csv(f"data/raw/conveyor_v{v}.csv",
                                     parse_dates=["event_timestamp"])) for v in (1, 2)]
    df = engineer(pd.concat(frames, ignore_index=True))
    os.makedirs("feature_repo/data", exist_ok=True)
    feats = df[["shift_id", "event_timestamp"] + FEATURES].astype({f: "float32" for f in FEATURES})
    feats.to_parquet("feature_repo/data/conveyor_features.parquet", index=False)
    df[["shift_id", "event_timestamp", "dataset_version", "next_shift_jam"]] \
        .to_parquet("data/labels.parquet", index=False)
    print("Saved", feats.shape, "feature rows")

if __name__ == "__main__":
    main()
