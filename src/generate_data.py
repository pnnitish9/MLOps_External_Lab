import numpy as np, pandas as pd, os

def generate(n=300, version=1, seed=42):
    rng = np.random.default_rng(seed + version)
    festival = version == 2
    rated = rng.choice([30.0, 40.0, 50.0], n)
    parcels = rng.normal(40 * (1.3 if festival else 1.0), 8, n).clip(10, None)
    belt_age = rng.integers(1, 96, n).astype(float)
    dust = rng.normal(52 if festival else 40, 18, n).clip(0, 100)
    days = rng.integers(0, 150 if festival else 120, n).astype(float)
    humidity = rng.normal(55, 12, n).clip(20, 95)
    vib = (1.5 + 0.03*belt_age + 0.02*dust + 0.02*parcels + rng.normal(0, 0.6, n)) \
          * (1.15 if festival else 1.0)
    vib = vib.clip(0.1, None)
    load = 0.45 + 0.008*parcels + 0.002*dust + rng.normal(0, 0.05, n)
    current = rated * load
    z = -11.5 + 6*load + 0.5*vib + 0.02*dust + 0.015*days + 0.01*belt_age + 0.02*(humidity-55)
    jam = (rng.random(n) < 1/(1+np.exp(-z))).astype(int)

    df = pd.DataFrame({
        "shift_id": [f"V{version}-{i:04d}" for i in range(n)],
        "conveyor_id": rng.choice([f"C{i:02d}" for i in range(1, 11)], n),
        "event_timestamp": pd.date_range("2026-08-01" if festival else "2026-01-01",
                                         periods=n, freq="8h", tz="UTC"),
        "motor_current": current.round(2),
        "rated_motor_current": rated,
        "vibration_level": vib.round(2),
        "belt_age_months": belt_age,
        "parcels_per_min": parcels.round(1),
        "dust_index": dust.round(1),
        "days_since_maintenance": days,
        "humidity": humidity.round(1),
        "next_shift_jam": jam,
        "dataset_version": version,
    })
    # inject a few missing values so preprocessing has real work to do
    df.loc[rng.random(n) < 0.02, "humidity"] = np.nan
    return df

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    for v in (1, 2):
        d = generate(300, v)
        d.to_csv(f"data/raw/conveyor_v{v}.csv", index=False)
        print(f"V{v}: {d.shape}, jam rate = {d.next_shift_jam.mean():.2%}, "
              f"mean parcels/min = {d.parcels_per_min.mean():.1f}")
