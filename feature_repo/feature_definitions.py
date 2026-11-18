from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32

shift = Entity(name="shift", join_keys=["shift_id"])

source = FileSource(path="data/conveyor_features.parquet",
                    timestamp_field="event_timestamp")

names = ["motor_load_ratio", "belt_wear_pressure_index", "vibration_level", "dust_index",
         "days_since_maintenance", "humidity", "parcels_per_min", "belt_age_months"]

conveyor_features = FeatureView(
    name="conveyor_features",
    entities=[shift],
    ttl=timedelta(days=3650),
    schema=[Field(name=n, dtype=Float32) for n in names],
    source=source,
)
