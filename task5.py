
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

# ------------------------------------------------
# 1. Project setup
# ------------------------------------------------

sns.set_theme(style="whitegrid")
os.makedirs("outputs", exist_ok=True)

FILE_NAME = "dataset_extracted/US_Accidents_March23.csv"
CHUNK_SIZE = 200000

print("==========================================")
print("   PRODIGY DATA SCIENCE - TASK 05")
print("       TRAFFIC ACCIDENT ANALYSIS")
print("==========================================")

required_columns = [
    "Severity", "Start_Time", "Start_Lat", "Start_Lng",
    "State", "Weather_Condition",
    "Amenity", "Bump", "Crossing", "Give_Way",
    "Junction", "No_Exit", "Railway", "Roundabout",
    "Station", "Stop", "Traffic_Calming", "Traffic_Signal"
]

road_columns = [
    "Amenity", "Bump", "Crossing", "Give_Way",
    "Junction", "No_Exit", "Railway", "Roundabout",
    "Station", "Stop", "Traffic_Calming", "Traffic_Signal"
]

# ------------------------------------------------
# 2. Initialize summary data
# ------------------------------------------------

severity_counts = pd.Series(dtype="int64")
hour_counts = pd.Series(dtype="int64")
weekday_counts = pd.Series(dtype="int64")
state_counts = pd.Series(dtype="int64")
weather_counts = pd.Series(dtype="int64")

road_counts = pd.Series(
    0, index=road_columns, dtype="int64"
)

location_samples = []
total_rows = 0

# ------------------------------------------------
# 3. Read and process dataset in chunks
# ------------------------------------------------

print("\nLoading dataset in smaller portions...")

try:

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            FILE_NAME,
            usecols=required_columns,
            chunksize=CHUNK_SIZE,
            low_memory=False,
            encoding="latin1"
        ),
        start=1
    ):

        total_rows += len(chunk)

        # Convert date and time
        chunk["Start_Time"] = pd.to_datetime(
            chunk["Start_Time"],
            errors="coerce"
        )

        # Severity analysis
        severity_counts = severity_counts.add(
            chunk["Severity"].value_counts(),
            fill_value=0
        )

        # Time analysis
        chunk["Hour"] = chunk["Start_Time"].dt.hour
        chunk["Weekday"] = chunk["Start_Time"].dt.day_name()

        hour_counts = hour_counts.add(
            chunk["Hour"].value_counts(),
            fill_value=0
        )

        weekday_counts = weekday_counts.add(
            chunk["Weekday"].value_counts(),
            fill_value=0
        )

        # State analysis
        state_counts = state_counts.add(
            chunk["State"].dropna().value_counts(),
            fill_value=0
        )

        # Weather analysis
        weather_counts = weather_counts.add(
            chunk["Weather_Condition"].dropna().value_counts(),
            fill_value=0
        )

        # Road-related infrastructure analysis
        for column in road_columns:

            road_counts[column] += (
                chunk[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("true")
                .sum()
            )

        # Geographic coordinates
        locations = chunk[
            ["Start_Lat", "Start_Lng"]
        ].copy()

        locations["Start_Lat"] = pd.to_numeric(
            locations["Start_Lat"],
            errors="coerce"
        )

        locations["Start_Lng"] = pd.to_numeric(
            locations["Start_Lng"],
            errors="coerce"
        )

        locations = locations.dropna()

        # Keep coordinates within the USA
        locations = locations[
            locations["Start_Lat"].between(24, 50)
            & locations["Start_Lng"].between(-125, -66)
        ]

        if len(locations) > 0:

            location_samples.append(
                locations.sample(
                    n=min(500, len(locations)),
                    random_state=chunk_number
                )
            )

        print(
            f"Processed chunk {chunk_number} | "
            f"Total records: {total_rows:,}"
        )

except FileNotFoundError:

    print("\nERROR: Dataset file not found!")
    print("Place the CSV file inside the project folder.")
    raise

except ValueError as error:

    print("\nERROR: Required columns could not be found.")
    print("Check whether you downloaded the correct dataset.")
    print("Details:", error)
    raise

# ------------------------------------------------
# 4. Prepare summary results
# ------------------------------------------------

severity_counts = severity_counts.sort_index().astype(int)
hour_counts = hour_counts.sort_index().astype(int)
weekday_counts = weekday_counts.astype(int)
state_counts = state_counts.astype(int)
weather_counts = weather_counts.astype(int)

road_counts = road_counts.sort_values(
    ascending=False
)

weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekday_counts = weekday_counts.reindex(
    weekday_order,
    fill_value=0
)

print("\n========== DATASET SUMMARY ==========")

print("Total accident records:", f"{total_rows:,}")

print("\nAccidents by Severity:")
print(severity_counts)

print("\nAccidents by Hour:")
print(hour_counts)

print("\nAccidents by Weekday:")
print(weekday_counts)

print("\nTop 10 States:")
print(state_counts.sort_values(
    ascending=False
).head(10))

print("\nTop 10 Weather Conditions:")
print(weather_counts.sort_values(
    ascending=False
).head(10))

print("\nRoad-Related Factors:")
print(road_counts)

# ------------------------------------------------
# 5. Save summary CSV files
# ------------------------------------------------

severity_counts.to_csv(
    "outputs/severity_summary.csv"
)

hour_counts.to_csv(
    "outputs/hourly_summary.csv"
)

weekday_counts.to_csv(
    "outputs/weekday_summary.csv"
)

state_counts.to_csv(
    "outputs/state_summary.csv"
)

weather_counts.to_csv(
    "outputs/weather_summary.csv"
)

road_counts.to_csv(
    "outputs/road_factors_summary.csv"
)

# ------------------------------------------------
# 6. Accident severity visualization
# ------------------------------------------------

if not severity_counts.empty:

    plt.figure(figsize=(8, 5))

    sns.barplot(
        x=severity_counts.index.astype(str),
        y=severity_counts.values
    )

    plt.title("Traffic Accidents by Severity")
    plt.xlabel("Severity Level")
    plt.ylabel("Number of Accidents")
    plt.tight_layout()

    plt.savefig(
        "outputs/accident_severity.png",
        dpi=300
    )

    plt.close()

# ------------------------------------------------
# 7. Accidents by hour
# ------------------------------------------------

if not hour_counts.empty:

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x=hour_counts.index,
        y=hour_counts.values
    )

    plt.title("Traffic Accidents by Hour of Day")
    plt.xlabel("Hour (0-23)")
    plt.ylabel("Number of Accidents")
    plt.tight_layout()

    plt.savefig(
        "outputs/accidents_by_hour.png",
        dpi=300
    )

    plt.close()

# ------------------------------------------------
# 8. Accidents by weekday
# ------------------------------------------------

if weekday_counts.sum() > 0:

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x=weekday_counts.index,
        y=weekday_counts.values
    )

    plt.title("Traffic Accidents by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        "outputs/accidents_by_weekday.png",
        dpi=300
    )

    plt.close()

# ------------------------------------------------
# 9. Top 10 weather conditions
# ------------------------------------------------

top_weather = weather_counts.sort_values(
    ascending=False
).head(10)

if not top_weather.empty:

    plt.figure(figsize=(11, 6))

    sns.barplot(
        x=top_weather.values,
        y=top_weather.index
    )

    plt.title("Top 10 Weather Conditions During Accidents")
    plt.xlabel("Number of Accidents")
    plt.ylabel("Weather Condition")
    plt.tight_layout()

    plt.savefig(
        "outputs/weather_conditions.png",
        dpi=300
    )

    plt.close()

# ------------------------------------------------
# 10. Top 10 states
# ------------------------------------------------

top_states = state_counts.sort_values(
    ascending=False
).head(10)

if not top_states.empty:

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=top_states.values,
        y=top_states.index
    )

    plt.title("Top 10 States with Most Traffic Accidents")
    plt.xlabel("Number of Accidents")
    plt.ylabel("State")
    plt.tight_layout()

    plt.savefig(
        "outputs/top_accident_states.png",
        dpi=300
    )

    plt.close()

# ------------------------------------------------
# 11. Road-related factors
# ------------------------------------------------

if road_counts.sum() > 0:

    plt.figure(figsize=(11, 6))

    sns.barplot(
        x=road_counts.values,
        y=road_counts.index
    )

    plt.title("Accidents Near Road Infrastructure Features")
    plt.xlabel("Number of Accidents")
    plt.ylabel("Road-Related Feature")
    plt.tight_layout()

    plt.savefig(
        "outputs/road_factors.png",
        dpi=300
    )

    plt.close()

# ------------------------------------------------
# 12. Geographic accident hotspot map
# ------------------------------------------------

print("\nCreating accident hotspot map...")

if location_samples:

    locations_df = pd.concat(
        location_samples,
        ignore_index=True
    )

    if len(locations_df) > 5000:

        locations_df = locations_df.sample(
            n=5000,
            random_state=42
        )

    accident_map = folium.Map(
        location=[39.0, -98.0],
        zoom_start=4,
        tiles="CartoDB positron"
    )

    heat_data = locations_df[
        ["Start_Lat", "Start_Lng"]
    ].values.tolist()

    HeatMap(
        heat_data,
        radius=8,
        blur=12
    ).add_to(accident_map)

    accident_map.save(
        "outputs/accident_hotspots.html"
    )

    print("Hotspot map saved successfully!")

else:

    print("No valid accident coordinates found.")

# ------------------------------------------------
# 13. Final results
# ------------------------------------------------

print("\n==========================================")
print("       TASK 05 ANALYSIS COMPLETED")
print("==========================================")

print("Total accident records:", f"{total_rows:,}")
print("Charts and summary files saved in outputs/")
print("Interactive hotspot map: outputs/accident_hotspots.html")

print("==========================================")
