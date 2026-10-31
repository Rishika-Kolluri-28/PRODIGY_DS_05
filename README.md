# Prodigy InfoTech – Data Science Internship

## Task 05: Traffic Accident Data Analysis

### Project Overview
This project analyzes US traffic accident data to identify patterns related to accident severity, road conditions, weather conditions, geographical locations, and time of day.

### Objectives
- Analyze traffic accidents based on severity.
- Identify accident patterns by hour and weekday.
- Analyze accident distribution across US states.
- Examine weather conditions during accidents.
- Study road infrastructure factors associated with accidents.
- Visualize geographical accident hotspots using an interactive map.

### Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Folium

### Dataset
**US Accidents (2016–2023)**

Source: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

The analysis processed 7,728,394 accident records using chunk-based CSV reading.

### Key Findings
- Total accident records analyzed: 7,728,394.
- Severity Level 2 had the highest number of records: 6,156,981.
- 7 AM recorded the highest number of accidents: 525,763.
- Friday recorded the highest weekday count: 1,160,588.
- California had the highest number of accident records: 1,741,433.
- Fair weather was the most frequently recorded weather condition: 2,560,802.

### Project Structure

```text
PRODIGY_DS_TASK_05/
│
├── task5.py
├── README.md
├── US_Accidents_March23.csv
│
└── outputs/
    ├── accident_severity.png
    ├── accidents_by_hour.png
    ├── accidents_by_weekday.png
    ├── weather_conditions.png
    ├── top_accident_states.png
    ├── road_factors.png
    ├── accident_hotspots.html
    └── Summary CSV files
```

### How to Run

1. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn folium
```

2. Place the dataset in the project folder.

3. Run the Python program:

```bash
python task5.py
```

4. View charts and summary files inside the `outputs` folder.

5. Open `outputs/accident_hotspots.html` in a browser to view the interactive map.

### Conclusion
This project demonstrates how Python-based data analysis and visualization can identify patterns in large-scale traffic accident datasets. The results provide insights into accident severity, timing, weather conditions, road infrastructure, and geographical distribution.

**Internship:** Prodigy InfoTech – Data Science  
**Task:** 05 – Traffic Accident Data Analysis