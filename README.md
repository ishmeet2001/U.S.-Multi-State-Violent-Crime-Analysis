# Datanauts_CMPT732


Website -> https://huggingface.co/spaces/rit0027/crime-analytics


# Multi-State Violent Crime Analytics Pipeline & Dashboard

## Overview

This project builds a complete end-to-end analytics pipeline for multi-state U.S. violent crime data (2020–2024), using state-level :contentReference[oaicite:0]{index=0} NIBRS data. The pipeline harmonizes disparate data formats from multiple states, processes and cleans the data, conducts exploratory data analysis (EDA), and builds a predictive hotspot model. The results are exposed via an interactive dashboard (built with :contentReference[oaicite:1]{index=1} + :contentReference[oaicite:2]{index=2}), enabling exploratory visualization and hotspot forecasting at the county-month level.

Key features:

- Unified schema across five states (New York, Texas, Washington, Colorado, New Mexico) over five years.  
- Robust ETL pipeline using PySpark and Python for scalable ingestion and cleaning.  
- Comprehensive EDA: offense types, victim/offender demographics, temporal and geographic trends, weapon usage, victim-offender relationships.  
- Machine-learning based hotspot prediction module (county-month level), using a lightweight :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}.  
- Interactive dashboard for exploring actual incident data and model-predicted hotspot probabilities.  

---

## Project Structure

```text
├── Data-Collection
│   ├── Dataset
│   ├── Extract
│   │   ├── Colorado
│   │   │   └── [2020-2024]
│   │   │       └── colorado_[year].py
│   │   ├── New_Mexico
│   │   │   └── [2020-2024]
│   │   │       └── new_mexico[year].py
│   │   ├── New_York
│   │   │   └── [2020-2024]
│   │   │       └── NY_[year].py
│   │   ├── Texas
│   │   │   └── [2020-2024]
│   │   │       └── TX_[year].py
│   │   └── Washington
│   │       └── [2020-2024]
│   │           └── W_[year].py
│   └── Transform & Load
│       ├── Merging_Scripts
│       │   ├── Merge_5years_5states.py
│       │   └── merge_2020_2024.py
│       └── cleaning_scripts
│           ├── ETL_A_DataProfiling.py
│           ├── ETL_A_Output.txt
│           ├── ETL_B.py
│           ├── ETL_C.py
│           ├── ETL_C_age_std.py
│           ├── ETL_C_outlierdetection.py
│           └── ETL_C_std.py
├── Data_Analysis
│   ├── A1.py
│   ├── A2.py
│   ├── DataAnalysis_B.ipynb
│   ├── DataAnalysis_C.ipynb
│   ├── crime_analysis_A1_output.txt
│   └── crime_analysis_A2_output.txt
└── Data-Visualization
    ├── EDA
    │   └── dashboard_data
    ├── app.py
    ├── PredictiveModelling.ipynb
    ├── preprocess.py
    ├── requirements.txt
    └── train_hotspot.py
    
    
```


