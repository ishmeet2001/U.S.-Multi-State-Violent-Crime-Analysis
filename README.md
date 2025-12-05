# Datanauts_CMPT732


Website -> https://huggingface.co/spaces/rit0027/crime-analytics


# Multi-State Violent Crime Analytics Pipeline & Dashboard

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


### 1. Extract (Data Extraction Scripts)
Contains Python scripts for each state and year, used to extract relevant fields from the raw NIBRS CSV files. These scripts standardize column names and output a consistent schema for all state-year datasets.

### 2. Transform & Load
**Merging scripts:** Used to merge datasets state-wise and then combine all states into a single unified multi-year dataset.  
**Cleaning scripts:** Used to clean, standardize, and transform the merged datasets. Multiple cleaning files exist because the work was divided among team members.

### 3. Data-Analysis
Contains EDA scripts and notebooks. Each team member worked on different parts of the analysis, including temporal trends, demographic patterns, geographic distributions, and offense-type breakdowns.
