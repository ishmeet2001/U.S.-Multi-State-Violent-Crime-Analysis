#   U.S. Multi-State Violent Crime Analysis

Dashboard -> https://huggingface.co/spaces/rit0027/crime-analytics

This project, completed for CMPT 732: Big Data Lab 1, focuses on building a unified analytics pipeline for violent-crime data reported through the FBI’s NIBRS system. Since each state structures and reports NIBRS data differently, the project harmonizes five states - New York, Texas, Washington, Colorado, and New Mexico into a single, consistent dataset covering 2020–2024.

The pipeline supports large-scale integration, multi-year trend analysis, demographic insights, geographic hotspot exploration, and predictive modeling. An interactive Streamlit dashboard provides an accessible front end for exploring offenders, victims, offense types, and regional crime patterns.


## System Architecture 
![Screenshot 2025-12-06 at 2 09 22 AM](https://media.github.sfu.ca/user/5139/files/4f4b7b77-c84d-49a1-b992-3bdb0e677333)

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
## Understanding the Repository Layout

### 1. Extract 
Contains Python scripts for each state and year, used to extract relevant fields from the raw NIBRS CSV files. These scripts standardize column names and output a consistent schema for all state-year datasets.

### 2. Transform & Load
**Merging scripts:** Used to merge datasets state-wise and then combine all states into a single unified multi-year dataset.  
**Cleaning scripts:** Used to clean, standardize, and transform the merged datasets. Multiple cleaning files exist because the work was divided among team members.

### 3. Data-Analysis
Contains EDA scripts and notebooks. Each team member worked on different parts of the analysis, including temporal trends, demographic patterns, geographic distributions, and offense-type breakdowns.
