#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import math

#Load Dataset
def load_dataset(path):
    # Read CSV
    df = pd.read_csv(path)
    
    #Ensure Incident Date is datetime
    df['Incident Date'] = pd.to_datetime(df['Incident Date'])
    return df

#Preprocessing 
def preprocess(df):
    df['year'] = df['Incident Date'].dt.year
    df['month'] = df['Incident Date'].dt.month
    
    rename_map = {}
    if 'State' in df.columns: rename_map['State'] = 'state'
    if 'County' in df.columns: rename_map['County'] = 'county'
    if 'Population' in df.columns: rename_map['Population'] = 'population'
    
    df = df.rename(columns=rename_map)
    
    df = df[['state', 'county', 'population', 'year', 'month']]
    
    monthly = df.groupby(['state', 'county', 'year', 'month']).agg(
        crime_count=('state', 'count'), # count any column
        population=('population', 'first')
    ).reset_index()
    
    monthly['population'] = monthly['population'].fillna(0)
    
    return monthly

#Feature Engineering :
def feature_engineer(monthly):
    #Global 90th percentile threshold
    threshold = monthly['crime_count'].quantile(0.90)
    print("Global 90th percentile crime_count:", threshold)
    
    monthly['is_hotspot'] = (monthly['crime_count'] >= threshold).astype(int)
    
    monthly['crime_rate_per_1000'] = np.where(
        monthly['population'] > 0,
        monthly['crime_count'] / (monthly['population'] / 1000.0),
        0
    )
    
    #seasonal sin/cos
    PI = math.pi
    monthly['sin_month'] = np.sin(2 * PI * monthly['month'] / 12)
    monthly['cos_month'] = np.cos(2 * PI * monthly['month'] / 12)
    
    monthly = monthly.sort_values(['state', 'county', 'year', 'month'])
    monthly['crime_last_month'] = monthly.groupby(['state', 'county'])['crime_count'].shift(1).fillna(0)
    
    #Label Encoding
    le_state = LabelEncoder()
    monthly['state_idx'] = le_state.fit_transform(monthly['state'].astype(str))
    
    le_county = LabelEncoder()
    monthly['county_idx'] = le_county.fit_transform(monthly['county'].astype(str))
    
    return monthly

#Time-aware Split
def split_data(data):
    max_year = data['year'].max()
    test_year = int(max_year)
    
    train = data[data['year'] < test_year].copy()
    test = data[data['year'] == test_year].copy()
    
    print(f"Train size = {len(train)}, Test size = {len(test)}")
    return train, test

#RandomForest Training
def train_rf(train, test):
    features = [
        "state_idx",
        "county_idx",
        "month",
        "sin_month",
        "cos_month",
        "crime_last_month",
        "crime_rate_per_1000",
    ]
    
    X_train = train[features]
    y_train = train['is_hotspot']
    
    X_test = test[features]
    y_test = test['is_hotspot']
    
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        random_state=42
    )
    
    rf.fit(X_train, y_train)
    
    #Predict
    probs = rf.predict_proba(X_test)[:, 1]
    preds = rf.predict(X_test)
    
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
    else:
        auc = 0.0
        f1 = 0.0
        
    acc = accuracy_score(y_test, preds)
    
    print("\n=== Sklearn RF Results ===")
    print("AUC =", auc)
    print("F1  =", f1)
    print("ACC =", acc)
    
    test['hotspot_probability'] = probs
    test['prediction'] = preds
    
    return rf, test

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to final CSV dataset"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save predictions CSV"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Loading dataset...")
    df = load_dataset(args.input_path)
    
    print("Aggregating monthly data...")
    monthly = preprocess(df)
    
    print("Feature engineering...")
    monthly = feature_engineer(monthly)
    
    print("Splitting train/test (time-aware)...")
    train, test = split_data(monthly)
    
    print("Training RandomForest...")
    rf_model, predictions = train_rf(train, test)
    
    print("Saving predictions...")
    
    final_preds = predictions[[
        "state", "county", "year", "month", "crime_count", "is_hotspot", "prediction", "hotspot_probability"
    ]]
    
    final_preds.to_csv(args.output_path, index=False)
    print(f"Predictions saved to {args.output_path}")
    
    print("\nTraining complete.")

if __name__ == "__main__":
    main()