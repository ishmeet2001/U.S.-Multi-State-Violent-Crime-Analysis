import pandas as pd
from pathlib import Path
import os

def process_data():
    print("Loading data...")
    #Read the large CSV
    df = pd.read_csv("final_combined_2020_2024_std10.csv", low_memory=False)
    
    print("Data loaded. Processing...")
    
    output_dir = Path("EDA/dashboard_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    #Ensure Date is datetime
    df['Incident Date'] = pd.to_datetime(df['Incident Date'], errors='coerce')
    df['Year'] = df['Incident Date'].dt.year
    df['Month'] = df['Incident Date'].dt.month
    
    #1. yearly_trends.csv
    print("Generating yearly_trends.csv...")
    yearly = df.groupby('Year').size().reset_index(name='count')
    yearly.to_csv(output_dir / "yearly_trends.csv", index=False)
    
    #2. monthly_trends.csv
    print("Generating monthly_trends.csv...")
    monthly = df.groupby(['Year', 'Month']).size().reset_index(name='count')
    monthly.to_csv(output_dir / "monthly_trends.csv", index=False)
    
    #3. hourly_distribution.csv
    print("Generating hourly_distribution.csv...")
    hourly = df.groupby('Incident Hour').size().reset_index(name='count')
    hourly.to_csv(output_dir / "hourly_distribution.csv", index=False)
    
    #4. offense_categories.csv
    print("Generating offense_categories.csv...")
    cats = df.groupby('Offense Category').size().reset_index(name='count')
    cats.to_csv(output_dir / "offense_categories.csv", index=False)
    
    #5. top_offense_names.csv
    print("Generating top_offense_names.csv...")
    offenses = df.groupby('Offense Name').size().reset_index(name='count')
    offenses.to_csv(output_dir / "top_offense_names.csv", index=False)
    
    #6. top_locations.csv
    print("Generating top_locations.csv...")
    locs = df.groupby('Location Name').size().reset_index(name='count')
    locs.to_csv(output_dir / "top_locations.csv", index=False)
    
    #7. victim_age_distribution.csv
    print("Generating victim_age_distribution.csv...")

    age_col = 'Victim Age Group' if 'Victim Age Group' in df.columns else 'Victim Age'
    if age_col in df.columns:
        age = df.groupby(age_col).size().reset_index(name='count')
        age.to_csv(output_dir / "victim_age_distribution.csv", index=False)
    
    #8. victim_race_by_offense.csv
    print("Generating victim_race_by_offense.csv...")
    race = df.groupby(['Offense Category', 'Victim Race']).size().reset_index(name='count')
    race.to_csv(output_dir / "victim_race_by_offense.csv", index=False)
    
    #9. victim_sex_by_offense.csv
    print("Generating victim_sex_by_offense.csv...")
    sex = df.groupby(['Offense Category', 'Victim Sex']).size().reset_index(name='count')
    sex.to_csv(output_dir / "victim_sex_by_offense.csv", index=False)
    
    #10. weapon_usage.csv
    print("Generating weapon_usage.csv...")
    weapon = df.groupby('Weapon Name').size().reset_index(name='count')
    weapon.to_csv(output_dir / "weapon_usage.csv", index=False)
    
    #11. state_crime_rates.csv
    print("Generating state_crime_rates.csv...")
    #Approximate population by summing max population of each agency
    if 'Population' in df.columns and 'Agency Name' in df.columns:
        agency_pop = df.groupby(['State', 'Agency Name'])['Population'].max().reset_index()
        state_pop = agency_pop.groupby('State')['Population'].sum().reset_index()
        state_counts = df.groupby('State').size().reset_index(name='count')
        state_rates = pd.merge(state_counts, state_pop, on='State')
        state_rates['CrimeRatePer100k'] = (state_rates['count'] / state_rates['Population']) * 100000
        state_rates.to_csv(output_dir / "state_crime_rates.csv", index=False)
    else:
        state_counts = df.groupby('State').size().reset_index(name='count')
        state_counts['CrimeRatePer100k'] = state_counts['count'] # Dummy
        state_counts.to_csv(output_dir / "state_crime_rates.csv", index=False)

    #day_of_week_patterns.csv
    print("Generating day_of_week_patterns.csv...")
    df['DayOfWeek'] = df['Incident Date'].dt.day_name()
    dow = df.groupby('DayOfWeek').size().reset_index(name='count')
    
    #Sort by day order
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow['DayOfWeek'] = pd.Categorical(dow['DayOfWeek'], categories=days, ordered=True)
    dow = dow.sort_values('DayOfWeek')
    dow.to_csv(output_dir / "day_of_week_patterns.csv", index=False)

    #victim_sex_by_offense_crosstab.csv
    print("Generating victim_sex_by_offense_crosstab.csv...")
    crosstab = pd.crosstab(df['Offense Category'], df['Victim Sex']).reset_index()
    crosstab.to_csv(output_dir / "victim_sex_by_offense_crosstab.csv", index=False)

    #12. county_heatmap.csv 
    print("Generating county_heatmap.csv...")
    # Group by State, County, AND Offense Category for filtering
    county_counts = df.groupby(['State', 'County', 'Offense Category']).size().reset_index(name='count')
    
    #Map FIPS codes for better plotting
    try:
        print("Downloading FIPS mapping...")
        fips_url = "https://raw.githubusercontent.com/kjhealy/fips-codes/master/county_fips_master.csv"
        fips_df = pd.read_csv(fips_url, encoding='ISO-8859-1')
        # fips_df has 'state_name', 'county_name', 'fips'
        # Normalize names for merging
        fips_df['state_name'] = fips_df['state_name'].str.upper()
        fips_df['county_name'] = fips_df['county_name'].str.upper().str.replace(' COUNTY', '').str.replace(' PARISH', '')
        
        state_map = {
            'NY': 'NEW YORK',
            'WA': 'WASHINGTON',
            'NM': 'NEW MEXICO',
            'TX': 'TEXAS',
            'CO': 'COLORADO',
            'Colorado': 'COLORADO',
            'New Mexico': 'NEW MEXICO',
            'Texas': 'TEXAS',
            'Washington': 'WASHINGTON'
        }
        
        county_counts['State_Upper'] = county_counts['State'].map(state_map).fillna(county_counts['State'].str.upper())
        county_counts['County_Upper'] = county_counts['County'].str.upper().str.replace(' COUNTY', '').str.replace(' PARISH', '')
        
        merged = pd.merge(
            county_counts, 
            fips_df[['state_name', 'county_name', 'fips']], 
            left_on=['State_Upper', 'County_Upper'], 
            right_on=['state_name', 'county_name'], 
            how='left'
        )

        merged['fips'] = merged['fips'].fillna(0).astype(int).astype(str).str.zfill(5)
        merged = merged[merged['fips'] != '00000'] # Remove unmapped
        
        merged.to_csv(output_dir / "county_heatmap.csv", index=False)
    except Exception as e:
        print(f"Failed to generate FIPS mapping: {e}")
        county_counts.to_csv(output_dir / "county_heatmap.csv", index=False)
    print("Done!")
    
if __name__ == "__main__":
    process_data()