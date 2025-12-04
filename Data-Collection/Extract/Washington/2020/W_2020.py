import pandas as pd
import os
import numpy as np

#Define Core Tables and Lookup Tables ---
CORE_TABLES = {
    'agencies': 'agencies.csv', 'incident': 'NIBRS_incident.csv', 'offense': 'NIBRS_OFFENSE.csv',
    'victim': 'NIBRS_VICTIM.csv', 'offender': 'NIBRS_OFFENDER.csv', 'victim_offense_link': 'NIBRS_VICTIM_OFFENSE.csv',
    'victim_offender_rel': 'NIBRS_VICTIM_OFFENDER_REL.csv', 'weapon_link': 'NIBRS_WEAPON.csv',
}

LOOKUP_TABLES = {
    'offense_type': 'NIBRS_OFFENSE_TYPE.csv', 'location_type': 'NIBRS_LOCATION_TYPE.csv', 
    'age': 'NIBRS_AGE.csv', 'race': 'REF_RACE.csv', 'relationship': 'NIBRS_RELATIONSHIP.csv', 
    'weapon_type': 'NIBRS_WEAPON_TYPE.csv', 
    'victim_type': 'NIBRS_VICTIM_TYPE.csv', 'assignment_type': 'NIBRS_ASSIGNMENT_TYPE.csv', 
}

OUTPUT_FILENAME = 'W_2020.csv' 

# --- Key Column Definitions for Data Type Coercion ---
JOIN_KEYS = ['incident_id', 'offense_id', 'victim_id', 'offender_id', 'agency_id', 
             'offense_code', 'location_id', 'weapon_id', 'age_id', 'race_id', 'offense_type_id'] 

def coerce_keys(df, keys):
    """Converts specified columns in a DataFrame to string type and cleans up IDs."""
    for key in keys:
        if key in df.columns:
            # Handle potential float/mixed types and remove '.0' suffix
            df[key] = df[key].astype(str).apply(lambda x: x.replace('.0', ''))
    return df

def load_and_select_data(file_map):
    """Loads CSV files, ensures robust loading, and returns a dictionary of DataFrames."""
    data = {}
    print("Loading NIBRS files...")
    COLUMNS_TO_DROP_GLOBALLY = ['data_year'] 
    
    for key, filename in file_map.items():
        try:
            if not os.path.exists(filename):
                 print(f"ERROR: File not found: {filename}. Skipping.")
                 continue

            df = pd.read_csv(filename, low_memory=False, encoding='latin-1')
            
            df = df.drop(columns=COLUMNS_TO_DROP_GLOBALLY, errors='ignore')
            df.columns = [c.lower() for c in df.columns] # Ensure all columns are lowercased
            df = coerce_keys(df, JOIN_KEYS)
            
            data[key] = df
            print(f"  Loaded {filename}")
        except Exception as e:
            print(f" ERROR loading {filename}: {e}. Skipping.")
    return data

def merge_comprehensive_summary():
    
    core_data = load_and_select_data(CORE_TABLES)
    lookup_data = load_and_select_data(LOOKUP_TABLES)

    if not core_data or not lookup_data:
        print("\nMerge failed due to missing or unreadable core files.")
        return

    print("-" * 50)
    print("Starting Relational Joins for Final 20-Column Summary...")

    # Base Incident and Agency Details ---
    essential_incident_cols = ['incident_id', 'agency_id', 'incident_date', 'incident_hour'] 
    base_df = core_data['incident'][essential_incident_cols].copy()
    
    base_df = pd.merge(base_df, core_data['agencies'][['agency_id', 'ucr_agency_name', 'county_name', 'population']], on='agency_id', how='left')

    # Offense Details ---
    main_df = pd.merge(base_df, core_data['offense'], on='incident_id', how='left', suffixes=('_inc', '_off'))
    
    #Determine the correct key for Offense Type join
    join_key = None
    offense_type_cols = ['offense_name', 'offense_category_name']
    
    if 'offense_type_id' in main_df.columns and 'offense_type_id' in lookup_data['offense_type'].columns:
        join_key = 'offense_type_id'
        offense_type_cols.append('offense_type_id')
    elif 'offense_code' in main_df.columns and 'offense_code' in lookup_data['offense_type'].columns:
        join_key = 'offense_code'
        offense_type_cols.append('offense_code')

    if join_key:
        print(f"  Joining Offense Type data on '{join_key}'...")
        main_df = pd.merge(main_df, lookup_data['offense_type'][offense_type_cols], 
                           on=join_key, how='left')
    else:
        print(" Could not find a suitable key (offense_type_id or offense_code) to join Offense Type. Skipping.")
    
    # Location Type join
    if 'location_type' in lookup_data and 'location_id' in main_df.columns:
        main_df = pd.merge(main_df, lookup_data['location_type'][['location_id', 'location_name']], on='location_id', how='left')
    
    #Victim Details (Includes Victim Id, Age Group, Race) ---
    victim_df_renamed = core_data['victim'].rename(columns={'age_id': 'age_id_victim', 'race_id': 'race_id_victim', 'sex_code': 'sex_code_victim_raw', 'resident_status_code': 'resident_status_code_victim', 'assignment_type_id': 'assignment_type_id_victim', 'age_num': 'age_num_victim'}) if 'victim' in core_data else pd.DataFrame()
    
    victim_link_cols = ['victim_id', 'incident_id', 'victim_type_id', 'age_id_victim', 
                        'race_id_victim', 'sex_code_victim_raw', 'resident_status_code_victim', 'age_num_victim']
    
    victim_link_df = pd.merge(core_data['victim_offense_link'], victim_df_renamed[victim_link_cols], on='victim_id', how='left')

    if 'offense_id' in main_df.columns and 'offense_id' in victim_link_df.columns:
        main_df = pd.merge(main_df, victim_link_df, on=['incident_id', 'offense_id'], how='left')
    else:
        main_df = pd.merge(main_df, victim_link_df, on='incident_id', how='left')
    
    #Add Victim descriptive lookups 
    if 'age' in lookup_data:
        main_df = pd.merge(main_df, lookup_data['age'][['age_id', 'age_name']], left_on='age_id_victim', right_on='age_id', how='left')
        main_df = main_df.rename(columns={'age_name': 'age_name_victim'}).drop(columns=['age_id'], errors='ignore')

    #Replace 'AGE IN YEARS' with "age_num years old"
    if 'age_num_victim' in main_df.columns:
        main_df['age_num_victim_str'] = pd.to_numeric(main_df['age_num_victim'], errors='coerce').astype('Int64').astype(str)
        main_df['age_name_victim'] = np.where(
            (main_df['age_name_victim'].astype(str).str.lower() == 'age in years') & (main_df['age_num_victim_str'] != '<NA>'),
            main_df['age_num_victim_str'] + ' years old',
            main_df['age_name_victim']
        )
    
    if 'race' in lookup_data:
        main_df = pd.merge(main_df, lookup_data['race'][['race_id', 'race_desc']], left_on='race_id_victim', right_on='race_id', how='left')
        main_df = main_df.rename(columns={'race_desc': 'race_desc_victim'}).drop(columns=['race_id'], errors='ignore')
    
    if 'victim_type' in lookup_data:
        main_df = pd.merge(main_df, lookup_data['victim_type'][['victim_type_id', 'victim_type_name']], on='victim_type_id', how='left')
    
    #Offender Details and Relationship (Includes Offender Age, Race) ---
    offender_df_base = core_data['offender'].copy() if 'offender' in core_data else pd.DataFrame()

    if not offender_df_base.empty:
        offender_desc_df = offender_df_base[['offender_id', 'sex_code', 'age_id', 'race_id', 'age_num']].copy().rename(columns={
            'age_id': 'age_id_offender_raw',
            'race_id': 'race_id_offender_raw',
            'sex_code': 'sex_code_offender',
            'age_num': 'age_num_offender_raw'
        })
    else:
        offender_desc_df = pd.DataFrame()

    rel_df = pd.merge(core_data['victim_offender_rel'], offender_desc_df, on='offender_id', how='left') if 'victim_offender_rel' in core_data else offender_desc_df.copy()
    if 'relationship' in lookup_data and 'relationship_id' in rel_df.columns:
        rel_df = pd.merge(rel_df, lookup_data['relationship'][['relationship_id', 'relationship_name']], on='relationship_id', how='left')

    # Add Offender Age descriptive names
    if 'age' in lookup_data:
        rel_df = pd.merge(rel_df, lookup_data['age'][['age_id', 'age_name']], left_on='age_id_offender_raw', right_on='age_id', how='left')
        rel_df = rel_df.rename(columns={'age_name': 'age_name_offender'}).drop(columns=['age_id'], errors='ignore') 

    #Replace 'AGE IN YEARS' with "age_num years old" for offender
    if 'age_num_offender_raw' in rel_df.columns:
        rel_df['age_num_offender_str'] = pd.to_numeric(rel_df['age_num_offender_raw'], errors='coerce').astype('Int64').astype(str)
        rel_df['age_name_offender'] = np.where(
            (rel_df['age_name_offender'].astype(str).str.lower() == 'age in years') & (rel_df['age_num_offender_str'] != '<NA>'),
            rel_df['age_num_offender_str'] + ' years old',
            rel_df['age_name_offender']
        )

    if 'race' in lookup_data:
        rel_df = pd.merge(rel_df, lookup_data['race'][['race_id', 'race_desc']], left_on='race_id_offender_raw', right_on='race_id', how='left')
        rel_df = rel_df.rename(columns={'race_desc': 'race_desc_offender'}).drop(columns=['race_id'], errors='ignore') 

    rel_df_dedup = rel_df.drop_duplicates(subset='victim_id', keep='first') if 'victim_id' in rel_df.columns else rel_df

    # Merge descriptive columns back into main_df
    if 'victim_id' in main_df.columns:
        main_df = pd.merge(main_df, rel_df_dedup[['victim_id', 'relationship_name', 'sex_code_offender', 
                                                  'age_name_offender', 'race_desc_offender']], 
                           on='victim_id', how='left', suffixes=('_victim', '_offender'))

    #Weapon Details ---
    if 'weapon_link' in core_data and 'weapon_type' in lookup_data:
        weapon_df = pd.merge(core_data['weapon_link'], lookup_data['weapon_type'], on='weapon_id', how='left')
        weapon_df_dedup = weapon_df.drop_duplicates(subset='offense_id', keep='first')
        main_df = pd.merge(main_df, weapon_df_dedup[['offense_id', 'weapon_name']], on='offense_id', how='left')

    #Convert incident_date to ISO yyyy-mm-dd (robust parsing) ---
    if 'incident_date' in main_df.columns:
        # normalize strings and treat empty-like as NA
        dates = main_df['incident_date'].astype(str).str.strip().replace({'': pd.NA, 'nan': pd.NA})
        # initial fast parse (no deprecated arg)
        parsed = pd.to_datetime(dates, errors='coerce', dayfirst=True)
        # explicit formats to try for remaining unparsed values
        formats = ['%d-%b-%y', '%d-%b-%Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']
        mask = parsed.isna() & dates.notna()
        for fmt in formats:
            if not mask.any():
                break
            parsed_try = pd.to_datetime(dates[mask], format=fmt, errors='coerce')
            parsed.loc[mask] = parsed_try
            mask = parsed.isna() & dates.notna()
        main_df['incident_date'] = parsed.dt.strftime('%Y-%m-%d').fillna('')

    #Select Final 20 Columns (Mapping to Final Output Names) ---
    final_cols_map = {
        'offense_id': 'Offense ID',
        'victim_id': 'Victim Id',
        'ucr_agency_name': 'Agency Name',
        'county_name': 'County',
        'population': 'Population',
        'location_name': 'Location Name',
        'incident_date': 'Incident Date',
        'incident_hour': 'Incident Hour',
        'offense_name': 'Offense Name',
        'offense_category_name': 'Offense Category', 
        'weapon_name': 'Weapon Name',
        'victim_type_name': 'Victim Type',
        'sex_code_victim_raw': 'Victim Sex',
        'age_name_victim': 'Victim Age Group',
        'race_desc_victim': 'Victim Race',
        'resident_status_code_victim': 'Victim Resident Status', 
        'sex_code_offender': 'Offender Sex',
        'age_name_offender': 'Offender Age', 
        'race_desc_offender': 'Offender Race', 
        'relationship_name': 'Victim-Offender Relationship',
    }
    
    # Filter and Rename
    cols_to_keep = [col for col in final_cols_map.keys() if col in main_df.columns]
    summary_df = main_df.filter(cols_to_keep)
    summary_df = summary_df.rename(columns=final_cols_map)

    # Reorder columns to match the requested 20-column sequence
    requested_order = [
        'Offense ID', 'Agency Name', 'County', 'Population', 'Location Name', 
        'Incident Date', 'Incident Hour', 'Offense Name', 'Offense Category', 
        'Weapon Name',
        'Victim Type', 'Victim Sex', 'Victim Age Group', 'Victim Race', 'Victim Resident Status', 
        'Offender Sex', 'Victim-Offender Relationship', 'Offender Age', 
        'Victim Id', 'Offender Race'
    ]
    
    # Drop helper columns used for the age fix
    helper_cols_to_drop = [col for col in summary_df.columns if 'age_num' in str(col)]
    summary_df = summary_df.drop(columns=helper_cols_to_drop, errors='ignore')

    final_columns = [col for col in requested_order if col in summary_df.columns]
    summary_df = summary_df[final_columns]
    
    print("-" * 50)
    print("Saving the final selected summary...")
    print(f"Final table has {len(summary_df):,} rows and {len(summary_df.columns)} columns.")
    summary_df.to_csv(OUTPUT_FILENAME, index=False)
    print(f"Success! Summary table saved as: *{OUTPUT_FILENAME}*")

# Execute the merge function
if __name__ == "__main__":
    merge_comprehensive_summary()