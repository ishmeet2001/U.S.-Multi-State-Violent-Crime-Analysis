import pandas as pd
import os

# Define Core Tables and Lookup Tables 
CORE_TABLES = {
    'agencies': 'agencies.csv', 'incident': 'NIBRS_incident.csv', 'offense': 'NIBRS_OFFENSE.csv',
    'victim': 'NIBRS_VICTIM.csv', 'offender': 'NIBRS_OFFENDER.csv', 'victim_offense_link': 'NIBRS_VICTIM_OFFENSE.csv',
    'victim_offender_rel': 'NIBRS_VICTIM_OFFENDER_REL.csv', 'weapon_link': 'NIBRS_WEAPON.csv',
    'bias_motivation': 'NIBRS_BIAS_MOTIVATION.csv', 
}

LOOKUP_TABLES = {
    'offense_type': 'NIBRS_OFFENSE_TYPE.csv', 'location_type': 'NIBRS_LOCATION_TYPE.csv', 
    'age': 'NIBRS_AGE.csv', 'race': 'REF_RACE.csv', 'relationship': 'NIBRS_RELATIONSHIP.csv', 
    'weapon_type': 'NIBRS_WEAPON_TYPE.csv', 
    'victim_type': 'NIBRS_VICTIM_TYPE.csv', 'assignment_type': 'NIBRS_ASSIGNMENT_TYPE.csv', 
}

OUTPUT_FILENAME = 'new_mexico_2021.csv' 

JOIN_KEYS = ['incident_id', 'offense_id', 'victim_id', 'offender_id', 'agency_id', 
             'offense_code', 'location_id', 'weapon_id', 'age_id', 'race_id', 'offense_type_id'] 

def coerce_keys(df, keys):
    """Converts specified columns in a DataFrame to string type and cleans up IDs."""
    for key in keys:
        if key in df.columns:
            
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
                 print(f"  ERROR: File not found: {filename}. Skipping.")
                 continue

            df = pd.read_csv(filename, low_memory=False, encoding='latin-1')
            
            df = df.drop(columns=COLUMNS_TO_DROP_GLOBALLY, errors='ignore')
            df.columns = [c.lower() for c in df.columns] 
            df = coerce_keys(df, JOIN_KEYS)
            
            data[key] = df
            print(f"  Loaded {filename}")
        except Exception as e:
            print(f"  CRITICAL ERROR loading {filename}: {e}. Skipping.")
    return data

def merge_comprehensive_summary():
    
    core_data = load_and_select_data(CORE_TABLES)
    lookup_data = load_and_select_data(LOOKUP_TABLES)

    if not core_data or not lookup_data:
        print("\nMerge failed due to missing or unreadable core files.")
        return

    # Base Incident and Agency Details
    essential_incident_cols = ['incident_id', 'agency_id', 'incident_date', 'incident_hour'] 
    base_df = core_data['incident'][essential_incident_cols].copy()
    
    base_df = pd.merge(base_df, core_data['agencies'][['agency_id', 'ucr_agency_name', 'county_name', 'population']], on='agency_id', how='left')

    # Offense Details 
    main_df = pd.merge(base_df, core_data['offense'], on='incident_id', how='left', suffixes=('_inc', '_off'))
    
    main_df = pd.merge(main_df, lookup_data['offense_type'][['offense_code', 'offense_name', 'offense_category_name']], 
                       on='offense_code', how='left')
    main_df = pd.merge(main_df, lookup_data['location_type'][['location_id', 'location_name']], on='location_id', how='left')
    

    # Victim Details 
    victim_df_renamed = core_data['victim'].rename(columns={'age_id': 'age_id_victim', 'race_id': 'race_id_victim', 'sex_code': 'sex_code_victim_raw', 'resident_status_code': 'resident_status_code_victim', 'assignment_type_id': 'assignment_type_id_victim'})
    
    victim_link_cols = ['victim_id', 'incident_id', 'victim_type_id', 'age_id_victim', 
                        'race_id_victim', 'sex_code_victim_raw', 'resident_status_code_victim']
    
    victim_link_df = pd.merge(core_data['victim_offense_link'], victim_df_renamed[victim_link_cols], on='victim_id', how='left')
    main_df = pd.merge(main_df, victim_link_df, on=['incident_id', 'offense_id'], how='left')
  
    main_df = pd.merge(main_df, lookup_data['age'][['age_id', 'age_name']], left_on='age_id_victim', right_on='age_id', how='left')
    main_df = main_df.rename(columns={'age_name': 'age_name_victim'}).drop(columns=['age_id'], errors='ignore')

    main_df = pd.merge(main_df, lookup_data['race'][['race_id', 'race_desc']], left_on='race_id_victim', right_on='race_id', how='left')
    main_df = main_df.rename(columns={'race_desc': 'race_desc_victim'}).drop(columns=['race_id'], errors='ignore')
    
    main_df = pd.merge(main_df, lookup_data['victim_type'][['victim_type_id', 'victim_type_name']], on='victim_type_id', how='left')
    
    # Offender Details and Relationship
    offender_df_base = core_data['offender'].copy()

    offender_desc_df = offender_df_base[['offender_id', 'sex_code', 'age_id', 'race_id']].copy().rename(columns={
        'age_id': 'age_id_offender_raw',
        'race_id': 'race_id_offender_raw',
        'sex_code': 'sex_code_offender'
    })
    
    rel_df = pd.merge(core_data['victim_offender_rel'], offender_desc_df, on='offender_id', how='left')
    rel_df = pd.merge(rel_df, lookup_data['relationship'][['relationship_id', 'relationship_name']], on='relationship_id', how='left')

    # Add Offender Age and Race descriptive names
    rel_df = pd.merge(rel_df, lookup_data['age'][['age_id', 'age_name']], left_on='age_id_offender_raw', right_on='age_id', how='left')
    rel_df = rel_df.rename(columns={'age_name': 'age_name_offender'}).drop(columns=['age_id'], errors='ignore') 

    rel_df = pd.merge(rel_df, lookup_data['race'][['race_id', 'race_desc']], left_on='race_id_offender_raw', right_on='race_id', how='left')
    rel_df = rel_df.rename(columns={'race_desc': 'race_desc_offender'}).drop(columns=['race_id'], errors='ignore') 

    rel_df_dedup = rel_df.drop_duplicates(subset='victim_id', keep='first')
    
    main_df = pd.merge(main_df, rel_df_dedup[['victim_id', 'relationship_name', 'sex_code_offender', 
                                              'age_name_offender', 'race_desc_offender']], 
                       on='victim_id', how='left', suffixes=('_victim', '_offender'))

    # Weapon Details 
    if 'weapon_link' in core_data and 'weapon_type' in lookup_data:
        weapon_df = pd.merge(core_data['weapon_link'], lookup_data['weapon_type'], on='weapon_id', how='left')
        weapon_df_dedup = weapon_df.drop_duplicates(subset='offense_id', keep='first')
        main_df = pd.merge(main_df, weapon_df_dedup[['offense_id', 'weapon_name']], on='offense_id', how='left')

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
    

    cols_to_keep = [col for col in final_cols_map.keys() if col in main_df.columns]
    summary_df = main_df.filter(cols_to_keep)
    summary_df = summary_df.rename(columns=final_cols_map)

    requested_order = [
        'Offense ID', 'Agency Name', 'County', 'Population', 'Location Name', 
        'Incident Date', 'Incident Hour', 'Offense Name', 'Offense Category', 
        'Weapon Name',
        'Victim Type', 'Victim Sex', 'Victim Age Group', 'Victim Race', 'Victim Resident Status', 
        'Offender Sex', 'Victim-Offender Relationship', 'Offender Age', 
        'Victim Id', 'Offender Race'
    ]
    
    final_columns = [col for col in requested_order if col in summary_df.columns]
    summary_df = summary_df[final_columns]
    

    print(f"Final table has {len(summary_df):,} rows and {len(summary_df.columns)} columns.")
    summary_df.to_csv(OUTPUT_FILENAME, index=False)
    print(f"\n Success! Summary table saved as: **{OUTPUT_FILENAME}**")

if __name__ == "__main__":
    merge_comprehensive_summary()