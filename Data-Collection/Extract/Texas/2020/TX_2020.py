import pandas as pd
import os
import numpy as np

# ------------------------------------------------------------
# File definitions
# ------------------------------------------------------------

CORE_TABLES = {
    'agencies': 'agencies.csv',
    'incident': 'NIBRS_incident.csv',
    'offense': 'NIBRS_OFFENSE.csv',
    'victim': 'NIBRS_VICTIM.csv',
    'offender': 'NIBRS_OFFENDER.csv',
    'victim_offense_link': 'NIBRS_VICTIM_OFFENSE.csv',
    'victim_offender_rel': 'NIBRS_VICTIM_OFFENDER_REL.csv',
    'weapon_link': 'NIBRS_WEAPON.csv'
}

LOOKUP_TABLES = {
    'offense_type': 'NIBRS_OFFENSE_TYPE.csv',
    'location_type': 'NIBRS_LOCATION_TYPE.csv',
    'age': 'NIBRS_AGE.csv',
    'race': 'REF_RACE.csv',
    'relationship': 'NIBRS_RELATIONSHIP.csv',
    'weapon_type': 'NIBRS_WEAPON_TYPE.csv',
    'victim_type': 'NIBRS_VICTIM_TYPE.csv',
    'assignment_type': 'NIBRS_ASSIGNMENT_TYPE.csv'
}

OUTPUT_FILENAME = 'TX-2020.csv'

JOIN_KEYS = [
    'incident_id', 'offense_id', 'victim_id', 'offender_id', 'agency_id',
    'offense_code', 'location_id', 'weapon_id', 'age_id', 'race_id', 'offense_type_id'
]


def coerce_keys(df, keys):
    """Convert join columns to clean string format."""
    for key in keys:
        if key in df.columns:
            df[key] = df[key].astype(str).str.replace('.0', '', regex=False)
    return df


def load_and_select_data(file_map):
    """Load CSVs and standardize formatting."""
    data = {}
    cols_to_drop = ['data_year']

    for key, filename in file_map.items():
        if not os.path.exists(filename):
            print(f"File not found: {filename}. Skipping.")
            continue

        try:
            df = pd.read_csv(filename, low_memory=False, encoding='latin-1')
            df = df.drop(columns=cols_to_drop, errors='ignore')
            df.columns = [c.lower() for c in df.columns]
            df = coerce_keys(df, JOIN_KEYS)
            data[key] = df
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    return data


# ------------------------------------------------------------
# Merge pipeline
# ------------------------------------------------------------

def merge_comprehensive_summary():

    core = load_and_select_data(CORE_TABLES)
    lookup = load_and_select_data(LOOKUP_TABLES)

    if not core or not lookup:
        print("Missing required input files.")
        return

    # Incident + agency info
    base_cols = ['incident_id', 'agency_id', 'incident_date', 'incident_hour']
    base = core['incident'][base_cols].copy()
    base = base.merge(
        core['agencies'][['agency_id', 'ucr_agency_name', 'county_name', 'population']],
        on='agency_id', how='left'
    )

    # Offense details
    df = base.merge(core['offense'], on='incident_id', how='left')

    # Join offense type
    offense_type_key = None
    if 'offense_type_id' in df and 'offense_type_id' in lookup['offense_type']:
        offense_type_key = 'offense_type_id'
    elif 'offense_code' in df and 'offense_code' in lookup['offense_type']:
        offense_type_key = 'offense_code'

    if offense_type_key:
        df = df.merge(
            lookup['offense_type'][['offense_name', 'offense_category_name', offense_type_key]],
            on=offense_type_key, how='left'
        )

    # Location
    df = df.merge(
        lookup['location_type'][['location_id', 'location_name']],
        on='location_id', how='left'
    )

    # Victim details
    victim = core['victim'].rename(columns={
        'age_id': 'age_id_victim',
        'race_id': 'race_id_victim',
        'sex_code': 'sex_code_victim_raw',
        'resident_status_code': 'resident_status_code_victim',
        'assignment_type_id': 'assignment_type_id_victim',
        'age_num': 'age_num_victim'
    })

    victim_cols = [
        'victim_id', 'incident_id', 'victim_type_id', 'age_id_victim',
        'race_id_victim', 'sex_code_victim_raw',
        'resident_status_code_victim', 'age_num_victim'
    ]

    victim_link = core['victim_offense_link'].merge(
        victim[victim_cols], on='victim_id', how='left'
    )

    df = df.merge(victim_link, on=['incident_id', 'offense_id'], how='left')

    # Victim age / race descriptive labels
    df = df.merge(
        lookup['age'][['age_id', 'age_name']],
        left_on='age_id_victim', right_on='age_id', how='left'
    ).rename(columns={'age_name': 'age_name_victim'}).drop(columns=['age_id'], errors='ignore')

    # Replace "AGE IN YEARS" with numeric text
    df['age_num_victim_str'] = pd.to_numeric(
        df['age_num_victim'], errors='coerce'
    ).astype('Int64').astype(str)

    df['age_name_victim'] = np.where(
        df['age_name_victim'].str.lower() == 'age in years',
        df['age_num_victim_str'] + ' years old',
        df['age_name_victim']
    )

    df = df.merge(
        lookup['race'][['race_id', 'race_desc']],
        left_on='race_id_victim', right_on='race_id', how='left'
    ).rename(columns={'race_desc': 'race_desc_victim'}).drop(columns=['race_id'], errors='ignore')

    df = df.merge(
        lookup['victim_type'][['victim_type_id', 'victim_type_name']],
        on='victim_type_id', how='left'
    )

    # Offender + relationship
    offender_base = core['offender'][['offender_id', 'sex_code', 'age_id', 'race_id', 'age_num']].rename(columns={
        'sex_code': 'sex_code_offender',
        'age_id': 'age_id_offender_raw',
        'race_id': 'race_id_offender_raw',
        'age_num': 'age_num_offender_raw'
    })

    rel = core['victim_offender_rel'].merge(
        offender_base, on='offender_id', how='left'
    ).merge(
        lookup['relationship'][['relationship_id', 'relationship_name']],
        on='relationship_id', how='left'
    )

    rel = rel.merge(
        lookup['age'][['age_id', 'age_name']],
        left_on='age_id_offender_raw', right_on='age_id', how='left'
    ).rename(columns={'age_name': 'age_name_offender'}).drop(columns=['age_id'], errors='ignore')

    rel['age_num_offender_str'] = pd.to_numeric(
        rel['age_num_offender_raw'], errors='coerce'
    ).astype('Int64').astype(str)

    rel['age_name_offender'] = np.where(
        rel['age_name_offender'].str.lower() == 'age in years',
        rel['age_num_offender_str'] + ' years old',
        rel['age_name_offender']
    )

    rel = rel.merge(
        lookup['race'][['race_id', 'race_desc']],
        left_on='race_id_offender_raw', right_on='race_id', how='left'
    ).rename(columns={'race_desc': 'race_desc_offender'}).drop(columns=['race_id'], errors='ignore')

    rel = rel.drop_duplicates(subset='victim_id')

    df = df.merge(
        rel[['victim_id', 'relationship_name', 'sex_code_offender',
             'age_name_offender', 'race_desc_offender']],
        on='victim_id', how='left'
    )

    # Weapon
    if 'weapon_link' in core and 'weapon_type' in lookup:
        weapon = core['weapon_link'].merge(
            lookup['weapon_type'], on='weapon_id', how='left'
        )
        weapon = weapon.drop_duplicates(subset='offense_id')
        df = df.merge(
            weapon[['offense_id', 'weapon_name']],
            on='offense_id', how='left'
        )

    # Select final columns
    final_cols = {
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
        'relationship_name': 'Victim-Offender Relationship',
        'age_name_offender': 'Offender Age',
        'race_desc_offender': 'Offender Race'
    }

    cols = [c for c in final_cols if c in df.columns]
    summary = df[cols].rename(columns=final_cols)

    # Drop temporary numeric age columns
    drop_cols = [c for c in summary.columns if 'age_num' in c]
    summary = summary.drop(columns=drop_cols, errors='ignore')

    summary.to_csv(OUTPUT_FILENAME, index=False)
    print(f"Saved: {OUTPUT_FILENAME}")


if __name__ == "__main__":
    merge_comprehensive_summary()
