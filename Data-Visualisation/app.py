import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import calendar
import plotly.io as pio
import numpy as np

#PAGE CONFIG
st.set_page_config(
    page_title="US Crime Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("United States Crime Patterns Dashboard (2020–2024)")
st.caption("Exploratory insights generated from NIBRS data across TX, NY, WA, NM, and CO.")

#Consistent palette and clean template
PALETTE = ["#0D3B66", "#1B9AAA", "#F4D35E", "#EE964B", "#F95738", "#6B9080", "#118AB2"]
pio.templates.default = "simple_white"
px.defaults.color_discrete_sequence = PALETTE

#DATA LOADING
DATA_DIR = Path(__file__).parent / "EDA" / "dashboard_data"

@st.cache_data
def load_all_data(base_dir: str):
    base = Path(base_dir)
    data = {
        "yearly": pd.read_csv(base / "yearly_trends.csv"),
        "monthly": pd.read_csv(base / "monthly_trends.csv"),
        "hourly": pd.read_csv(base / "hourly_distribution.csv"),
        "categories": pd.read_csv(base / "offense_categories.csv"),
        "offenses": pd.read_csv(base / "top_offense_names.csv"),
        "locations": pd.read_csv(base / "top_locations.csv"),
        "victim_age": pd.read_csv(base / "victim_age_distribution.csv"),
        "victim_race": pd.read_csv(base / "victim_race_by_offense.csv"),
        "victim_sex": pd.read_csv(base / "victim_sex_by_offense.csv"),
        "weapon": pd.read_csv(base / "weapon_usage.csv"),
        "state_rates": pd.read_csv(base / "state_crime_rates.csv"),
    }

    optional_files = {
        "day_of_week": "day_of_week_patterns.csv",
        "victim_type_severity": "victim_type_by_severity.csv",
        "relationship_severity": "relationship_by_severity.csv",
        "victim_age_range": "victim_age_distribution_summary.csv",
        "victim_age_night": "victim_age_distribution_nighttime.csv",
        "victim_age_vulnerable": "victim_age_vulnerability_nighttime.csv",
        "victim_sex_crosstab": "victim_sex_by_offense_crosstab.csv",
        "victim_sex_percent": "victim_sex_by_offense_percentages.csv",
    }

    for key, filename in optional_files.items():
        path = base / filename
        if path.exists():
            data[key] = pd.read_csv(path)
    return data

data = load_all_data(str(DATA_DIR))

#SIDEBAR CONTROLS
st.sidebar.title("⚙️ Controls")

year_options = sorted(data["monthly"]["Year"].unique())
selected_year = st.sidebar.selectbox("Select Year", year_options, index=len(year_options) - 1)

#KPI CARDS
total_incidents = int(data["yearly"]["count"].sum())
peak_year_row = data["yearly"].loc[data["yearly"]["count"].idxmax()]
latest_year_row = data["yearly"][data["yearly"]["Year"] == selected_year].iloc[0]

col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
col_kpi1.metric("Total incidents (5 yrs)", f"{total_incidents:,}")
col_kpi2.metric("Peak year", f"{int(peak_year_row['Year'])}", f"{int(peak_year_row['count']):,} incidents")
col_kpi3.metric("Selected year volume", f"{int(latest_year_row['count']):,}", f"Year {selected_year}")

st.markdown("---")

#HEATMAP 
st.subheader("Crime Hotspots (County Level)")

#Load heatmap data
heatmap_path = DATA_DIR / "county_heatmap.csv"
predictions_path = DATA_DIR / "hotspot_predictions.csv"

if heatmap_path.exists():
    #Read FIPS 
    heatmap_data = pd.read_csv(heatmap_path, dtype={'fips': str})
    
    #Ensuring FIPS are 5 digits
    heatmap_data['fips'] = heatmap_data['fips'].astype(str).str.zfill(5)
    
    map_mode = st.radio("Map View", ["Actual Incidents", "Predicted Hotspots (ML)"], horizontal=True)
    
    if map_mode == "Actual Incidents":
        #Aggregate all data by FIPS/County
        plot_data = heatmap_data.groupby(['fips', 'State', 'County'])['count'].sum().reset_index()
        
        plot_data['log_count'] = np.log1p(plot_data['count'])
        
        color_col = 'log_count'
        hover_data = {"State": True, "County": True, "count": True, "log_count": False, "fips": False}
        labels = {'count': 'Incidents', 'log_count': 'Severity (Log Scale)'}

        custom_scale = [
            [0.0, "#00FF00"],   # Green
            [0.5, "#FFFF00"],   # Yellow
            [1.0, "#FF0000"]    # Red
        ]
        
    else: # Predicted Hotspots
        if predictions_path.exists():
            preds_df = pd.read_csv(predictions_path)

            fips_map = heatmap_data[['State', 'County', 'fips']].drop_duplicates()
            
            preds_df = preds_df.rename(columns={'state': 'State', 'county': 'County'})
            
            # Aggregate by county ~ mean probability 
            plot_data = preds_df.groupby(['State', 'County'])['hotspot_probability'].mean().reset_index()
            plot_data = plot_data.merge(fips_map, on=['State', 'County'], how='inner')
            
            color_col = 'hotspot_probability'
            hover_data = {"State": True, "County": True, "hotspot_probability": True, "fips": False}
            labels = {'hotspot_probability': 'Hotspot Probability'}
            
            #Probability scale: 0 (Low) -> 1 (High)
            custom_scale = [
                [0.0, "#00FF00"],   # Green
                [0.5, "#FFFF00"],   # Yellow
                [1.0, "#FF0000"]    # Red
            ]
        else:
            st.warning("Prediction data not found. Please run training script.")
            plot_data = pd.DataFrame() # Empty
            color_col = None

    if not plot_data.empty:
        # Load GeoJSON for US Counties
        from urllib.request import urlopen
        import json
        
        @st.cache_data
        def get_geojson():
            with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
                counties = json.load(response)
            return counties

        counties_geojson = get_geojson()

        fig_map = px.choropleth(
            plot_data,
            geojson=counties_geojson,
            locations='fips',
            color=color_col, 
            color_continuous_scale=custom_scale,
            scope="usa",
            hover_data=hover_data,
            labels=labels
        )
        fig_map.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            geo=dict(
                bgcolor= 'rgba(0,0,0,0)',
                lakecolor='#263238',
                landcolor='#263238',
                subunitcolor='#455A64'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_map, use_container_width=True)
        if map_mode == "Actual Incidents":
            st.caption("Visualized based on incident severity (count) from Green (Low) to Red (High).")
        else:
            st.caption("Visualized based on predicted hotspot probability from Green (Low Risk) to Red (High Risk).")
else:
    st.warning("Heatmap data not found. Please run preprocessing.")

st.markdown("---")

# TRENDING OVER TIME
c1, c2 = st.columns(2)
with c1:
    st.subheader("Yearly trend")
    fig_yearly = px.line(
        data["yearly"],
        x="Year",
        y="count",
        markers=True,
        color_discrete_sequence=["#2a9d8f"]
    )
    fig_yearly.update_layout(margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Incidents")
    st.plotly_chart(fig_yearly, use_container_width=True)

with c2:
    st.subheader(f"Monthly pattern — {selected_year}")
    monthly_year = data["monthly"][data["monthly"]["Year"] == selected_year].copy()
    monthly_year["Month Name"] = monthly_year["Month"].apply(lambda m: calendar.month_abbr[int(m)])
    fig_monthly = px.area(
        monthly_year,
        x="Month Name",
        y="count",
        color_discrete_sequence=["#264653"]
    )
    fig_monthly.update_layout(margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Incidents")
    st.plotly_chart(fig_monthly, use_container_width=True)

#HOURLY DISTRIBUTION
st.subheader("Hour-of-day distribution")
fig_hourly = px.bar(
    data["hourly"],
    x="Incident Hour",
    y="count",
    labels={"count": "Incidents"},
    color_discrete_sequence=["#e76f51"]
)
fig_hourly.update_layout(margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_hourly, use_container_width=True)

st.markdown("---")

#OFFENSE INSIGHTS
st.subheader("Top offenses (counts)")
offense_top = data["offenses"].sort_values("count", ascending=False).head(15)
fig_offenses = px.bar(
    offense_top,
    y="Offense Name",
    x="count",
    orientation="h",
    labels={"count": "Incidents"},
    color_discrete_sequence=PALETTE
)
fig_offenses.update_layout(margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_offenses, use_container_width=True)

#LOCATION + WEAPON INSIGHTS
l1, l2 = st.columns(2)
with l1:
    st.subheader("Top locations")
    loc_top = data["locations"].sort_values("count", ascending=False).head(12)
    fig_loc = px.bar(
        loc_top,
        y="Location Name",
        x="count",
        orientation="h",
        labels={"count": "Incidents"},
        color_discrete_sequence=["#8ecae6"]
    )
    fig_loc.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_loc, use_container_width=True)

with l2:
    st.subheader("Weapon impact")
    weapon_top = data["weapon"].copy()
    weapon_top["percent"] = (weapon_top["count"] / weapon_top["count"].sum()) * 100
    weapon_top.sort_values("count", ascending=False, inplace=True)
    fig_weapon = px.pie(
        weapon_top,
        names="Weapon Name",
        values="count",
        hole=0.35,
        labels={"count": "Incidents"},
        color_discrete_sequence=PALETTE
    )
    fig_weapon.update_traces(textposition="inside", textinfo="percent+label")
    fig_weapon.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_weapon, use_container_width=True)

st.markdown("---")

#VICTIM DEMOGRAPHICS
d1, d2 = st.columns(2)
with d1:
    st.subheader("Victim age distribution")
    age_df = data.get("victim_age_range", data["victim_age"]).copy()
    x_col = "Age Range" if "Age Range" in age_df.columns else "Victim Age Group"
    fig_age = px.bar(
        age_df.sort_values(x_col),
        x=x_col,
        y="count",
        labels={"count": "Incidents"},
        color_discrete_sequence=["#0096c7"]
    )
    fig_age.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_age, use_container_width=True)

with d2:
    offense_categories = sorted(data["victim_race"]["Offense Category"].unique())
    selected_offense_cat = st.selectbox(
        "Victim demography by offense",
        offense_categories,
        key="victim_offense_select"
    )
    st.subheader(f"Victim race by offense — {selected_offense_cat}")
    race_filtered = data["victim_race"][data["victim_race"]["Offense Category"] == selected_offense_cat]
    fig_race = px.bar(
        race_filtered,
        x="Victim Race",
        y="count",
        labels={"count": "Incidents"},
        color="Victim Race",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_race.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_race, use_container_width=True)

st.subheader("Victim sex distribution by offense category")
sex_top_cats = (
    data["victim_sex"]
    .groupby("Offense Category")["count"].sum()
    .sort_values(ascending=False)
    .head(8)
    .index
)
sex_filtered = data["victim_sex"][data["victim_sex"]["Offense Category"].isin(sex_top_cats)]
fig_sex = px.bar(
    sex_filtered,
    x="Offense Category",
    y="count",
    color="Victim Sex",
    barmode="stack",
    labels={"count": "Incidents"}
)
fig_sex.update_layout(margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_sex, use_container_width=True)

if "victim_age_night" in data:
    st.markdown("---")
    st.subheader("Nighttime victim distribution")
    night_df = data["victim_age_night"]
    fig_night = px.bar(
        night_df,
        x="Age Range",
        y="Nighttime_Incident_Count",
        labels={"Nighttime_Incident_Count": "Incidents"},
        color_discrete_sequence=["#1b4332"]
    )
    fig_night.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_night, use_container_width=True)

# STATE COMPARISON
st.markdown("---")
st.subheader("State crime rate comparison")

state_rates = data["state_rates"].copy()
metric_choice = st.selectbox(
    "Metric",
    ["CrimeRatePer100k", "count"],
    format_func=lambda m: "Crime rate per 100k" if m == "CrimeRatePer100k" else "Total incidents",
    key="state_metric_select"
)

sorted_state_rates = state_rates.sort_values(metric_choice, ascending=False)
fig_states = px.bar(
    sorted_state_rates,
    x="State",
    y=metric_choice,
    text=metric_choice,
    labels={
        "CrimeRatePer100k": "Incidents per 100k (max pop)",
        "count": "Total incidents"
    },
    color="State",
    color_discrete_sequence=PALETTE
)
fig_states.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
fig_states.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), yaxis_title=None)
st.plotly_chart(fig_states, use_container_width=True)
st.caption("Data source: pre-aggregated outputs from EDA/Patterns_Analysis.ipynb")