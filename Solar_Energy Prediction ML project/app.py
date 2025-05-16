import streamlit as st
import pandas as pd
import joblib

# Load your saved model
best_model = joblib.load('best_model.pkl')

st.title("🌞 Solar Power Prediction App")

st.markdown("""
Enter the weather and time details manually or upload a CSV/Excel file with multiple records to predict solar power output.
""")

# --- Manual Input Section ---
st.header("Manual Input")

day_of_year = st.number_input('Day of Year', min_value=1, max_value=366, value=120)
year = st.selectbox('Year', [2020, 2021, 2022, 2023], index=2)
month = st.number_input('Month', min_value=1, max_value=12, value=4)
day = st.number_input('Day', min_value=1, max_value=31, value=30)
first_hour_of_period = st.number_input('First Hour of Period', min_value=0, max_value=23, value=12)
is_daylight = st.checkbox('Is Daylight?', value=True)
distance_to_solar_noon = st.number_input('Distance to Solar Noon', value=1.5)
avg_temp_day = st.number_input('Average Temperature (day)', value=28)
avg_wind_dir_day = st.number_input('Average Wind Direction (day)', value=180)
avg_wind_speed_day = st.number_input('Average Wind Speed (day)', value=4.2)
visibility = st.number_input('Visibility', value=10.0)
rel_humidity = st.number_input('Relative Humidity', value=60)
avg_wind_speed_period = st.number_input('Average Wind Speed (period)', value=4.5)
avg_barometric_pressure_period = st.number_input('Average Barometric Pressure (period)', value=1013.2)
season_spring = st.checkbox('Season: Spring')
season_summer = st.checkbox('Season: Summer')
season_winter = st.checkbox('Season: Winter')
time_of_day_evening = st.checkbox('Time of Day: Evening')
time_of_day_morning = st.checkbox('Time of Day: Morning')
time_of_day_night = st.checkbox('Time of Day: Night')
solar_proximity_midday = st.checkbox('Solar Proximity: Midday')
solar_proximity_near_noon = st.checkbox('Solar Proximity: Near Noon')
sky_cover_1 = st.checkbox('Sky Cover 1')
sky_cover_2 = st.checkbox('Sky Cover 2')
sky_cover_3 = st.checkbox('Sky Cover 3')
sky_cover_4 = st.checkbox('Sky Cover 4')

manual_data = {
    'day of year': [day_of_year],
    'year': pd.Categorical([year]),
    'month': [month],
    'day': [day],
    'first hour of period': pd.Categorical([first_hour_of_period]),
    'is daylight': [is_daylight],
    'distance to solar noon': [distance_to_solar_noon],
    'average temperature (day)': [avg_temp_day],
    'average wind direction (day)': [avg_wind_dir_day],
    'average wind speed (day)': [avg_wind_speed_day],
    'visibility': [visibility],
    'relative humidity': [rel_humidity],
    'average wind speed (period)': [avg_wind_speed_period],
    'average barometric pressure (period)': [avg_barometric_pressure_period],
    'season_spring': [season_spring],
    'season_summer': [season_summer],
    'season_winter': [season_winter],
    'time of day_evening': [time_of_day_evening],
    'time of day_morning': [time_of_day_morning],
    'time of day_night': [time_of_day_night],
    'solar proximity_midday': [solar_proximity_midday],
    'solar proximity_near noon': [solar_proximity_near_noon],
    'sky cover_1': [sky_cover_1],
    'sky cover_2': [sky_cover_2],
    'sky cover_3': [sky_cover_3],
    'sky cover_4': [sky_cover_4]
}

manual_df = pd.DataFrame(manual_data)

# Ensure manual_df matches model's expected features
model_columns = best_model.feature_names_in_ if hasattr(best_model, 'feature_names_in_') else manual_df.columns

# Add any missing columns with default values
for col in model_columns:
    if col not in manual_df.columns:
        if pd.api.types.is_bool_dtype(manual_df.dtypes[0]):
            manual_df[col] = False
        else:
            manual_df[col] = 0

# Reorder columns to match model input
manual_df = manual_df[model_columns]


# --- Button to predict manual input ---
if st.button("Predict Solar Power for Manual Input"):
    prediction = best_model.predict(manual_df)
    st.success(f"Predicted Solar Power Output: {prediction[0]:.2f} W/m²")

st.markdown("---")

# --- Batch Prediction Section ---
st.header("Batch Prediction via File Upload")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_upload = pd.read_csv(uploaded_file)
        else:
            df_upload = pd.read_excel(uploaded_file)
        
        # Optional: Check and preprocess df_upload columns to match training data columns here
        # For example, convert categorical columns if needed, fill missing columns with zeros, reorder columns.
        
        # Make sure 'year' and 'first hour of period' columns are categorical
        if 'year' in df_upload.columns:
            df_upload['year'] = df_upload['year'].astype('category')
        if 'first hour of period' in df_upload.columns:
            df_upload['first hour of period'] = df_upload['first hour of period'].astype('category')
        if 'is daylight' in df_upload.columns:
            df_upload['is daylight'] = df_upload['is daylight'].astype(bool)
        
        # Add missing columns with 0 or False (adjust this as per your model columns)
        model_columns = best_model.feature_names_in_ if hasattr(best_model, 'feature_names_in_') else manual_df.columns
        for col in model_columns:
            if col not in df_upload.columns:
                if manual_df[col].dtype == bool:
                    df_upload[col] = False
                else:
                    df_upload[col] = 0
        
        # Reorder columns to match model input
        df_upload = df_upload[model_columns]

        predictions = best_model.predict(df_upload)
        df_upload['Predicted Solar Power Output'] = predictions

        st.write("Prediction Results:")
        st.dataframe(df_upload)

        csv = df_upload.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name='solar_power_predictions.csv',
            mime='text/csv',
        )
    except Exception as e:
        st.error(f"Error processing file: {e}")

