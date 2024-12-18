import streamlit as st
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

st.set_page_config(page_title="Viz Demo")

# Get the current working directory using pathlib
current_dir = Path.cwd()
print(f"Current Working Directory: {current_dir}")

# Define the relative path to the models directory
models_dir = current_dir / "models"

# Path to the pickle files
df_path = models_dir / "df.pkl"
pipeline_path = models_dir / "pipeline.pkl"

# Check and load df.pkl
if df_path.is_file():
    with open(df_path, "rb") as file:
        df = pickle.load(file)
else:
    raise FileNotFoundError(f"{df_path} not found. Check your file path.")

# Check and load pipeline.pkl
if pipeline_path.is_file():
    with open(pipeline_path, "rb") as file:
        pipeline = pickle.load(file)
else:
    raise FileNotFoundError(f"{pipeline_path} not found. Check your file path.")



st.header('Enter your inputs')

# property_type
property_type = st.selectbox('Property Type',['flat','house'])

# sector
sector = st.selectbox('Sector',sorted(df['sector'].unique().tolist()))

bedrooms = float(st.selectbox('Number of Bedroom',sorted(df['bedRoom'].unique().tolist())))

bathroom = float(st.selectbox('Number of Bathrooms',sorted(df['bathroom'].unique().tolist())))

balcony = st.selectbox('Balconies',sorted(df['balcony'].unique().tolist()))

property_age = st.selectbox('Property Age',sorted(df['agePossession'].unique().tolist()))

built_up_area = float(st.number_input('Built Up Area'))

servant_room = float(st.selectbox('Servant Room',[0.0, 1.0]))
store_room = float(st.selectbox('Store Room',[0.0, 1.0]))

furnishing_type = st.selectbox('Furnishing Type',sorted(df['furnishing_type'].unique().tolist()))
luxury_category = st.selectbox('Luxury Category',sorted(df['luxury_category'].unique().tolist()))
floor_category = st.selectbox('Floor Category',sorted(df['floor_category'].unique().tolist()))

if st.button('Predict'):

    # form a dataframe
    data = [[property_type, sector, bedrooms, bathroom, balcony, property_age, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
               'agePossession', 'built_up_area', 'servant room', 'store room',
               'furnishing_type', 'luxury_category', 'floor_category']

    # Convert to DataFrame
    one_df = pd.DataFrame(data, columns=columns)

    #st.dataframe(one_df)

    # predict
    base_price = np.expm1(pipeline.predict(one_df))[0]
    low = base_price - 0.22
    high = base_price + 0.22

    # display
    st.text("The price of the flat is between {} Cr and {} Cr".format(round(low,2),round(high,2)))