import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import geohash2 as gh2
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import requests

# Load the trained model
xgb_best = joblib.load('may2024_model_compressed.pkl')

data_structure = pd.read_csv('data_format_may.csv')

# Function to preprocess input data
def preprocess_input(data):
    tr2 = pd.DataFrame(columns=data_structure.columns)
    tr2.loc[0] = 0
    tr2.loc[0, 'otaq_sayi'] = data['otaq_sayi'][0]
    tr2.loc[0, 'sahe_kvm'] = np.log(data['sahe_kvm'][0])
    tr2.loc[0, 'mertebe_yer'] = data['mertebe_yer'][0]
    tr2.loc[0, 'mertebe_say'] = data['mertebe_say'][0]

    tr2.loc[0, 'kateqoriyaYeni tikili'] = 1 if data['kateqoriya'][0] == 'Yeni tikili' else 0

    tr2.loc[0, 'ipotekavar'] = 1 if data['ipoteka'][0] == 'var' else 0
    tr2.loc[0, 'ipotekayoxdur'] = 1 if data['ipoteka'][0] == 'yoxdur' else 0

    tr2.loc[0, 'temirvar'] = 1 if data['temir'][0] == 'var' else 0

    tr2.loc[0, 'geohash' + data['geohash'][0]] = 1

#tr2.loc[0, 'year_month' + data['year_month'][0]] = 1 if data['year_month'][0] != '2023-09' else pass
    if data['year_month'][0] != '2023-08':
        tr2.loc[0, 'year_month' + data['year_month'][0]] = 1
    tr2.loc[0, 'mertebe_ratio'] = float(data['mertebe_yer'][0]) / float(data['mertebe_say'][0])
    tr2.loc[0, 'area_per_room'] = np.log(float(data['sahe_kvm'][0]) / float(data['otaq_sayi'][0]))
    tr2.loc[0,'tikili_temir']=tr2.loc[0,'kateqoriyaYeni tikili']*tr2.loc[0,'temirvar']
    tr2.loc[0,'ipoteka_temir']=tr2.loc[0,'ipotekavar']*tr2.loc[0,'temirvar']

    return tr2

# Function to get coordinates from address
def get_coordinates(address):
    geolocator = Nominatim(user_agent="myGeocoder")
    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            st.error("Address not found!")
            return None, None
    except GeocoderTimedOut:
        st.error("Geocoding service timed out. Please try again.")
        return None, None

# Function to get address suggestions using Google Places API
def get_address_suggestions(query, api_key):
    url = f"https://maps.googleapis.com/maps/api/place/autocomplete/json?input={query}&key={api_key}&components=country:az"
    response = requests.get(url)
    if response.status_code == 200:
        predictions = response.json().get('predictions', [])
        suggestions = [prediction['description'] for prediction in predictions]
        return suggestions
    else:
        st.error("Error fetching address suggestions")
        return []

# Streamlit app
def main():
    api_key = "AIzaSyCpaXsr9psJYjFXp8eHPgbGe-9pxEvLIWE"  # Replace with your Google Places API key

    menu = ["Visual", "Mənzil qiymətləndirməsi", "Kirayə qiymətləndirməsi", "..."]
    choices = st.sidebar.selectbox("Menu", menu)

    if choices == 'Mənzil qiymətləndirməsi':
        st.title("Mənzil Qiymətini Müəyyənləşdir")

        # Input fields
        col1, col2 = st.columns([1, 1])
        with col1:
            otaq_sayi = st.number_input("Otaq sayı", min_value=1, value=2, max_value=15)
        with col2:
            sahe_kvm = st.number_input("Sahə (kv.m)", min_value=10, value=95, max_value=500)
        col1, col2 = st.columns([1, 1])  
        with col1:  
            mertebe_yer = st.number_input("Mərtəbə", min_value=0, value=10, max_value=50)
        with col2:
            mertebe_say = st.number_input("Mərtəbə sayı", min_value=1, value=20, max_value=50)
        col1, col2 = st.columns([1, 1])  
        with col1:
            kateqoriya = st.selectbox("Kateqoriya", ['Yeni tikili', 'Köhnə tikili'])
        with col2:
            temir = st.selectbox("Təmir", ['var', 'yoxdur'])
        col1, col2 = st.columns([1, 1])
        with col1:
            ipoteka = st.selectbox("İpoteka", ['var', 'yoxdur'])
        with col2:
            year_month = st.selectbox("Qimətləndirmə tarixi", ['2024-05','2024-04', '2024-03', '2024-02', '2024-01', '2023-12', '2023-11', '2023-10','2023-09'])

        # Initialize default coordinates
        latitude, longitude = None, None

        # Address input or map selection
        location_method = st.radio("Select location method", ["Select on map","Enter address"])

        if location_method == "Enter address":
            address = st.text_input("Enter the address")
            if address:
                suggestions = get_address_suggestions(address, api_key)
                st.write("Suggestions: ", suggestions)  # Debug statement
                if suggestions:
                    selected_address = st.selectbox("Suggestions", suggestions)
                    st.write("Selected Address: ", selected_address)  # Debug statement
                    if selected_address:
                        if st.button("Get Coordinates and Predict Price"):
                            latitude, longitude = get_coordinates(selected_address)
                            st.write("Coordinates: ", latitude, longitude)  # Debug statement
                            if latitude and longitude:
                                st.success(f"Coordinates: Latitude: {latitude}, Longitude: {longitude}")
                                # Predict price
                                data = {
                                    'otaq_sayi': [otaq_sayi],
                                    'sahe_kvm': [sahe_kvm],
                                    'mertebe_yer': [mertebe_yer],
                                    'mertebe_say': [mertebe_say],
                                    'kateqoriya': [kateqoriya],
                                    'ipoteka': [ipoteka],
                                    'temir': [temir],
                                    'geohash': [gh2.encode(latitude, longitude, precision=6)],
                                    'year_month': [year_month]
                                }
                                tr2 = preprocess_input(data)
                                predicted_price = xgb_best.predict(tr2)[0]
                                predicted_price = np.exp(predicted_price)
                                st.success(f"Predicted house price: {predicted_price:.2f} AZN")
        else:
            st.write("Select the location on the map:")
            default_location = [40.3874646, 49.8030282]

            # Initialize folium map
            m = folium.Map(location=default_location, zoom_start=12)

            # Add a marker to the map that updates on click
            marker = folium.Marker(location=default_location, draggable=True)
            marker.add_to(m)

            # Display the map
            map_data = st_folium(m, width=700, height=500)

            # Update marker position based on user click
            if map_data and map_data['last_clicked']:
                latitude = map_data['last_clicked']['lat']
                longitude = map_data['last_clicked']['lng']
                # Update the marker position
                marker.location = [latitude, longitude]
                folium.Marker(location=[latitude, longitude], draggable=True).add_to(m)
                st.write(f"Selected location: Latitude: {latitude}, Longitude: {longitude}")
            else:
                latitude = default_location[0]
                longitude = default_location[1]


        # Predict button for map selection
        if latitude is not None and longitude is not None:
            data = {
                'otaq_sayi': [otaq_sayi],
                'sahe_kvm': [sahe_kvm],
                'mertebe_yer': [mertebe_yer],
                'mertebe_say': [mertebe_say],
                'kateqoriya': [kateqoriya],
                'ipoteka': [ipoteka],
                'temir': [temir],
                'geohash': [gh2.encode(latitude, longitude, precision=6)],
                'year_month': [year_month]
            }

            if st.button("Qiymətləndir"):
                tr2 = preprocess_input(data)
                predicted_price = xgb_best.predict(tr2)[0]
                predicted_price = np.exp(predicted_price)
                st.success(f"Predicted house price: {predicted_price:.2f} AZN")

            show_graph = st.checkbox("Digər dövrlər üzrə qiymətləndir")
            # Plot predicted prices for all possible values of year_month
            if show_graph:
                st.subheader("")
                year_months = ['2023-10', '2023-11', '2023-12', '2024-01', '2024-02', '2024-03', '2024-04']
                predicted_prices = []
                for month in year_months:
                    data['year_month'] = [month]
                    tr2 = preprocess_input(data)
                    predicted_price = xgb_best.predict(tr2)[0]
                    predicted_prices.append(np.exp(predicted_price))

                fig = go.Figure(data=go.Scatter(x=year_months, y=predicted_prices, mode='lines+markers'))
                fig.update_layout(title='Dövrlər üzrə mənzilin qiyməti',
                                  xaxis_title='Tarix',
                                  yaxis_title='Qiymət (AZN)')
                st.plotly_chart(fig)
        else:
            st.warning("Please provide a location either by entering an address or selecting on the map.")

    if choices == 'Visual':
        st.title("Mənzil elanları, May 2024")
        st.components.v1.html(open('hyperlink_error_may.html').read(), width=1000, height=1200) 

if __name__ == "__main__":
    main()
