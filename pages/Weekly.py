import datetime
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import requests
import joblib

data = pd.read_csv('polusi udara.csv')
df1 = data.copy()
df1['pm10'] = df1['pm10'].fillna((df1['pm10'].median()))
df1['pm25'] = df1['pm25'].fillna((df1['pm25'].median()))
df1['o3'] = df1['o3'].fillna((df1['o3'].median()))
df1['max'] = df1['max'].fillna((df1['max'].median()))

y = df1["max"]
x = df1[['pm10', 'pm25', 'o3']]

lr = LinearRegression(fit_intercept=True)
lr.fit(x, y)

model_file = open("Linreg_Air_Quality.pkl", "wb")
joblib.dump(lr, model_file)
model_file.close()

real_time_data = pd.DataFrame(columns=["pm25", "Value"])

api_url = "https://api.waqi.info/feed/here/?token=c52ccf49b2567789207ac64811bca82be9d8c31c"


def fetch_realtime_data():
        response = requests.get(api_url)
        if response.status_code == 200:
            rtdata = response.json()
            return rtdata
        else:
            st.write("Error fetching data")


rt_data_generator = fetch_realtime_data()

st.title("Weekly Air Quality Prediction")
st.write("Welcome to weekly air quality information website. We are dedicated  to delivering valuable insights about the air you breathe. Our mission is to empower you to make informed decisions that contribute to your overall health and well-being. Data are provided by BMKG and will updated every saturday.")

rtdata = rt_data_generator


loc = rtdata["data"]["city"]
location_name = loc["name"]

st.write(f"**Your location: {location_name}**")

st.subheader("Weekly Air Quality")



o3_data = rtdata["data"]["forecast"]["daily"]["o3"]
o3_df = pd.DataFrame(o3_data)
o3_df['day'] = pd.to_datetime(o3_df['day']).dt.date

pm25_data = rtdata["data"]["forecast"]["daily"]["pm25"]
pm25_df = pd.DataFrame(pm25_data)
pm25_df['day'] = pd.to_datetime(pm25_df['day']).dt.date

pm10_data = rtdata["data"]["forecast"]["daily"]["pm10"]
pm10_df = pd.DataFrame(pm10_data)
pm10_df['day'] = pd.to_datetime(pm10_df['day']).dt.date


d = st.date_input("Select Date", datetime.date(2023, 10, 23))
st.write('Air prediction in:', d)

filtered_o3 = o3_df[o3_df['day'] == d]
filtered_pm25 = pm25_df[pm25_df['day'] == d]
filtered_pm10 = pm10_df[pm10_df['day'] == d]

if not filtered_o3.empty and not filtered_pm25.empty and not filtered_pm10.empty:
    o3 = filtered_o3['avg'].values[0]
    pm25 = filtered_pm25['avg'].values[0]
    pm10 = filtered_pm10['avg'].values[0]

    new_value = [[pm10, pm25, o3]]
    new_prediction = lr.predict(new_value)

    if new_prediction <= 50:
        st.success("Good 😊")
        st.write("Air Quality in your area currently in :green[good condition], it means that the air we breathe is free from major pollutants and poses little to no risk to our health, Here are a few ways you can help keep our air quality at its best:")
        st.write("1. Conserve Energy: Use energy-efficient appliances and reduce your energy consumption. The less energy we use, the fewer emissions are released into the atmosphere.")
        st.write("2. Reduce Car Emissions: Consider carpooling, using public transportation, biking, or walking when possible. These actions can significantly reduce air pollution from vehicles.")
        st.write("3. Plant Trees and Maintain Green Spaces: Trees and greenery help absorb pollutants and improve air quality. Participate in local tree-planting initiatives or take care of community gardens..")
    elif new_prediction >= 51 and new_prediction <= 100:
        st.warning("Moderate 😐")
        st.write("Air Quality in your area currently in :orange[moderate condition], it means that while the general population is not likely to be affected, individuals who are sensitive to air pollution should take precautions. This is an opportune moment for all of us to come together and take steps to maintain and improve our air quality. Here are some actions you can take:")
        st.write("1. Conserve Energy: Use energy-efficient appliances and reduce your energy consumption. The less energy we use, the fewer emissions are released into the atmosphere.")
        st.write("2. Reduce Car Emissions: Consider carpooling, using public transportation, biking, or walking when possible. These actions can significantly reduce air pollution from vehicles.")
        st.write("3. Stay Informed: Stay updated on air quality reports and forecasts. Use this information to make informed decisions about outdoor activities, especially for sensitive individuals..")
    elif new_prediction >= 101 and new_prediction <= 200:
        st.error("Unhealthy 😷")
        st.write("Air Quality in your area currently in :red[unhealthy condition], it means that everyone may begin to experience health effects, and individuals in sensitive groups, such as children, the elderly, and those with respiratory or heart conditions, are at a heightened risk of more severe health issues. The situation at hand requires an immediate and collective response. We must take the following steps to protect ourselves and our community: ")
        st.write("1. Limit Outdoor Activities: It's crucial that we minimize outdoor activities, especially strenuous exercise, during Unhealthy air quality conditions.")
        st.write("2. Use Air Filters: Consider using air purifiers with HEPA filters to improve indoor air quality.")
        st.write("3. Particulate Matter Protection: Masks designed to filter particulate matter can help you avoid inhaling fine particles that can be detrimental to your health.")
    else:
        st.error("Hazardous 🥵")
        st.write("Air Quality in your area currently in :red[Hazardous condition], it means  air pollution reaches dangerous levels, and everyone is at risk of severe health effects. It is of utmost importance that we take immediate steps to protect ourselves and our community:")
        st.write("1. Stay Indoors: It is crucial that you remain indoors as much as possible and keep doors and windows closed. Use air purifiers with HEPA filters to improve indoor air quality.")
        st.write("2. Avoid Outdoor Activities: All outdoor activities, including exercising, should be suspended during this time.")
        st.write("3. Check on Vulnerable Individuals: Please check on neighbors, friends, and family members, especially those who are elderly, have respiratory conditions, or young children, to ensure their safety.")         
