import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
data1 = pd.read_csv('dataset1.csv')
data2 = pd.read_csv('dataset2.csv')
data3 = pd.read_csv('dataset3.csv')
data4 = pd.read_csv('dataset4.csv')

# Merge datasets based on common columns
data = pd.concat([data1, data2, data3, data4], axis=0, ignore_index=True)

# Print column names to check if 'IKG', 'IDG', and 'RLS' exist
print("Column names:", data.columns)

# Define feature columns
feature_cols = ['IKG', 'IDG', 'RLS']

# Check if feature columns exist in the dataset
if not all(col in data.columns for col in feature_cols):
    print("Error: One or more feature columns are missing from the dataset.")
    exit()

# Split data into features and target
X = data[feature_cols]
y = data['MPK20']

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create a model using the Random Forest algorithm
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create a Streamlit app
st.title("Gender Equality Analysis and Prediction")

# Prediction page
st.header("Prediction")
with st.form("prediction_form"):
    IKG = st.number_input("IKG")
    IDG = st.number_input("IDG")
    RLS = st.number_input("RLS")
    submit_button = st.form_submit_button("Predict")

if submit_button:
    X_new = pd.DataFrame({'IKG': [IKG], 'IDG': [IDG], 'RLS': [RLS]})
    X_new_scaled = scaler.transform(X_new)
    y_pred = model.predict(X_new_scaled)
    st.write("Predicted MPK20:", y_pred[0])

# Analysis page
st.header("Analysis")
with st.expander("Data Analysis"):
    data_analysis = data.groupby('Kabupaten/Kota')[feature_cols + ['MPK20']].mean()
    st.write(data_analysis)

# Comparison page
st.header("Comparison")
with st.expander("Comparison of IKG, IDG, RLS, and MPK20 values across kabupaten/kota"):
    data_comparison = data.groupby('Kabupaten/Kota')[feature_cols + ['MPK20']].mean().reset_index()
    st.write(data_comparison)

# Recommendation page
st.header("Recommendation")
with st.expander("Recommendations for improving gender equality in Provinsi Nusa Tenggara Barat"):
    recommendation = "Berikut adalah rekomendasi untuk meningkatkan kesetaraan gender di Provinsi Nusa Tenggara Barat:... "
    st.write(recommendation)

if __name__ == '__main__':
    st.run()