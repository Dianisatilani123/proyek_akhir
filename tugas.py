import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

# Load dataset
data = pd.read_csv('aug_train.csv')

st.title("Rekrutmen Tanpa Bias")
st.write("Dataset:")
st.write(data.head())

# Ensure 'city' column exists
if 'city' not in data.columns:
    st.error("'city' column not found in the dataset.")
else:
    # Preprocessing
    # Drop columns that may cause bias
    data = data.drop(['enrollee_id', 'gender'], axis=1)

    # Handle missing values
    data = data.dropna()

    # Convert categorical data to numerical
    data = pd.get_dummies(data)

    # Split features and target
    X = data.drop('target', axis=1)
    y = data['target']

    # Check if target variable is binary
    if len(y.unique())!= 2:
        st.error("Target variable is not binary. Please ensure it has only two unique values (0 and 1).")
    else:
        # Check for class imbalance
        class_counts = y.value_counts()
        st.write("Class distribution:")
        st.write(class_counts)
        if class_counts[0] / class_counts[1] > 5 or class_counts[1] / class_counts[0] > 5:
            st.warning("Class imbalance detected. You may want to consider class weighting, oversampling, or undersampling.")

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Initialize the Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train the model
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Accuracy:", accuracy)
        st.write("Classification Report:")
        st.write(classification_report(y_test, y_pred))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))

# Create two columns
col1, col2 = st.columns(2)

# Form input data kandidat di kolom kiri
with col1:
    st.write("Form Input Data Kandidat:")
    city = st.text_input('City')
    city_development_index = st.number_input('City Development Index')
    relevent_experience = st.selectbox('Relevent Experience', ['Has relevent experience', 'No relevent experience'])
    enrolled_university = st.selectbox('Enrolled University', ['no_enrollment', 'Full time course', 'Part time course'])
    education_level = st.selectbox('Education Level', ['Graduate', 'Masters', 'High School', 'Primary School'])
    major_discipline = st.selectbox('Major Discipline', ['STEM', 'Business Degree', 'Humanities'])
    experience = st.selectbox('Experience', ['<1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '>20'])
    company_size = st.selectbox('Company Size', ['50-99', '100-500', '500-999', 'Oct-49'])
    company_type = st.selectbox('Company Type', ['Pvt Ltd', 'Funded Startup', 'Public Sector'])
    last_new_job = st.selectbox('Last New Job', ['never', '1', '2', '3', '4', '>4'])
    training_hours = st.number_input('Training Hours')

# Button prediksi di kolom kanan
with col2:
    if st.button('Predict'):
        input_data = {
            'city': city,
            'city_development_index': city_development_index,
            'elevent_experience': relevent_experience,
            'enrolled_university': enrolled_university,
            'education_level': education_level,
            'ajor_discipline': major_discipline,
            'experience': experience,
            'company_size': company_size,
            'company_type': company_type,
            'last_new_job': last_new_job,
            'training_hours': training_hours
        }
        result = predict(input_data)
        st.write(f'Result: {result}')
