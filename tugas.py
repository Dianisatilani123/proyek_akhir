import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

# Load dataset
data = pd.read_csv('aug_train.csv')

# Display the dataset
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

    # One-hot encode categorical data
    data = pd.get_dummies(data, columns=['city', 'elevent_experience', 'enrolled_university', 'education_level', 'ajor_discipline', 'company_size', 'company_type', 'last_new_job'])

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

        # Streamlit application
        st.title("Rekrutmen Tanpa Bias")

        # Get unique values for categorical columns
        unique_city = data['city'].unique()
        unique_relevent_experience = data['relevent_experience'].unique()
        unique_enrolled_university = data['enrolled_university'].unique()
        unique_education_level = data['education_level'].unique()
        unique_major_discipline = data['major_discipline'].unique()
        unique_experience = data['experience'].unique()
        unique_company_size = data['company_size'].unique()
        unique_company_type = data['company_type'].unique()
        unique_last_new_job = data['last_new_job'].unique()

        # Form input data kandidat
        input_data = {
            'city': st.selectbox('City', unique_city),
            'city_development_index': st.number_input('City Development Index'),
            'elevent_experience': st.selectbox('Relevent Experience', unique_relevent_experience),
            'enrolled_university': st.selectbox('Enrolled University', unique_enrolled_university),
            'education_level': st.selectbox('Education Level', unique_education_level),
            'ajor_discipline': st.selectbox('Major Discipline', unique_major_discipline),
            'experience': st.selectbox('Experience', unique_experience),
            'company_size': st.selectbox('Company Size', unique_company_size),
            'company_type': st.selectbox('Company Type', unique_company_type),
            'last_new_job': st.selectbox('Last New Job', unique_last_new_job),
            'training_hours': st.number_input('Training Hours')
        }

        def predict(input_data):
            input_df = pd.DataFrame([input_data])
            input_df = pd.get_dummies(input_df, columns=['city', 'elevent_experience', 'enrolled_university', 'education_level', 'ajor_discipline', 'company_size', 'company_type', 'last_new_job'])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
            return prediction

        # Predict button
        if st.button('Predict'):
            result = predict(input_data)
            st.write(f'Result: {result[0]}')
