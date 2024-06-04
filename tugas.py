import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
import streamlit as st

# Load dataset
data = pd.read_csv('aug_train.csv')
print(data.head())

# Define features for model training
features = ['relevent_experience', 'enrolled_university', 
            'education_level', 'training_hours', 'gender']
target = 'target'

# Drop rows with missing values in selected features and target
data = data.dropna(subset=features + [target])

# Encode enrolled_university to numeric
enrolled_university_mapping = {
    'no_enrollment': 0,
    'Full time course': 1,
    'Part time course': 2
}
data['enrolled_university'] = data['enrolled_university'].map(enrolled_university_mapping)

# Encode education_level to numeric
education_level_mapping = {
    'Graduate': 0,
    'Masters': 1,
    'Phd': 2
}
data['education_level'] = data['education_level'].map(education_level_mapping)

# Convert relevent_experience to numeric
relevent_experience_mapping = {
    'Has relevent experience': 1,
    'No relevent experience': 0
}
data['relevent_experience'] = data['relevent_experience'].map(relevent_experience_mapping)

# Encode gender to numeric
gender_mapping = {
    'Male': 0,
    'Female': 1,
    'Other': 2
}
data['gender'] = data['gender'].map(gender_mapping)

# Split data into train and test
X = data[features]
y = data[target]

# Check for missing values
na_count = X.isna().sum().sum()
if na_count > 0:
    print(f"Found {na_count} missing values.")
    imputer = SimpleImputer(strategy='most_frequent')  # Use most frequent strategy
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)  # Impute missing values
    print("Missing values imputed.")

print("Shape of X after imputing NaN values:", X.shape)

# Ensure all columns are numeric
X = X.astype(float)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
svm = SVC(random_state=42)
svm.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = svm.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)
print(report)

def predict_acceptance(input_data):
    """
    Predict whether a candidate will be accepted based on input features.

    Args:
    input_data (list): List of input features.

    Returns:
    int: 1 if the candidate will be accepted, 0 otherwise.
    """
    relevent_experience_mapping = {
        "Has relevent experience": 1,
        "No relevent experience": 0
    }
    enrolled_university_mapping = {
        'no_enrollment': 0,
        'Full time course': 1,
        'Part time course': 2
    }
    education_level_mapping = {
        'Graduate': 0,
        'Masters': 1,
        'Phd': 2
    }
    
    relevent_experience = input_data[0]
    enrolled_university = input_data[1]
    education_level = input_data[2]
    training_hours = input_data[3]

    relevent_experience = relevent_experience_mapping[relevent_experience]
    enrolled_university = enrolled_university_mapping[enrolled_university]
    education_level = education_level_mapping[education_level]

    input_features = np.array([relevent_experience, enrolled_university, education_level, training_hours]).reshape(1, -1)
    input_features = scaler.transform(input_features)  # Transform input data
    prediction = svm.predict(input_features)
    return prediction[0]

def main():
    """
    Main function to create a Streamlit app.
    """
    st.title("AI Deteksi Bias Gender pada Perekrutan Kerja")

    st.write("Masukkan fitur-fitur untuk memprediksi apakah kandidat diterima:")

    relevent_experience = st.selectbox("Relevent Experience", ["Has relevent experience", "No relevent experience"])
    enrolled_university = st.selectbox("Enrolled University", ['no_enrollment', 'Full time course', 'Part time course'])
    education_level = st.selectbox("Education Level", ['Graduate', 'Masters', 'Phd'])
    training_hours = st.slider("Training Hours", min_value=0, step=1)

if st.button("Prediksi"):
    try:
        result = predict_acceptance([relevent_experience, enrolled_university, education_level, gender, training_hours])
        if result == 1:
            st.write("Kandidat diterima")
        else:
            st.write("Kandidat ditolak")
        st.write(f"Akurasi model: {accuracy * 100:.2f}%")
        st.write(report)
    except Exception as e:
        st.write("Error:", str(e))


if __name__ == "__main__":
    main()
