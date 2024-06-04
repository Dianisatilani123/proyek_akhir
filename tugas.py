import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import streamlit as st

# Load dataset
data = pd.read_csv('aug_train.csv')
print(data.head())

# Define all features including gender and enrollee_id
all_features = ['enrollee_id', 'relevent_experience', 'enrolled_university', 
                'education_level', 'training_hours', 'gender', 'city_development_index', 'major_discipline']

# Define features for model training
features = ['relevent_experience', 'enrolled_university', 
            'education_level', 'training_hours', 'city_development_index', 'major_discipline']
target = 'target'

# Drop rows with missing values in selected features and target
data = data.dropna(subset=features + [target])

# Encode categorical variables
mapping_dict = {
    'enrolled_university': {
        'no_enrollment': 0,
        'Full time course': 1,
        'Part time course': 2
    },
    'education_level': {
        'Graduate': 0,
        'Masters': 1,
        'Phd': 2
    },
    'relevent_experience': {
        'Has relevent experience': 1,
        'No relevent experience': 0
    },
    'gender': {
        'Male': 0,
        'Female': 1,
        'Other': 2
    },
    'major_discipline': {
        'STEM': 0,
        'Business Degree': 1,
        'Arts': 2,
        'Humanities': 3,
        'No Major': 4,
        'Other': 5
    }
}

for col, mapping in mapping_dict.items():
    data[col] = data[col].map(mapping)

# Fill remaining NaN values in features with the most frequent value
imputer = SimpleImputer(strategy='most_frequent')
data[features] = imputer.fit_transform(data[features])

# Split data into train and test
X = data[features]
y = data[target]

# Handle imbalanced dataset with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split resampled data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and tune model using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5)
grid_search = GridSearchCV(rf, param_grid, cv=cv, n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Evaluate model for accuracy
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

def predict_acceptance(input_data):
    """
    Predict whether a candidate will be accepted based on input features.

    Args:
    input_data (list): List of input features.

    Returns:
    int: 1 if the candidate will be accepted, 0 otherwise.
    """
    input_data[1] = mapping_dict['relevent_experience'][input_data[1]]
    input_data[2] = mapping_dict['enrolled_university'][input_data[2]]
    input_data[3] = mapping_dict['education_level'][input_data[3]]
    input_data[5] = mapping_dict['major_discipline'][input_data[5]]
    input_features = input_data[1:]
    input_features = np.array(input_features, dtype=float)  # Convert to float array
    input_features = input_features.reshape(1, -1)  # Reshape to 2D array
    input_features = scaler.transform(input_features)  # Transform input data
    prediction = best_model.predict(input_features)
    return prediction[0]

def main():
    """
    Main function to create a Streamlit app.
    """
    st.title("AI Deteksi Bias Gender pada Perekrutan Kerja")

    st.write("Masukkan fitur-fitur untuk memprediksi apakah kandidat diterima:")

    enrollee_id = st.text_input("Enrollee ID", "")
    relevent_experience = st.selectbox("Relevent Experience", ["Has relevent experience", "No relevent experience"])
    enrolled_university = st.selectbox("Enrolled University", list(mapping_dict['enrolled_university'].keys()), index=0)
    education_level = st.selectbox("Education Level", list(mapping_dict['education_level'].keys()), index=0)
    gender = st.selectbox("Gender", list(mapping_dict['gender'].keys()), index=0)
    training_hours = st.slider("Training Hours", min_value=0, step=1)
    city_development_index = st.slider("City Development Index", min_value=0.0, max_value=1.0, step=0.01)
    major_discipline = st.selectbox("Major Discipline", list(mapping_dict['major_discipline'].keys()), index=0)

    if st.button("Prediksi"):
        try:
            result = predict_acceptance([enrollee_id, relevent_experience, enrolled_university, education_level, training_hours, major_discipline, city_development_index, gender])
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
