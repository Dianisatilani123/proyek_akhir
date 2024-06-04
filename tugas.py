import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib
import streamlit as st

# Load dataset
df = pd.read_csv('aug_train.csv')

# Drop unnecessary columns
df = df.drop(columns=['gender'])

# Fill missing values for numeric columns with median
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Fill missing values for categorical columns with mode
categorical_columns = df.select_dtypes(exclude=[np.number]).columns
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# Encode categorical variables
df = pd.get_dummies(df, columns=categorical_columns)

# Save the list of columns for later use
columns_after_dummies = df.columns.tolist()
columns_after_dummies.remove('target')  # Remove target from columns list

# Scale numeric columns
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Split data
X = df.drop(columns=['target'])
y = df['target']

# Convert y to a classification target
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(columns_after_dummies, 'columns_after_dummies.joblib')

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"Classification Report:\n{report}")

# Streamlit application
st.title('Aplikasi Rekrutmen Tanpa Bias')

# Load the saved model, scaler, and column names
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
columns_after_dummies = joblib.load('columns_after_dummies.joblib')

# Input fields
enrollee_id = st.text_input('Enrollee ID')
city = st.text_input('Kota')
city_development_index = st.number_input('City Development Index', min_value=0.0, max_value=1.0)
gender = st.selectbox('Jenis Kelamin', ['Male', 'Female', 'Unknown'])
relevent_experience = st.selectbox('Relevent Experience', ['Has relevent experience', 'No relevent experience'])

enrolled_university = st.selectbox('Enrolled University', ['no_enrollment', 'Full time course', 'Part time course'])
education_level = st.selectbox('Education Level', ['Graduate', 'Masters', 'High School', 'Primary School'])
major_discipline = st.selectbox('Major Discipline', ['STEM', 'Business Degree', 'Humanities', 'Unknown'])
last_new_job = st.selectbox('Last New Job', ['never', '1', '2', '3', '4', '>4'])
experience = st.selectbox('Experience', ['>1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '>20'])
training_hours = st.number_input('Training Hours', min_value=0)

# Create input data
input_data = pd.DataFrame({
    'city_development_index': [city_development_index],
    'relevent_experience': [1 if relevent_experience == 'Has relevent experience' else 0],
    'experience': [int(experience.replace('>','')) if experience != '>20' else 20],
    'training_hours': [training_hours]
}, index=[0])

# Add dummy columns for categorical variables
for col in ['enrolled_university', 'education_level', 'major_discipline', 'last_new_job']:
    input_data[f'{col}_{eval(col)}'] = 1

# Ensure all columns are present
for col in columns_after_dummies:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns to match the training data
input_data = input_data[columns_after_dummies]

# Scale input data
input_data[numeric_columns] = scaler.transform(input_data[numeric_columns])

# Predict
if st.button('Predict'):
    try:
        prediction = model.predict(input_data)
        if prediction == 1:
            st.success('Kandidat Berpotensi Diterima')
        else:
            st.error('Kandidat Tidak Berpotensi Diterima')
    except Exception as e:
        st.error(f"Error: {str(e)}")
