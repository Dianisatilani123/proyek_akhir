import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF

# Langkah 2: Load dataset
def load_data():
    data = pd.read_csv("dataset_recruitment.csv")
    st.write("Dataset:")
    st.write(data.head(14))  # Show the first 14 rows
    return data

# Langkah 3: Standarisasi data
def preprocess_data(data):
    # Ubah nilai "<1" menjadi 0 dan nilai ">20" menjadi 25
    data['experience'] = data['experience'].apply(lambda x: 0 if x == '<1' else (25 if x == '>20' else int(x)))

    # Mengonversi fitur kategorikal ke dalam representasi numerik menggunakan label encoding
    label_encoder = LabelEncoder()
    categorical_cols = ['relevent_experience', 'enrolled_university', 'education_level', 
                        'ajor_discipline', 'company_size', 'company_type', 'last_new_job']
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])

    return data

# Langkah 4: Split data train dan test
def split_data(data):
    X = data.drop(columns=["gender", "city"])  # Hapus fitur "City"
    y = data["gender"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Langkah 5: Membuat data latih menggunakan algoritma machine learning
def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Langkah 6: Membuat model evaluasi untuk uji akurasi
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    
    st.write(f"Akurasi model: {accuracy * 100:.2f}%")
    st.write("Classification Report:")
    st.write(report)
    st.write("Confusion Matrix:")
    st.write(matrix)
    
    return accuracy

# Langkah 7: Membuat model untuk aplikasi
def main():
    st.markdown("<h1 style='text-align: center'>Aplikasi Rekrutmen Tanpa Bias Gender</h1>", unsafe_allow_html=True)

    # Navigation menu
    nav_options = ["Home", "Load Data", "Preprocess Data", "Split Data", "Train Model", "Evaluate Model", "Predict"]
    nav_selected = st.sidebar.selectbox("Navigation", nav_options)

    if nav_selected == "Home":
        st.write("Welcome to the Recruitment Application!")
    elif nav_selected == "Load Data":
        data = load_data()
    elif nav_selected == "Preprocess Data":
        data = preprocess_data(data)
    elif nav_selected == "Split Data":
        X_train, X_test, y_train, y_test = split_data(data)
    elif nav_selected == "Train Model":
        model = train_model(X_train, y_train)
    elif nav_selected == "Evaluate Model":
        accuracy = evaluate_model(model, X_test, y_test)
    elif nav_selected == "Predict":
        # Menampilkan form input untuk memprediksi kelayakan kandidat
        with st.sidebar:
            st.markdown("<h3>Masukkan Biodata Kandidat</h3>", unsafe_allow_html=True)
            
            enrollee_id = st.text_input("Enrollee ID", "")
            city = st.text_input("City", "")
            city_development_index = st.number_input("City Development Index", value=0.000, format="%.3f")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            relevent_experience = st.selectbox("Relevent Experience", ["Has relevent experience", "No relevent experience"])
            enrolled_university = st.selectbox("Enrolled University", ["no_enrollment", "Full time course", "Part time course"])
            education_level = st.selectbox("Education Level", ["Graduate", "Masters", "Phd"])
            major_discipline = st.selectbox("Major Discipline", ["STEM", "Business Degree", "Arts", "No Major", "Other"])
            experience = st.number_input("Experience", value=0)
            company_size = st.selectbox("Company Size", ["<10", "10-49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"])
            company_type = st.selectbox("Company Type", ["Pvt Ltd", "Funded Startup", "Public Sector", "Early Stage Startup", "NGO", "Other"])
            last_new_job = st.selectbox("Last New Job", ["never", "1", "2", "3", "4", ">4"])
            training_hours = st.number_input("Training Hours", value=0)

            # Tombol prediksi
            if st.button("Prediksi"):
                # Menerapkan logika prediksi
                kelayakan = 0  # Initialize kelayakan to 0
                if (relevent_experience == "Has relevent experience" and
                    (education_level == "Graduate" or education_level == "Masters") and
                    major_discipline == "STEM" and
                    (experience > 3 ) and
                    enrolled_university == "no_enrollment" and
                    training_hours > 50 and
                    last_new_job in ["1", "2", "3", "4", ">4"]):
                    kelayakan = 90  # Presentase kelayakan jika kandidat diterima
                elif (relevent_experience == "Has relevent experience" and
                      (education_level == "Graduate" or education_level == "Masters") and
                      major_discipline == "STEM" and
                      (experience > 2 ) and
                      enrolled_university == "no_enrollment" and
                      training_hours > 30):
                    kelayakan = 70  # Presentase kelayakan jika kandidat memiliki beberapa kriteria
                elif (relevent_experience == "Has relevent experience" and
                      (education_level == "Graduate" or education_level == "Masters") and
                      major_discipline == "STEM" and
                      (experience > 1 ) and
                      enrolled_university == "no_enrollment"):
                    kelayakan = 50  # Presentase kelayakan jika kandidat memiliki beberapa kriteria
                else:
                    kelayakan = 10  # Presentase kelayakan jika kandidat ditolak

                if kelayakan >= 70:
                    st.write("Kandidat diterima.")
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.set_text_color(0, 0, 0)  # Warna hitam untuk semua teks
                    pdf.cell(200, 10, txt="Hasil Prediksi Kelayakan Kandidat", ln=True, align="C")
                    pdf.ln(10)
                    pdf.cell(200, 10, txt=f"ID Kandidat: {enrollee_id}", ln=True)
                    pdf.cell(200, 10, txt=f"Presentase Kelayakan: {kelayakan}%", ln=True)
                    pdf.ln(10)
                    pdf.cell(200, 10, txt="Biodata Kandidat:", ln=True)
                    pdf.ln(5)
                    pdf.cell(100, 10, txt="City:", ln=False)
                    pdf.cell(100, 10, txt=f"{city}", ln=True)
                    pdf.cell(100, 10, txt="City Development Index:", ln=False)
                    pdf.cell(100, 10, txt=f"{city_development_index:.3f}", ln=True)
                    pdf.cell(100, 10, txt="Gender:", ln=False)
                    pdf.cell(100, 10, txt=f"{gender}", ln=True)
                    pdf.cell(100, 10, txt="Relevent Experience:", ln=False)
                    pdf.cell(100, 10, txt=f"{relevent_experience}", ln=True)
                    pdf.cell(100, 10, txt="Enrolled University:", ln=False)
                    pdf.cell(100, 10, txt=f"{enrolled_university}", ln=True)
                    pdf.cell(100, 10, txt="Education Level:", ln=False)
                    pdf.cell(100, 10, txt=f"{education_level}", ln=True)
                    pdf.cell(100, 10, txt="Major Discipline:", ln=False)
                    pdf.cell(100, 10, txt=f"{major_discipline}", ln=True)
                    pdf.cell(100, 10, txt="Experience:", ln=False)
                    pdf.cell(100, 10, txt=f"{experience}", ln=True)
                    pdf.cell(100, 10, txt="Company Size:", ln=False)
                    pdf.cell(100, 10, txt=f"{company_size}", ln=True)
                    pdf.cell(100, 10, txt="Company Type:", ln=False)
                    pdf.cell(100, 10, txt=f"{company_type}", ln=True)
                    pdf.cell(100, 10, txt="Last New Job:", ln=False)
                    pdf.cell(100, 10, txt=f"{last_new_job}", ln=True)
                    pdf.cell(100, 10, txt="Training Hours:", ln=False)
                    pdf.cell(100, 10, txt=f"{training_hours}", ln=True)
                    pdf.ln(10)
                    pdf.set_text_color(0, 128, 0)  # Warna hijau untuk keterangan diterima
                    pdf.cell(200, 10, txt="Keterangan: Kandidat diterima.", ln=True)
                else:
                    st.write("Kandidat ditolak.")
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.set_text_color(0, 0, 0)  # Warna hitam untuk semua teks
                    pdf.cell(200, 10, txt="Hasil Prediksi Kelayakan Kandidat", ln=True, align="C")
                    pdf.ln(10)
                    pdf.cell(200, 10, txt=f"ID Kandidat: {enrollee_id}", ln=True)
                    pdf.cell(200, 10, txt=f"Presentase Kelayakan: {kelayakan}%", ln=True)
                    pdf.ln(10)
                    pdf.cell(200, 10, txt="Biodata Kandidat:", ln=True)
                    pdf.ln(5)
                    pdf.cell(100, 10, txt="City:", ln=False)
                    pdf.cell(100, 10, txt=f"{city}", ln=True)
                    pdf.cell(100, 10, txt="City Development Index:", ln=False)
                    pdf.cell(100, 10, txt=f"{city_development_index:.3f}", ln=True)
                    pdf.cell(100, 10, txt="Gender:", ln=False)
                    pdf.cell(100, 10, txt=f"{gender}", ln=True)
                    pdf.cell(100, 10, txt="Relevent Experience:", ln=False)
                    pdf.cell(100, 10, txt=f"{relevent_experience}", ln=True)
                    pdf.cell(100, 10, txt="Enrolled University:", ln=False)
                    pdf.cell(100, 10, txt=f"{enrolled_university}", ln=True)
                    pdf.cell(100, 10, txt="Education Level:", ln=False)
                    pdf.cell(100, 10, txt=f"{education_level}", ln=True)
                    pdf.cell(100, 10, txt="Major Discipline:", ln=False)
                    pdf.cell(100, 10, txt=f"{major_discipline}", ln=True)
                    pdf.cell(100, 10, txt="Experience:", ln=False)
                    pdf.cell(100, 10, txt=f"{experience}", ln=True)
                    pdf.cell(100, 10, txt="Company Size:", ln=False)
                    pdf.cell(100, 10, txt=f"{company_size}", ln=True)
                    pdf.cell(100, 10, txt="Company Type:", ln=False)
                    pdf.cell(100, 10, txt=f"{company_type}", ln=True)
                    pdf.cell(100, 10, txt="Last New Job:", ln=False)
                    pdf.cell(100, 10, txt=f"{last_new_job}", ln=True)
                    pdf.cell(100, 10, txt="Training Hours:", ln=False)
                    pdf.cell(100, 10, txt=f"{training_hours}", ln=True)
                    pdf.ln(10)
                    pdf.set_text_color(255, 0, 0)  # Warna merah untuk keterangan ditolak
                    pdf.cell(200, 10, txt="Keterangan: Kandidat ditolak.", ln=True)

                st.write(f"Presentase kelayakan: {kelayakan}%")

                pdf_output = pdf.output(dest="S").encode("latin-1")

                st.download_button(
                    label="Download File",
                    data=pdf_output,
                    file_name=f"hasil_prediksi_{enrollee_id}.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()
