import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
import matplotlib.pyplot as plt

# Langkah 2: Load dataset
def load_data():
    data = pd.read_csv("dataset_recruitment.csv")
    st.write("Dataset:")
    st.write(data.head(14))  # Show the first 14 rows
    return data

# Langkah 3: Standarisasi data
def preprocess_data(data, fit_label_encoders=False):
    # Ubah nilai "<1" menjadi 0 dan nilai ">20" menjadi 25
    data['experience'] = data['experience'].apply(lambda x: 0 if x == '<1' else (25 if x == '>20' else int(x)))

    # Mengonversi fitur kategorikal ke dalam representasi numerik menggunakan label encoding
    categorical_cols = ['relevent_experience', 'enrolled_university', 'education_level', 
                        'major_discipline', 'company_size', 'company_type', 'last_new_job']
    if fit_label_encoders:
        global label_encoders
        label_encoders = {col: LabelEncoder().fit(data[col]) for col in categorical_cols}
        
    for col in categorical_cols:
        # Check if any new category not seen during training
        unseen_labels = set(data[col]) - set(label_encoders[col].classes_)
        if unseen_labels:
            # Add the unseen labels to the encoder
            label_encoders[col].classes_ = np.append(label_encoders[col].classes_, list(unseen_labels))
        data[col] = label_encoders[col].transform(data[col])

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

# Fungsi untuk menyimpan grafik sebagai gambar
def save_plot_as_image(fig, filename):
    fig.savefig(filename, format='png')

# Langkah 7: Membuat laporan analitik dan keberagaman
def generate_diversity_report(data):
    st.markdown("<h2>Laporan Analitik dan Keberagaman</h2>", unsafe_allow_html=True)
    
    # Menampilkan grafik dalam aplikasi Streamlit
    gender_counts = data['gender'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.bar(gender_counts.index, gender_counts.values)
    ax1.set_title("Jumlah pelamar berdasarkan gender")
    st.pyplot(fig1)
    
    education_counts = data['education_level'].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.bar(education_counts.index, education_counts.values)
    ax2.set_title("Jumlah pelamar berdasarkan tingkat pendidikan")
    st.pyplot(fig2)
    
    experience_counts = data['relevent_experience'].value_counts()
    fig3, ax3 = plt.subplots()
    ax3.bar(experience_counts.index, experience_counts.values)
    ax3.set_title("Jumlah pelamar berdasarkan pengalaman relevan")
    st.pyplot(fig3)
    
    company_type_counts = data['company_type'].value_counts()
    fig4, ax4 = plt.subplots()
    ax4.bar(company_type_counts.index, company_type_counts.values)
    ax4.set_title("Jumlah pelamar berdasarkan perusahaan sebelumnya")
    st.pyplot(fig4)
    
    company_size_counts = data['company_size'].value_counts()
    fig5, ax5 = plt.subplots()
    ax5.bar(company_size_counts.index, company_size_counts.values)
    ax5.set_title("Jumlah pelamar berdasarkan ukuran perusahaan sebelumnya")
    st.pyplot(fig5)
    
    discipline_counts = data['major_discipline'].value_counts()
    fig6, ax6 = plt.subplots()
    ax6.bar(discipline_counts.index, discipline_counts.values)
    ax6.set_title("Jumlah pelamar berdasarkan disiplin ilmu")
    st.pyplot(fig6)
    
    last_new_job_counts = data['last_new_job'].value_counts()
    fig7, ax7 = plt.subplots()
    ax7.bar(last_new_job_counts.index, last_new_job_counts.values)
    ax7.set_title("Jumlah pelamar berdasarkan waktu terakhir kali pindah kerja")
    st.pyplot(fig7)
    
    # Membuat file PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Laporan Analitik dan Keberagaman", ln=True, align="C")
    
    pdf.ln(10)
    pdf.cell(200, 10, txt="Jumlah pelamar berdasarkan gender", ln=True)
    pdf.image("gender_counts.png", x=10, y=30, w=190)
    
    pdf.add_page()
    pdf.cell(200, 10, txt="Jumlah pelamar berdasarkan tingkat pendidikan", ln=True)
    pdf.image("education_counts.png", x=10, y=30, w=190)
    
    pdf.add_page()
    pdf.cell(200, 10, txt="Jumlah pelamar berdasarkan pengalaman relevan", ln=True)
    pdf.image("experience_counts.png", x=10, y=30, w=190)
    
    pdf.add_page()
    pdf.cell(200, 10, txt="Jumlah pelamar berdasarkan perusahaan sebelumnya", ln=True)
    pdf.image("company_type_counts.png", x=10, y=30, w=190)
    
    pdf.add_page()
    pdf.cell(200, 10, txt="Jumlah pelamar berdasarkan ukuran perusahaan sebelumnya", ln=True)
    pdf.image("company_size_counts.png", x=10, y=30, w=190)
    
    pdf.add_page()
    pdf.cell(200, 10, txt="Jumlah pelamar berdasarkan disiplin ilmu", ln=True)
    pdf.image("discipline_counts.png", x=10, y=30, w=190)
    
    pdf.add_page()
    pdf.cell(200, 10, txt="Jumlah pelamar berdasarkan waktu terakhir kali pindah kerja", ln=True)
    pdf.image("last_new_job_counts.png", x=10, y=30, w=190)
    
    pdf_output = pdf.output(dest="S").encode("latin-1")

    st.download_button(
        label="Download Laporan Keanekaragaman",
        data=pdf_output,
        file_name="laporan_keanekaragaman.pdf",
        mime="application/pdf"
    )

# Langkah 8: Membuat model untuk aplikasi
def main():
    st.markdown("<h1 style='text-align: center'>Aplikasi Rekrutmen Tanpa Bias Gender</h1>", unsafe_allow_html=True)

    # Navigasi header
    navigation = st.sidebar.selectbox("Navigasi", ["HOME", "Prediksi", "Laporan Keanekaragaman"])

    if navigation == "HOME":
        st.write("Selamat datang di Aplikasi Rekrutmen Tanpa Bias Gender!")
    elif navigation == "Prediksi":
        # Load data
        data = load_data()

        # Preprocessing data
        data = preprocess_data(data, fit_label_encoders=True)

        # Split data
        X_train, X_test, y_train, y_test = split_data(data)

        # Train model
        model = train_model(X_train, y_train)

        # Evaluate model
        accuracy = evaluate_model(model, X_test, y_test)
        st.write(f"Akurasi model: {accuracy * 100:.2f}%")

        # Menampilkan form input untuk memprediksi kelayakan kandidat
        with st.sidebar:
            st.markdown("<h1>Masukkan Biodata Kandidat</h1>", unsafe_allow_html=True)
            
            enrollee_id = st.text_input("Enrollee ID", "")
            city = st.text_input("City", "")
            city_development_index = st.number_input("City Development Index", value=0.000, format="%.3f")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            relevent_experience = st.selectbox("Relevent Experience", ["Has relevent experience", "No relevent experience"])
            enrolled_university = st.selectbox("Enrolled University", ["no_enrollment", "Full time course", "Part time course"])
            education_level = st.selectbox("Education Level", ["Graduate", "Masters", "Phd"])
            major_discipline = st.selectbox("Major Discipline", ["STEM", "Business Degree", "Arts", "Humanities", "No Major"])
            experience = st.number_input("Experience (Years)", min_value=0, max_value=25, step=1)
            company_size = st.selectbox("Company Size", ["<10", "10/49", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"])
            company_type = st.selectbox("Company Type", ["Private", "Public", "NGO", "Government", "Other"])
            last_new_job = st.selectbox("Last New Job", ["never", "1", "2", "3", "4", "4+"])

            predict_button = st.button("Prediksi")
            
            if predict_button:
                input_data = {
                    "enrollee_id": enrollee_id,
                    "city": city,
                    "city_development_index": city_development_index,
                    "gender": gender,
                    "relevent_experience": relevent_experience,
                    "enrolled_university": enrolled_university,
                    "education_level": education_level,
                    "major_discipline": major_discipline,
                    "experience": experience,
                    "company_size": company_size,
                    "company_type": company_type,
                    "last_new_job": last_new_job
                }
                
                input_df = pd.DataFrame([input_data])
                input_df = preprocess_data(input_df)
                input_df = input_df.drop(columns=["gender", "city"])

                prediction = model.predict(input_df)
                st.write(f"Hasil prediksi: {prediction[0]}")
                
    elif navigation == "Laporan Keanekaragaman":
        data = load_data()
        generate_diversity_report(data)

if __name__ == "__main__":
    main()
