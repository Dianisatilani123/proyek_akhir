import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# CSS untuk mengubah gaya tombol
def add_custom_css():
    st.markdown(
        """
        <style>
        .stButton button {
            background-color: #4CAF50; /* Hijau */
            color: white;
            border: none;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 12px;
            transition-duration: 0.4s;
        }

        .stButton button:hover {
            background-color: white;
            color: black;
            border: 2px solid #4CAF50;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Langkah 2: Load dataset
def load_data():
    data = pd.read_csv("dataset_recruitment.csv")
    st.write("Dataset:")
    st.write(data.head(14))  # Show the first 14 rows
    st.write(f"Jumlah data pada dataset: {len(data)}")  # Menambahkan informasi jumlah data
    return data

# Langkah 3: Standarisasi data
def preprocess_data(data):
    # Ubah nilai "<1" menjadi 0 dan nilai ">20" menjadi 25
    data['experience'] = data['experience'].apply(lambda x: 0 if x == '<1' else (25 if x == '>20' else int(x)))

    # Mengonversi fitur kategorikal ke dalam representasi numerik menggunakan label encoding
    label_encoder = LabelEncoder()
    categorical_cols = ['relevent_experience', 'enrolled_university', 'education_level', 
                        'major_discipline', 'company_size', 'company_type', 'last_new_job']
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

    # Visualisasi distribusi hasil prediksi
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.countplot(x=y_pred, ax=ax1)
    ax1.set_title("Distribusi Hasil Prediksi")
    ax1.set_xlabel("Label Prediksi")
    ax1.set_ylabel("Jumlah")
    st.pyplot(fig1)

    # Visualisasi confusion matrix
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title("Confusion Matrix")
    ax2.set_xlabel("Prediksi")
    ax2.set_ylabel("Aktual")
    st.pyplot(fig2)

    return accuracy

# Langkah 7: Membuat laporan analitik dan keberagaman
def generate_diversity_report(data):
    st.markdown("<h2>Laporan Analitik dan Keberagaman</h2>", unsafe_allow_html=True)

    figures = []

    # Plotting gender counts
    st.write("Jumlah pelamar berdasarkan gender:")
    gender_counts = data['gender'].value_counts()
    fig1, ax1 = plt.subplots()
    sns.barplot(x=gender_counts.index, y=gender_counts.values, ax=ax1)
    ax1.set_title("Jumlah pelamar berdasarkan gender")
    ax1.set_xlabel("Gender")
    ax1.set_ylabel("Jumlah")
    st.pyplot(fig1)
    figures.append(fig1)

    # Plotting education level counts
    st.write("Jumlah pelamar berdasarkan tingkat pendidikan:")
    education_counts = data['education_level'].value_counts()
    fig2, ax2 = plt.subplots()
    sns.barplot(x=education_counts.index, y=education_counts.values, ax=ax2)
    ax2.set_title("Jumlah pelamar berdasarkan tingkat pendidikan")
    ax2.set_xlabel("Tingkat Pendidikan")
    ax2.set_ylabel("Jumlah")
    st.pyplot(fig2)
    figures.append(fig2)

    # Plotting relevant experience counts
    st.write("Jumlah pelamar berdasarkan pengalaman relevan:")
    experience_counts = data['relevent_experience'].value_counts()
    fig3, ax3 = plt.subplots()
    sns.barplot(x=experience_counts.index, y=experience_counts.values, ax=ax3)
    ax3.set_title("Jumlah pelamar berdasarkan pengalaman relevan")
    ax3.set_xlabel("Pengalaman Relevan")
    ax3.set_ylabel("Jumlah")
    st.pyplot(fig3)
    figures.append(fig3)

    # Plotting company type counts
    st.write("Jumlah pelamar berdasarkan perusahaan sebelumnya:")
    company_type_counts = data['company_type'].value_counts()
    fig4, ax4 = plt.subplots()
    sns.barplot(x=company_type_counts.index, y=company_type_counts.values, ax=ax4)
    ax4.set_title("Jumlah pelamar berdasarkan perusahaan sebelumnya")
    ax4.set_xlabel("Tipe Perusahaan")
    ax4.set_ylabel("Jumlah")
    st.pyplot(fig4)
    figures.append(fig4)

    # Plotting company size counts
    st.write("Jumlah pelamar berdasarkan ukuran perusahaan sebelumnya:")
    company_size_counts = data['company_size'].value_counts()
    fig5, ax5 = plt.subplots()
    sns.barplot(x=company_size_counts.index, y=company_size_counts.values, ax=ax5)
    ax5.set_title("Jumlah pelamar berdasarkan ukuran perusahaan sebelumnya")
    ax5.set_xlabel("Ukuran Perusahaan")
    ax5.set_ylabel("Jumlah")
    st.pyplot(fig5)
    figures.append(fig5)

    # Plotting major discipline counts
    st.write("Jumlah pelamar berdasarkan disiplin ilmu:")
    discipline_counts = data['major_discipline'].value_counts()
    fig6, ax6 = plt.subplots()
    sns.barplot(x=discipline_counts.index, y=discipline_counts.values, ax=ax6)
    ax6.set_title("Jumlah pelamar berdasarkan disiplin ilmu")
    ax6.set_xlabel("Disiplin Ilmu")
    ax6.set_ylabel("Jumlah")
    st.pyplot(fig6)
    figures.append(fig6)

    # Plotting last new job counts
    st.write("Jumlah pelamar berdasarkan waktu terakhir kali pindah kerja:")
    last_new_job_counts = data['last_new_job'].value_counts()
    fig7, ax7 = plt.subplots()
    sns.barplot(x=last_new_job_counts.index, y=last_new_job_counts.values, ax=ax7)
    ax7.set_title("Jumlah pelamar berdasarkan waktu terakhir kali pindah kerja")
    ax7.set_xlabel("Waktu Terakhir Pindah Kerja")
    ax7.set_ylabel("Jumlah")
    st.pyplot(fig7)
    figures.append(fig7)

    return gender_counts, education_counts, experience_counts, company_type_counts, company_size_counts, discipline_counts, last_new_job_counts, figures

# Ekspor laporan ke PDF
def export_report_to_pdf(data, gender_counts, education_counts, experience_counts, company_type_counts, company_size_counts, discipline_counts, last_new_job_counts, figures):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Menambahkan konten ke PDF
    pdf.cell(200, 10, txt="Laporan Analitik dan Keberagaman", ln=True, align='C')

    # Menambahkan jumlah pelamar berdasarkan gender
    pdf.cell(200, 10, txt="Jumlah pelamar berdasarkan gender:", ln=True)
    for gender, count in gender_counts.items():
        pdf.cell(200, 10, txt=f"{gender}: {count}", ln=True)

        # Menambahkan jumlah pelamar berdasarkan tingkat pendidikan
    pdf.cell(200, 10, txt="Jumlah pelamar berdasarkan tingkat pendidikan:", ln=True)
    for education, count in education_counts.items():
        pdf.cell(200, 10, txt=f"{education}: {count}", ln=True)

    # Menambahkan jumlah pelamar berdasarkan pengalaman relevan
    pdf.cell(200, 10, txt="Jumlah pelamar berdasarkan pengalaman relevan:", ln=True)
    for experience, count in experience_counts.items():
        pdf.cell(200, 10, txt=f"{experience}: {count}", ln=True)

    # Menambahkan jumlah pelamar berdasarkan perusahaan sebelumnya
    pdf.cell(200, 10, txt="Jumlah pelamar berdasarkan perusahaan sebelumnya:", ln=True)
    for company_type, count in company_type_counts.items():
        pdf.cell(200, 10, txt=f"{company_type}: {count}", ln=True)

    # Menambahkan jumlah pelamar berdasarkan ukuran perusahaan sebelumnya
    pdf.cell(200, 10, txt="Jumlah pelamar berdasarkan ukuran perusahaan sebelumnya:", ln=True)
    for company_size, count in company_size_counts.items():
        pdf.cell(200, 10, txt=f"{company_size}: {count}", ln=True)

    # Menambahkan jumlah pelamar berdasarkan disiplin ilmu
    pdf.cell(200, 10, txt="Jumlah pelamar berdasarkan disiplin ilmu:", ln=True)
    for discipline, count in discipline_counts.items():
        pdf.cell(200, 10, txt=f"{discipline}: {count}", ln=True)

    # Menambahkan jumlah pelamar berdasarkan waktu terakhir kali pindah kerja
    pdf.cell(200, 10, txt="Jumlah pelamar berdasarkan waktu terakhir kali pindah kerja:", ln=True)
    for last_new_job, count in last_new_job_counts.items():
        pdf.cell(200, 10, txt=f"{last_new_job}: {count}", ln=True)

    # Simpan laporan sebagai file PDF
    pdf_file = "laporan_analitik_dan_keberagaman.pdf"
    pdf.output(pdf_file)

    # Fungsi untuk mengubah file PDF menjadi base64
    def get_pdf_download_link(pdf_file):
        with open(pdf_file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="{pdf_file}">Unduh laporan PDF</a>'
        return href

    st.markdown(get_pdf_download_link(pdf_file), unsafe_allow_html=True)

# Langkah 8: Prediksi gaji dengan menyesuaikan UMR Jakarta
def predict_salary_with_umr(model, X_test):
    y_pred_salary = model.predict(X_test)
    umr_jakarta = 4725000  # Nilai UMR Jakarta terbaru

    # Sesuaikan prediksi gaji jika lebih rendah dari UMR Jakarta
    y_pred_salary_adjusted = [max(salary, umr_jakarta) for salary in y_pred_salary]
    
    return y_pred_salary_adjusted

# Streamlit UI
def main():
    st.title("Analisis Data Pelamar dan Prediksi Gaji")
    add_custom_css()
    
    # Langkah 1: Load dataset
    data = load_data()

    # Langkah 2: Preprocessing data
    data = preprocess_data(data)

    # Langkah 3: Split data train dan test
    X_train, X_test, y_train, y_test = split_data(data)

    # Langkah 4: Train model
    model = train_model(X_train, y_train)

    # Langkah 5: Evaluate model
    evaluate_model(model, X_test, y_test)

    # Langkah 6: Generate diversity report
    gender_counts, education_counts, experience_counts, company_type_counts, company_size_counts, discipline_counts, last_new_job_counts, figures = generate_diversity_report(data)

    # Langkah 7: Export report to PDF
    export_report_to_pdf(data, gender_counts, education_counts, experience_counts, company_type_counts, company_size_counts, discipline_counts, last_new_job_counts, figures)

    # Langkah 8: Predict salary with UMR adjustment
    y_pred_salary_adjusted = predict_salary_with_umr(model, X_test)
    st.write("Prediksi gaji setelah penyesuaian dengan UMR Jakarta:")
    st.write(y_pred_salary_adjusted)

if __name__ == "__main__":
    main()

