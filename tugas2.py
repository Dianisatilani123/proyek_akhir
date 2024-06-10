import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# CSS untuk mengubah gaya tombol dan memperindah tampilan aplikasi
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

        .sidebar .sidebar-content {
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Sidebar untuk navigasi
def sidebar_navigation():
    st.sidebar.title("Navigasi")
    options = st.sidebar.radio("Pilih halaman:", ["Home", "Prediksi", "Laporan Keanekaragaman"])
    return options

# Validasi input
def validate_input(data):
    required_columns = ['relevent_experience', 'enrolled_university', 'education_level', 
                        'major_discipline', 'company_size', 'company_type', 'last_new_job']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Kolom yang diperlukan tidak ada dalam dataset: {', '.join(missing_columns)}")
        return False
    return True

# Penanganan error saat memproses data
def load_data(file=None):
    try:
        if file:
            data = pd.read_csv(file)
        else:
            data = pd.read_csv("dataset_recruitment.csv")
        
        st.write("Dataset:")
        st.write(data.head(14))  # Show the first 14 rows
        st.write(f"Jumlah data pada dataset: {len(data)}")  # Menambahkan informasi jumlah data
        return data
    except Exception as e:
        st.error(f"Error saat memuat data: {e}")
        logging.error(f"Error saat memuat data: {e}")
        return None

def preprocess_data(data):
    try:
        # Ubah nilai "<1" menjadi 0 dan nilai ">20" menjadi 25
        data['experience'] = data['experience'].apply(lambda x: 0 if x == '<1' else (25 if x == '>20' else int(x)))

        # Mengonversi fitur kategorikal ke dalam representasi numerik menggunakan label encoding
        label_encoder = LabelEncoder()
        categorical_cols = ['relevent_experience', 'enrolled_university', 'education_level', 
                            'major_discipline', 'company_size', 'company_type', 'last_new_job']
        for col in categorical_cols:
            data[col] = label_encoder.fit_transform(data[col])

        st.write("Data setelah preprocessing:")
        st.write(data.head(14))  # Tampilkan 14 baris pertama data setelah preprocessing
        return data
    except Exception as e:
        st.error(f"Error saat memproses data: {e}")
        logging.error(f"Error saat memproses data: {e}")
        return None

# Split data train dan test
def split_data(data):
    try:
        X = data.drop(columns=["gender", "city"])  # Hapus fitur "City"
        y = data["gender"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
    except Exception as e:
        st.error(f"Error saat membagi data: {e}")
        logging.error(f"Error saat membagi data: {e}")
        return None, None, None, None

# Membuat model dan melatihnya
def train_model(X_train, y_train):
    try:
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        st.error(f"Error saat melatih model: {e}")
        logging.error(f"Error saat melatih model: {e}")
        return None

# Evaluasi model
def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)
        
        st.write(f"Akurasi model: {accuracy * 100:.2f}%")
        st.write("Classification Report:")
        st.write(report)
        st.write("Confusion Matrix:")
        st.write(matrix)
        
        # Visualisasi Confusion Matrix
        fig, ax = plt.subplots()
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        return accuracy
    except Exception as e:
        st.error(f"Error saat evaluasi model: {e}")
        logging.error(f"Error saat evaluasi model: {e}")
        return None

# Generate laporan keberagaman
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

    # Menambahkan grafik ke PDF
    for i, fig in enumerate(figures):
        fig.savefig(f"plot_{i}.png")
        pdf.image(f"plot_{i}.png", x=10, y=pdf.get_y(), w=180)
        pdf.ln(85)  # Sesuaikan nilai ini sesuai dengan ukuran gambar dan jarak antar gambar

    # Simpan laporan sebagai PDF
    pdf_output = "laporan_analitik_dan_keberagaman.pdf"
    pdf.output(pdf_output)
    return pdf_output

# Simpan dan muat model
def save_model(model):
    joblib.dump(model, 'model_recruitment.pkl')
    st.success("Model berhasil disimpan.")

def load_model():
    try:
        model = joblib.load('model_recruitment.pkl')
        st.success("Model berhasil dimuat.")
        return model
    except FileNotFoundError:
        st.error("Model tidak ditemukan, silakan latih dan simpan model terlebih dahulu.")
        return None

# Main function
def main():
    st.title("Sistem Analitik dan Keberagaman Pelamar Kerja")
    st.write("Aplikasi ini membantu dalam menganalisis dan menghasilkan laporan keberagaman pelamar kerja berdasarkan dataset yang diunggah.")

    # Tambahkan custom CSS
    add_custom_css()

    # Sidebar navigation
    page = sidebar_navigation()

    if page == "Home":
        st.write("Selamat datang di Sistem Analitik dan Keberagaman Pelamar Kerja.")
        st.write("Silakan navigasi menggunakan sidebar untuk memulai.")
    elif page == "Prediksi":
        st.write("Halaman Prediksi")

        # Upload file CSV
        uploaded_file = st.file_uploader("Unggah file CSV dataset", type=["csv"])

        if uploaded_file:
            data = load_data(uploaded_file)
            if validate_input(data):
                data = preprocess_data(data)
                if data is not None:
                    X_train, X_test, y_train, y_test = split_data(data)
                    if X_train is not None:
                        model = train_model(X_train, y_train)
                        if model is not None:
                            accuracy = evaluate_model(model, X_test, y_test)
                            save_model(model)
                            
                            if st.button("Muat Model yang Tersimpan"):
                                model = load_model()
                                if model:
                                    accuracy = evaluate_model(model, X_test, y_test)
    elif page == "Laporan Keanekaragaman":
        st.write("Halaman Laporan Keanekaragaman")

        # Upload file CSV
        uploaded_file = st.file_uploader("Unggah file CSV dataset", type=["csv"])

        if uploaded_file:
            data = load_data(uploaded_file)
            if validate_input(data):
                data = preprocess_data(data)
                if data is not None:
                    gender_counts, education_counts, experience_counts, company_type_counts, company_size_counts, discipline_counts, last_new_job_counts, figures = generate_diversity_report(data)
                    
                    # Export to PDF button
                    if st.button("Ekspor laporan ke PDF"):
                        pdf_output = export_report_to_pdf(data, gender_counts, education_counts, experience_counts, company_type_counts, company_size_counts, discipline_counts, last_new_job_counts, figures)
                        st.success(f"Laporan berhasil diekspor ke {pdf_output}")

if __name__ == "__main__":
    main()
