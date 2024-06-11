import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
from sqlalchemy import create_engine
import base64
import matplotlib.pyplot as plt
import seaborn as sns

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
def load_data(file):
    data = pd.read_csv(file)
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

    # Menyimpan grafik sebagai gambar dan menambahkannya ke PDF
    for i, fig in enumerate(figures, start=1):
        img_path = f"figure_{i}.png"
        fig.savefig(img_path)
        pdf.add_page()
        pdf.image(img_path, x=10, y=10, w=pdf.w - 20)

    pdf_file = "Laporan_Keberagaman.pdf"
    pdf.output(pdf_file)
    
    return pdf_file

# Fungsi untuk menampilkan tautan unduhan
def download_file(file_path):
    with open(file_path, "rb") as file:
        btn = st.download_button(
            label="Download Laporan",
            data=file,
            file_name=file_path,
            mime="application/octet-stream"
        )
        return btn

# Halaman login
def login():
    st.markdown("<h2>Login Admin</h2>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
       if username == "admin" and password == "admin123":
            st.session_state['logged_in'] = True
            st.success("Login berhasil!")
            st.experimental_rerun()  # Refresh halaman setelah login berhasil
    else:
            st.error("Username atau Password salah!")
           
# Tombol logout
def logout():
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.success("Logout berhasil!")
        st.experimental_rerun()  # Refresh halaman setelah logout berhasil

# Langkah 8: Membuat model untuk aplikasi
def main():
    st.markdown("<h1 style='text-align: center'>Aplikasi Rekrutmen Tanpa Bias Gender</h1>", unsafe_allow_html=True)
    add_custom_css()  # Tambahkan CSS khusus untuk tombol

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        login()
    else:
        # Navigasi header
        navigation = st.sidebar.selectbox("Navigasi", ["HOME", "Prediksi", "Laporan Keanekaragaman", "Upload Dataset"])

        if navigation == "HOME":
            st.write("Selamat datang di Aplikasi Rekrutmen Tanpa Bias Gender!")
        elif navigation == "Prediksi":
            # Load data
            data = load_data("dataset_recruitment.csv")

            # Preprocessing data
            data = preprocess_data(data)

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
                major_discipline = st.selectbox("Major Discipline", ["STEM", "Business Degree", "Arts", "No Major", "Other"])
                experience = st.number_input("Experience", value=0)
                company_size = st.selectbox("Company Size", ["<10", "10-49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"])
                company_type = st.selectbox("Company Type", ["Pvt Ltd", "Funded Startup", "Public Sector", "Early Stage Startup", "NGO", "Other"])
                last_new_job = st.selectbox("Last New Job", ["never", "1", "2", "3", "4", ">4"])
                training_hours = st.number_input("Training Hours", value=0)

                # Tombol prediksi
                prediksi_button = st.button("Prediksi")

                if prediksi_button:
                    if (enrollee_id == "" or city == "" or gender == "" or relevent_experience == "" or 
                        enrolled_university == "" or education_level == "" or major_discipline == "" or 
                        experience == 0 or company_size == "" or company_type == "" or last_new_job == "" or 
                        training_hours == 0):
                        st.error("Silakan isi semua form inputan terlebih dahulu!")
                    else:
                       # Menerapkan logika prediksi
                        kelayakan = 0  # Initialize kelayakan to 0
                        if (relevent_experience == "Has relevent experience" and
                        (education_level == "Graduate" or education_level == "Masters" or education_level == "Phd") and
                        major_discipline == "STEM" and  # Tambahkan syarat Major Discipline wajib STEM
                        training_hours >= 50):
                         kelayakan = 1

                        if kelayakan == 1:
                            st.success("Kandidat layak untuk dipertimbangkan!")
                        else:
                            st.error("Kandidat tidak layak untuk dipertimbangkan!")

        elif navigation == "Laporan Keanekaragaman":
            data = load_data("dataset_recruitment.csv")
            gender_counts, education_counts, experience_counts, company_type_counts, company_size_counts, discipline_counts, last_new_job_counts, figures = generate_diversity_report(data)
            if st.button("Export Laporan ke PDF"):
                pdf_file = export_report_to_pdf(data, gender_counts, education_counts, experience_counts, company_type_counts, company_size_counts, discipline_counts, last_new_job_counts, figures)
                st.success("Laporan berhasil diekspor ke PDF!")
                download_file(pdf_file)

         elif navigation == "Upload Dataset":
        st.markdown("<h2>Upload Dataset</h2>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
        if uploaded_file:
            data = load_data(uploaded_file)
            data = preprocess_data(data)
            st.success("Dataset berhasil di-upload dan diproses!")
            
            # Visualisasi Hasil Prediksi
            st.write("Visualisasi Hasil Prediksi:")
            st.write("Distribusi hasil prediksi menggunakan plot bar:")
            
            # Code for bar plot visualization
            pred_counts = data['prediction'].value_counts()
            fig_pred, ax_pred = plt.subplots()
            sns.barplot(x=pred_counts.index, y=pred_counts.values, ax=ax_pred)
            ax_pred.set_title("Distribusi Hasil Prediksi")
            ax_pred.set_xlabel("Hasil Prediksi")
            ax_pred.set_ylabel("Jumlah")
            st.pyplot(fig_pred)
            
            st.write("Performa model secara visual menggunakan Confusion Matrix:")
            
            # Code for confusion matrix visualization
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='g', cmap='Blues', cbar=False)
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.title('Confusion Matrix')
            st.pyplot()
            
            st.write("Fitur Save dan Load Model:")
            st.write("Anda dapat menyimpan model yang sudah dilatih dan memuatnya kembali tanpa harus melatih dari awal.")

            # Button export pdf dan download
            if st.button("Export Dataset ke PDF"):
                export_data_to_pdf(data)
                
        # Tombol logout
        logout()

if __name__ == "__main__":
    main()

