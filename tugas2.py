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
    
    st.write(f"Akurasi model: {accuracy * .2f}%")
    st.write("Classification Report:")
    st.write(report)
    st.write("Confusion Matrix:")
    st.write(matrix)
    
    return accuracy

# Langkah 7: Membuat model untuk aplikasi
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

                    st.write(f"Presentase kelayakan: {kelayakan}%")
                    if kelayakan >= 70:
                        st.write("Kandidat diterima.")
                    else:
                        st.write("Kandidat ditolak.")

                    # Membuat file PDF hasil prediksi
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt=f"HASIL SELEKSI KANDIDAT", ln=True, align="C")
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

                    if kelayakan == 90:
                        pdf.cell(200, 10, txt="Kriteria dan Tingkat Kelayakan:", ln=True)
                        pdf.cell(200, 10, txt="- Pengalaman Relevan: Kandidat memiliki pengalaman yang relevan.", ln=True)
                        pdf.cell(200, 10, txt="- Tingkat Pendidikan: Kandidat memiliki gelar Sarjana atau Magister.", ln=True)
                        pdf.cell(200, 10, txt="- Disiplin Utama: Kandidat berasal dari bidang STEM.", ln=True)
                        pdf.cell(200, 10, txt="- Pengalaman Kerja: Kandidat memiliki pengalaman kerja lebih dari 3 tahun.", ln=True)
                        pdf.cell(200, 10, txt="- Status Pendaftaran Universitas: Kandidat tidak sedang terdaftar di universitas.", ln=True)
                        pdf.cell(200, 10, txt="- Jam Pelatihan: Kandidat memiliki lebih dari 50 jam pelatihan.", ln=True)
                        pdf.cell(200, 10, txt="- Durasi Pekerjaan Terakhir: Kandidat telah bekerja dalam durasi tertentu pada pekerjaan terakhir mereka (1-4 tahun atau lebih).", ln=True)
                    elif kelayakan == 70:
                        pdf.cell(200, 10, txt="Kriteria dan Tingkat Kelayakan:", ln=True)
                        pdf.cell(200, 10, txt="- Pengalaman Relevan: Kandidat memiliki pengalaman yang relevan.", ln=True)
                        pdf.cell(200, 10, txt="- Tingkat Pendidikan: Kandidat memiliki gelar Sarjana atau Magister.", ln=True)
                        pdf.cell(200, 10, txt="- Disiplin Utama: Kandidat berasal dari bidang STEM.", ln=True)
                        pdf.cell(200, 10, txt="- Pengalaman Kerja: Kandidat memiliki pengalaman kerja lebih dari 2 tahun.", ln=True)
                        pdf.cell(200, 10, txt="- Status Pendaftaran Universitas: Kandidat tidak sedang terdaftar di universitas.", ln=True)
                        pdf.cell(200, 10, txt="- Jam Pelatihan: Kandidat memiliki lebih dari 30 jam pelatihan.", ln=True)
                    elif kelayakan == 50:
                        pdf.cell(200, 10, txt="Kriteria dan Tingkat Kelayakan:", ln=True)
                        pdf.cell(200, 10, txt="- Pengalaman Relevan: Kandidat memiliki pengalaman yang relevan.", ln=True)
                        pdf.cell(200, 10, txt="- Tingkat Pendidikan: Kandidat memiliki gelar Sarjana atau Magister.", ln=True)
                        pdf.cell(200, 10, txt="- Disiplin Utama: Kandidat berasal dari bidang STEM.", ln=True)
                        pdf.cell(200, 10, txt="- Pengalaman Kerja: Kandidat memiliki pengalaman kerja lebih dari 1 tahun.", ln=True)
                        pdf.cell(200, 10, txt="- Status Pendaftaran Universitas: Kandidat tidak sedang terdaftar di universitas.", ln=True)
                    else:
                        pdf.cell(200, 10, txt="Kandidat tidak memenuhi salah satu atau lebih dari kriteria di atas.", ln=True)

                    if kelayakan >= 70:
                        pdf.set_font("Arial", size=12, style="B")
                        pdf.set_text_color(0, 128, 0)  # Green color
                        pdf.cell(200, 10, txt="Kandidat diterima.", ln=True)
                    else:
                        pdf.set_font("Arial", size=12, style="B")
                        pdf.set_text_color(255, 0, 0)  # Red color
                        pdf.cell(200, 10, txt="Kandidat ditolak.", ln=True)

                    pdf.set_font("Arial", size=12)
                    pdf.set_text_color(0, 0, 0)  # Black color

                    pdf_output = pdf.output(dest="S").encode("latin-1")

                    # Tombol download PDF
                    st.download_button(
                        label="Download File",
                        data=pdf_output,
                        file_name=f"hasil_prediksi_{enrollee_id}.pdf",
                        mime="application/pdf"
                    )
    elif navigation == "Laporan Keanekaragaman":
        st.write("Laporan Keanekaragaman:")

        # Load data
        data = load_data()

        # Membuat laporan keanekaragaman
        st.write("Distribusi Gender:")
        st.write(data["gender"].value_counts())

        st.write("Distribusi Tingkat Pendidikan:")
        st.write(data["education_level"].value_counts())

        st.write("Distribusi Disiplin Utama:")
        st.write(data["major_discipline"].value_counts())

        st.write("Distribusi Pengalaman Kerja:")
        st.write(data["experience"].value_counts())

        st.write("Distribusi Status Pendaftaran Universitas:")
        st.write(data["enrolled_university"].value_counts())

        st.write("Distribusi Jam Pelatihan:")
        st.write(data["training_hours"].value_counts())

        st.write("Distribusi Durasi Pekerjaan Terakhir:")
        st.write(data["last_new_job"].value_counts())

        # Membuat laporan keanekaragaman berdampingan
        cols = st.beta_columns(2)

        with cols[0]:
            st.write("Distribusi Gender:")
            st.write(data["gender"].value_counts())

        with cols[1]:
            st.write("Distribusi Tingkat Pendidikan:")
            st.write(data["education_level"].value_counts())

        with cols[0]:
            st.write("Distribusi Disiplin Utama:")
            st.write(data["major_discipline"].value_counts())

        with cols[1]:
            st.write("Distribusi Pengalaman Kerja:")
            st.write(data["experience"].value_counts())

        with cols[0]:
            st.write("Distribusi Status Pendaftaran Universitas:")
            st.write(data["enrolled_university"].value_counts())

        with cols[1]:
            st.write("Distribusi Jam Pelatihan:")
            st.write(data["training_hours"].value_counts())

        with cols[0]:
            st.write("Distribusi Durasi Pekerjaan Terakhir:")
            st.write(data["last_new_job"].value_counts())

if __name__ == "__main__":
    main()
