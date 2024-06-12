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

    # Menyimpan file PDF
    pdf_file = "Laporan_Keberagaman.pdf"
    pdf.output(pdf_file)

    return pdf_file

# Fungsi untuk mengubah file menjadi format base64
def file_to_base64(file_path):
    with open(file_path, "rb") as file:
        file_content = file.read()
    return base64.b64encode(file_content).decode()

# Langkah 8: Membuat sistem prediksi
def predict_candidate(model, candidate_data):
    prediction = model.predict(candidate_data)
    probability = model.predict_proba(candidate_data)

    return prediction[0], probability.max()

# Ekspor hasil prediksi ke PDF dengan warna teks
def export_prediction_to_pdf(prediction_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Menambahkan konten ke PDF
    pdf.cell(200, 10, txt="Hasil Prediksi Kandidat", ln=True, align='C')
    for key, value in prediction_data.items():
        # Tentukan warna teks berdasarkan keterangan
        if key == "Keterangan":
            if value == "Kandidat Ditolak":
                pdf.set_text_color(255, 0, 0)  # Merah untuk kandidat ditolak
            elif value == "Kandidat Diterima":
                pdf.set_text_color(0, 255, 0)  # Hijau untuk kandidat diterima
        else:
            pdf.set_text_color(0, 0, 0)  # Hitam untuk teks lainnya

        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    pdf_file = "Hasil_Prediksi.pdf"
    pdf.output(pdf_file)

    return pdf_file

# Streamlit app
def main():
    st.title("Aplikasi Analitik dan Prediksi Kandidat")

    add_custom_css()

    menu = ["Upload Dataset", "Preprocessing", "Train Model", "Evaluate Model", "Diversity Report", "Predict Candidate"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Upload Dataset":
        st.subheader("Upload Dataset")
        data = load_data()

    elif choice == "Preprocessing":
        st.subheader("Preprocessing Data")
        data = load_data()
        processed_data = preprocess_data(data)
        st.write("Data setelah preprocessing:")
        st.write(processed_data.head(14))  # Show the first 14 rows after preprocessing

    elif choice == "Train Model":
        st.subheader("Train Model")
        data = load_data()
        processed_data = preprocess_data(data)
        X_train, X_test, y_train, y_test = split_data(processed_data)
        model = train_model(X_train, y_train)
        st.success("Model berhasil dilatih!")

    elif choice == "Evaluate Model":
        st.subheader("Evaluate Model")
        data = load_data()
        processed_data = preprocess_data(data)
        X_train, X_test, y_train, y_test = split_data(processed_data)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)

    elif choice == "Diversity Report":
        st.subheader("Diversity Report")
        data = load_data()
        gender_counts, education_counts, experience_counts, company_type_counts, company_size_counts, discipline_counts, last_new_job_counts, figures = generate_diversity_report(data)
        pdf_file = export_report_to_pdf(data, gender_counts, education_counts, experience_counts, company_type_counts, company_size_counts, discipline_counts, last_new_job_counts, figures)
        st.success(f"Laporan berhasil diekspor ke {pdf_file}")
        pdf_base64 = file_to_base64(pdf_file)
        href = f'<a href="data:application/octet-stream;base64,{pdf_base64}" download="{pdf_file}">Download Laporan PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

    elif choice == "Predict Candidate":
        st.subheader("Predict Candidate")
        data = load_data()
        processed_data = preprocess_data(data)
        model = train_model(*split_data(processed_data)[:2])

        st.write("Masukkan data kandidat:")
        experience = st.slider("Experience (years)", min_value=0, max_value=25, value=0)
        relevant_experience = st.selectbox("Relevant Experience", options=["No relevant experience", "Has relevant experience"])
        enrolled_university = st.selectbox("Enrolled University", options=["no_enrollment", "Full time course", "Part time course"])
        education_level = st.selectbox("Education Level", options=["Primary School", "High School", "Graduate", "Masters", "Phd"])
        major_discipline = st.selectbox("Major Discipline", options=["STEM", "Business Degree", "Arts", "Humanities", "No Major", "Other"])
        company_size = st.selectbox("Company Size", options=["<10", "10-49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"])
        company_type = st.selectbox("Company Type", options=["NGO", "Public Sector", "Pvt Ltd", "Early Stage Startup", "Funded Startup", "Other"])
        last_new_job = st.selectbox("Last New Job", options=["never", "1", "2", "3", "4", "5+"])

        candidate_data = pd.DataFrame({
            "experience": [experience],
            "relevent_experience": [relevant_experience],
            "enrolled_university": [enrolled_university],
            "education_level": [education_level],
            "major_discipline": [major_discipline],
            "company_size": [company_size],
            "company_type": [company_type],
            "last_new_job": [last_new_job]
        })

        label_encoder = LabelEncoder()
        categorical_cols = ['relevent_experience', 'enrolled_university', 'education_level', 
                            'major_discipline', 'company_size', 'company_type', 'last_new_job']
        for col in categorical_cols:
            candidate_data[col] = label_encoder.fit_transform(candidate_data[col])

        prediction, probability = predict_candidate(model, candidate_data)

        prediction_data = {
            "Experience": experience,
            "Relevant Experience": relevant_experience,
            "Enrolled University": enrolled_university,
            "Education Level": education_level,
            "Major Discipline": major_discipline,
            "Company Size": company_size,
            "Company Type": company_type,
            "Last New Job": last_new_job,
            "Prediksi": prediction,
            "Probabilitas": f"{probability * 100:.2f}%",
            "Keterangan": "Kandidat Diterima" if prediction == 1 else "Kandidat Ditolak"
        }

        st.write("Hasil Prediksi:")
        for key, value in prediction_data.items():
            st.write(f"{key}: {value}")

        pdf_file = export_prediction_to_pdf(prediction_data)
        st.success(f"Hasil prediksi berhasil diekspor ke {pdf_file}")
        pdf_base64 = file_to_base64(pdf_file)
        href = f'<a href="data:application/octet-stream;base64,{pdf_base64}" download="{pdf_file}">Download Hasil Prediksi PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
