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
def load_data(uploaded_file=None):
    if uploaded_file is None:
        data = pd.read_csv("dataset_recruitment.csv")
    else:
        data = pd.read_csv(uploaded_file)
    
    st.write("Dataset:")
    st.write(data.head(14))  # Show the first 14 rows
    st.write(f"Jumlah data pada dataset: {len(data)}")  # Menambahkan informasi jumlah data
    return data

# Langkah 3: Standarisasi data
def preprocess_data(data):
    # Ubah nilai "<1" menjadi 0 dan nilai ">20" menjadi 25
    if 'experience' in data.columns:
        data['experience'] = data['experience'].apply(lambda x: 0 if x == '<1' else (25 if x == '>20' else int(x)))

    # Mengonversi fitur kategorikal ke dalam representasi numerik menggunakan label encoding
    label_encoder = LabelEncoder()
    categorical_cols = ['relevent_experience', 'enrolled_university', 'education_level', 
                        'major_discipline', 'company_size', 'company_type', 'last_new_job']
    for col in categorical_cols:
        if col in data.columns:
            data[col] = label_encoder.fit_transform(data[col])

    return data

# Langkah 4: Split data train dan test
def split_data(data):
    if 'gender' not in data.columns or 'city' not in data.columns:
        st.error("Dataset tidak memiliki kolom 'gender' atau 'city' yang diperlukan untuk proses ini.")
        return None, None, None, None
    
    X = data.drop(columns=["gender", "city"])  # Hapus fitur "City"
    y = data["gender"]

    test_size = 0.2  # Tentukan ukuran test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Menghitung persentase data training dan testing
    train_percentage = len(X_train) / len(data) * 100
    test_percentage = len(X_test) / len(data) * 100

    st.write(f"Persentase data training: {train_percentage:.2f}%")
    st.write(f"Persentase data testing: {test_percentage:.2f}%")

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
    if 'gender' in data.columns:
        st.write("Jumlah pelamar berdasarkan gender:")
        gender_counts = data['gender'].value_counts()
        fig1, ax1 = plt.subplots()
        sns.barplot(x=gender_counts.index, y=gender_counts.values, ax=ax1)
        ax1.set_title("Jumlah pelamar berdasarkan gender")
        ax1.set_xlabel("Gender")
        ax1.set_ylabel("Jumlah")
        st.pyplot(fig1)
        figures.append(fig1)
    else:
        gender_counts = None

    # Plotting education level counts
    if 'education_level' in data.columns:
        st.write("Jumlah pelamar berdasarkan tingkat pendidikan:")
        education_counts = data['education_level'].value_counts()
        fig2, ax2 = plt.subplots()
        sns.barplot(x=education_counts.index, y=education_counts.values, ax=ax2)
        ax2.set_title("Jumlah pelamar berdasarkan tingkat pendidikan")
        ax2.set_xlabel("Tingkat Pendidikan")
        ax2.set_ylabel("Jumlah")
        st.pyplot(fig2)
        figures.append(fig2)
    else:
        education_counts = None

    # Plotting relevant experience counts
    if 'relevent_experience' in data.columns:
        st.write("Jumlah pelamar berdasarkan pengalaman relevan:")
        experience_counts = data['relevent_experience'].value_counts()
        fig3, ax3 = plt.subplots()
        sns.barplot(x=experience_counts.index, y=experience_counts.values, ax=ax3)
        ax3.set_title("Jumlah pelamar berdasarkan pengalaman relevan")
        ax3.set_xlabel("Pengalaman Relevan")
        ax3.set_ylabel("Jumlah")
        st.pyplot(fig3)
        figures.append(fig3)
    else:
        experience_counts = None

    # Plotting company type counts
    if 'company_type' in data.columns:
        st.write("Jumlah pelamar berdasarkan perusahaan sebelumnya:")
        company_type_counts = data['company_type'].value_counts()
        fig4, ax4 = plt.subplots()
        sns.barplot(x=company_type_counts.index, y=company_type_counts.values, ax=ax4)
        ax4.set_title("Jumlah pelamar berdasarkan perusahaan sebelumnya")
        ax4.set_xlabel("Tipe Perusahaan")
        ax4.set_ylabel("Jumlah")
        st.pyplot(fig4)
        figures.append(fig4)
    else:
        company_type_counts = None

    # Plotting company size counts
    if 'company_size' in data.columns:
        st.write("Jumlah pelamar berdasarkan ukuran perusahaan sebelumnya:")
        company_size_counts = data['company_size'].value_counts()
        fig5, ax5 = plt.subplots()
        sns.barplot(x=company_size_counts.index, y=company_size_counts.values, ax=ax5)
        ax5.set_title("Jumlah pelamar berdasarkan ukuran perusahaan sebelumnya")
        ax5.set_xlabel("Ukuran Perusahaan")
        ax5.set_ylabel("Jumlah")
        st.pyplot(fig5)
        figures.append(fig5)
    else:
        company_size_counts = None

    # Plotting major discipline counts
    if 'major_discipline' in data.columns:
        st.write("Jumlah pelamar berdasarkan disiplin ilmu:")
        discipline_counts = data['major_discipline'].value_counts()
        fig6, ax6 = plt.subplots()
        sns.barplot(x=discipline_counts.index, y=discipline_counts.values, ax=ax6)
        ax6.set_title("Jumlah pelamar berdasarkan disiplin ilmu")
        ax6.set_xlabel("Disiplin Ilmu")
        ax6.set_ylabel("Jumlah")
        st.pyplot(fig6)
        figures.append(fig6)
    else:
        discipline_counts = None

    # Plotting last new job counts
    if 'last_new_job' in data.columns:
        st.write("Jumlah pelamar berdasarkan waktu terakhir kali pindah kerja:")
        last_new_job_counts = data['last_new_job'].value_counts()
        fig7, ax7 = plt.subplots()
        sns.barplot(x=last_new_job_counts.index, y=last_new_job_counts.values, ax=ax7)
        ax7.set_title("Jumlah pelamar berdasarkan waktu terakhir kali pindah kerja")
        ax7.set_xlabel("Waktu Terakhir Pindah Kerja")
        ax7.set_ylabel("Jumlah")
        st.pyplot(fig7)
        figures.append(fig7)
    else:
        last_new_job_counts = None

    return figures

# Langkah 8: Fungsi untuk mengubah gambar menjadi base64
def fig_to_base64(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# Langkah 8: Generate PDF report
def generate_pdf_report(accuracy, figures):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Laporan Model dan Analisis Keberagaman", ln=True, align='C')
    pdf.ln(10)
    
    pdf.cell(200, 10, txt=f"Akurasi Model: {accuracy * 100:.2f}%", ln=True)
    pdf.ln(10)
    
    for i, fig in enumerate(figures):
        img_base64 = fig_to_base64(fig)
        img_data = base64.b64decode(img_base64)
        img_filename = f"figure_{i}.png"
        with open(img_filename, "wb") as img_file:
            img_file.write(img_data)
        pdf.image(img_filename, x=10, y=None, w=190)
        pdf.ln(10)
    
    pdf_filename = "laporan_analisis.pdf"
    pdf.output(pdf_filename)
    
    with open(pdf_filename, "rb") as file:
        pdf_data = file.read()
    
    b64_pdf = base64.b64encode(pdf_data).decode()
    pdf_display = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="laporan_analisis.pdf">Download Laporan PDF</a>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit app
def main():
    st.title("Analisis Model dan Keberagaman")

    add_custom_css()  # Memanggil fungsi CSS
    
    # Opsi untuk upload dataset
    uploaded_file = st.file_uploader("Upload dataset (CSV file)", type=["csv"])
    
    data = load_data(uploaded_file)
    if data is not None:
        data = preprocess_data(data)
        X_train, X_test, y_train, y_test = split_data(data)
        
        if X_train is not None and X_test is not None:
            model = train_model(X_train, y_train)
            accuracy = evaluate_model(model, X_test, y_test)
            figures = generate_diversity_report(data)
            
            # Generate PDF report button
            if st.button("Generate PDF Report"):
                generate_pdf_report(accuracy, figures)
                st.success("Laporan PDF telah berhasil dibuat.")

if __name__ == "__main__":
    main()
