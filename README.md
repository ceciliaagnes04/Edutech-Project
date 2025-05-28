# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

**Latar Belakang Bisnis:**  
Perusahaan Edutech adalah perusahaan teknologi pendidikan yang menghadapi tantangan dalam mempertahankan karyawan terbaiknya. Tingginya tingkat attrition (keluar-masuk karyawan) berdampak pada stabilitas tim, biaya rekrutmen, dan kualitas layanan. Diperlukan analisis mendalam dan sistem prediksi untuk membantu HR mengambil keputusan berbasis data.

### Permasalahan Bisnis

- Bagaimana memprediksi karyawan yang berisiko tinggi untuk keluar dari perusahaan?
- Apa saja faktor utama yang menyebabkan attrition di perusahaan Edutech?
- Bagaimana perusahaan dapat menurunkan tingkat attrition dan meningkatkan retensi karyawan?

### Cakupan Proyek

- Analisis data karyawan untuk memahami pola attrition.
- Pembuatan model prediksi attrition berbasis machine learning.
- Pembuatan business dashboard interaktif untuk visualisasi data, feature importance, dan prediksi.
- Penyusunan rekomendasi strategis untuk HR berdasarkan hasil analisis dan model.

## Persiapan

### 1. Sumber Data
Dataset yang digunakan adalah **IBM HR Analytics Employee Attrition & Performance** yang berisi data demografi, pekerjaan, dan status attrition karyawan Edutech.  
Link dataset: [Kaggle - IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

### 2. Setup Environment

#### a. Dengan Anaconda
conda create --name main-ds python=3.9
conda activate main-ds
pip install -r requirements.txt

#### b. Dengan Pipenv
pip install pipenv
pipenv install
pipenv shell
pip install -r requirements.txt

### 3. Menjalankan File Prediksi (Opsional)
- Untuk prediksi satu data baru:
python prediction.py
- Untuk menjalankan dashboard:
streamlit run employee_dashboard.py

## Business Dashboard

Dashboard dibuat menggunakan **Streamlit** dan menyediakan fitur berikut:

- **Visualisasi utama**: Distribusi attrition secara keseluruhan, breakdown berdasarkan departemen, usia, dan pendapatan bulanan.
- **Filter interaktif**: Sidebar untuk filter berdasarkan departemen dan gender.
- **Feature Importance**: Menampilkan 10 fitur terpenting dari model Random Forest yang mempengaruhi risiko attrition.
- **Form Prediksi Interaktif**: User dapat mengisi data karyawan baru dan mendapatkan hasil prediksi apakah karyawan tersebut berisiko keluar atau tidak.

**Link Dashboard (Deploy Streamlit):**  
ceciliaagnes04/Edutech-Project

## EDA (Exploratory Data Analysis)

- **EDA Univariate:**  
Distribusi fitur numerik dan kategorikal: usia, pendapatan, attrition, dsb.
- **EDA Multivariate:**  
Korelasi antar fitur, hubungan attrition dengan fitur lain seperti OverTime, JobRole, dsb.
- **EDA Numerikal:**  
Statistik deskriptif pada fitur numerik.
- **EDA Kategorikal:**  
Proporsi kategori pada Department, Gender, MaritalStatus, dsb.
- **Visualisasi:**  
Grafik countplot, boxplot, histogram, heatmap korelasi.

## Contoh Hasil Prediksi

### 1. Hasil Prediksi: **Tidak Keluar**

**Input:**
- Age: 40
- BusinessTravel: Travel_Rarely
- DailyRate: 1200
- Department: Research & Development
- DistanceFromHome: 3
- Education: 3
- EducationField: Medical
- EnvironmentSatisfaction: 3
- Gender: Male
- HourlyRate: 80
- JobInvolvement: 3
- JobLevel: 2
- JobRole: Research Scientist
- JobSatisfaction: 4
- MaritalStatus: Married
- MonthlyIncome: 8000
- MonthlyRate: 15000
- NumCompaniesWorked: 1
- OverTime: No
- PercentSalaryHike: 15
- PerformanceRating: 3
- RelationshipSatisfaction: 3
- StockOptionLevel: 1
- TotalWorkingYears: 15
- TrainingTimesLastYear: 3
- WorkLifeBalance: 3
- YearsAtCompany: 10
- YearsInCurrentRole: 5
- YearsSinceLastPromotion: 2
- YearsWithCurrManager: 5

**Output:**  
`Hasil Prediksi: Tidak Keluar`

### 2. Hasil Prediksi: **Keluar**

**Input:**
- Age: 23
- BusinessTravel: Travel_Frequently
- DailyRate: 350
- Department: Sales
- DistanceFromHome: 20
- Education: 1
- EducationField: Marketing
- EnvironmentSatisfaction: 1
- Gender: Female
- HourlyRate: 35
- JobInvolvement: 1
- JobLevel: 1
- JobRole: Sales Representative
- JobSatisfaction: 1
- MaritalStatus: Single
- MonthlyIncome: 2000
- MonthlyRate: 4000
- NumCompaniesWorked: 5
- OverTime: Yes
- PercentSalaryHike: 11
- PerformanceRating: 3
- RelationshipSatisfaction: 1
- StockOptionLevel: 0
- TotalWorkingYears: 1
- TrainingTimesLastYear: 0
- WorkLifeBalance: 1
- YearsAtCompany: 1
- YearsInCurrentRole: 0
- YearsSinceLastPromotion: 0
- YearsWithCurrManager: 0

**Output:**  
`Hasil Prediksi: Keluar`

## Conclusion

Model Random Forest yang dibangun mampu memprediksi attrition dengan akurasi yang baik.  
Faktor-faktor utama yang mempengaruhi attrition antara lain: lembur (OverTime), peran kerja (JobRole), dan kepuasan kerja.

Dashboard interaktif ini sangat membantu HR untuk melakukan monitoring, analisis, dan prediksi secara real time, serta mengambil keputusan berbasis data.

### Rekomendasi Action Items

- Tingkatkan program work-life balance dan kurangi lembur berlebihan.
- Lakukan survei kepuasan kerja secara berkala dan tindak lanjuti hasilnya.
- Berikan pelatihan dan pengembangan karir untuk meningkatkan engagement karyawan.
- Fokus pada departemen dengan tingkat attrition tinggi untuk intervensi lebih awal.
