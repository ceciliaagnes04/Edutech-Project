import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")
st.title("Employee Attrition Dashboard")

df = pd.read_csv("employee_data.csv")
df['Attrition'] = df['Attrition'].map({1.0: "Yes", 0.0: "No"})

st.sidebar.header("Filter Data")
departments = st.sidebar.multiselect(
    "Department", options=df["Department"].dropna().unique(), default=df["Department"].dropna().unique()
)
genders = st.sidebar.multiselect(
    "Gender", options=df["Gender"].dropna().unique(), default=df["Gender"].dropna().unique()
)
df_filtered = df[df["Department"].isin(departments) & df["Gender"].isin(genders)]

st.header("Data Preview")
st.dataframe(df_filtered.head())

st.header("Attrition Count")
fig1, ax1 = plt.subplots()
sns.countplot(x="Attrition", data=df_filtered, order=["No", "Yes"], ax=ax1)
ax1.set_title("Distribusi Attrition")
st.pyplot(fig1)

st.header("Attrition by Department")
fig2, ax2 = plt.subplots(figsize=(8,4))
sns.countplot(x="Department", hue="Attrition", data=df_filtered, ax=ax2)
plt.xticks(rotation=30)
ax2.set_title("Attrition per Department")
st.pyplot(fig2)

st.header("Attrition by Age")
fig3, ax3 = plt.subplots()
sns.histplot(data=df_filtered, x="Age", hue="Attrition", multiple="stack", bins=15, ax=ax3)
ax3.set_title("Distribusi Usia Berdasarkan Attrition")
st.pyplot(fig3)

st.header("Attrition by Monthly Income")
fig4, ax4 = plt.subplots()
sns.boxplot(x="Attrition", y="MonthlyIncome", data=df_filtered, ax=ax4)
ax4.set_title("Pendapatan Bulanan Berdasarkan Attrition")
st.pyplot(fig4)

st.markdown("### Gunakan sidebar untuk memfilter data berdasarkan Departemen dan Gender.")

# FEATURE IMPORTANCE
st.header("Feature Importance (Top 10)")
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('num_cols.pkl', 'rb') as f:
    num_cols = pickle.load(f)
with open('cat_cols.pkl', 'rb') as f:
    cat_cols = pickle.load(f)
feature_names = num_cols + cat_cols
importances = model.feature_importances_
feat_importances = pd.Series(importances, index=feature_names)
top_feat = feat_importances.nlargest(10)
fig5, ax5 = plt.subplots()
top_feat.plot(kind='barh', ax=ax5, color='teal')
ax5.set_title('Top 10 Feature Importance')
st.pyplot(fig5)

# PREDIKSI DATA BARU
st.header("Prediksi Attrition Karyawan Baru")
with st.form("prediction_form"):
    Age = st.number_input("Age", min_value=18, max_value=60, value=30)
    BusinessTravel = st.selectbox("BusinessTravel", df["BusinessTravel"].unique())
    DailyRate = st.number_input("DailyRate", min_value=0, value=800)
    Department = st.selectbox("Department", df["Department"].unique())
    DistanceFromHome = st.number_input("DistanceFromHome", min_value=0, value=5)
    Education = st.selectbox("Education", sorted(df["Education"].unique()))
    EducationField = st.selectbox("EducationField", df["EducationField"].unique())
    EnvironmentSatisfaction = st.selectbox("EnvironmentSatisfaction", sorted(df["EnvironmentSatisfaction"].unique()))
    Gender = st.selectbox("Gender", df["Gender"].unique())
    HourlyRate = st.number_input("HourlyRate", min_value=0, value=60)
    JobInvolvement = st.selectbox("JobInvolvement", sorted(df["JobInvolvement"].unique()))
    JobLevel = st.selectbox("JobLevel", sorted(df["JobLevel"].unique()))
    JobRole = st.selectbox("JobRole", df["JobRole"].unique())
    JobSatisfaction = st.selectbox("JobSatisfaction", sorted(df["JobSatisfaction"].unique()))
    MaritalStatus = st.selectbox("MaritalStatus", df["MaritalStatus"].unique())
    MonthlyIncome = st.number_input("MonthlyIncome", min_value=0, value=6000)
    MonthlyRate = st.number_input("MonthlyRate", min_value=0, value=15000)
    NumCompaniesWorked = st.number_input("NumCompaniesWorked", min_value=0, value=2)
    OverTime = st.selectbox("OverTime", df["OverTime"].unique())
    PercentSalaryHike = st.number_input("PercentSalaryHike", min_value=0, value=13)
    PerformanceRating = st.selectbox("PerformanceRating", [3, 4])
    RelationshipSatisfaction = st.selectbox("RelationshipSatisfaction", sorted(df["RelationshipSatisfaction"].unique()))
    StockOptionLevel = st.selectbox("StockOptionLevel", sorted(df["StockOptionLevel"].unique()))
    TotalWorkingYears = st.number_input("TotalWorkingYears", min_value=0, value=8)
    TrainingTimesLastYear = st.number_input("TrainingTimesLastYear", min_value=0, value=2)
    WorkLifeBalance = st.selectbox("WorkLifeBalance", sorted(df["WorkLifeBalance"].unique()))
    YearsAtCompany = st.number_input("YearsAtCompany", min_value=0, value=5)
    YearsInCurrentRole = st.number_input("YearsInCurrentRole", min_value=0, value=2)
    YearsSinceLastPromotion = st.number_input("YearsSinceLastPromotion", min_value=0, value=1)
    YearsWithCurrManager = st.number_input("YearsWithCurrManager", min_value=0, value=3)
    submitted = st.form_submit_button("Prediksi Attrition")

    if submitted:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        data_new = {
            'Age': [Age],
            'BusinessTravel': [BusinessTravel],
            'DailyRate': [DailyRate],
            'Department': [Department],
            'DistanceFromHome': [DistanceFromHome],
            'Education': [Education],
            'EducationField': [EducationField],
            'EnvironmentSatisfaction': [EnvironmentSatisfaction],
            'Gender': [Gender],
            'HourlyRate': [HourlyRate],
            'JobInvolvement': [JobInvolvement],
            'JobLevel': [JobLevel],
            'JobRole': [JobRole],
            'JobSatisfaction': [JobSatisfaction],
            'MaritalStatus': [MaritalStatus],
            'MonthlyIncome': [MonthlyIncome],
            'MonthlyRate': [MonthlyRate],
            'NumCompaniesWorked': [NumCompaniesWorked],
            'OverTime': [OverTime],
            'PercentSalaryHike': [PercentSalaryHike],
            'PerformanceRating': [PerformanceRating],
            'RelationshipSatisfaction': [RelationshipSatisfaction],
            'StockOptionLevel': [StockOptionLevel],
            'TotalWorkingYears': [TotalWorkingYears],
            'TrainingTimesLastYear': [TrainingTimesLastYear],
            'WorkLifeBalance': [WorkLifeBalance],
            'YearsAtCompany': [YearsAtCompany],
            'YearsInCurrentRole': [YearsInCurrentRole],
            'YearsSinceLastPromotion': [YearsSinceLastPromotion],
            'YearsWithCurrManager': [YearsWithCurrManager]
        }
        df_pred = pd.DataFrame(data_new)
        for col in cat_cols:
            df_pred[col] = encoders[col].transform(df_pred[col])
        df_pred[num_cols] = scaler.transform(df_pred[num_cols])
        pred = model.predict(df_pred)[0]
        pred_label = "Keluar (Attrition)" if pred == 1 else "Tidak Keluar"
        st.success(f"Hasil Prediksi: {pred_label}")
