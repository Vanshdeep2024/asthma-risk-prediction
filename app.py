import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="Asthma Risk Prediction",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 Asthma Risk Prediction")
st.write("Machine Learning Based Asthma Risk Assessment System")

st.info(
    "⚠️ This application is an educational machine-learning project "
    "and is not a medical diagnosis."
)


# =====================================================
# DATASET UPLOAD
# =====================================================

st.sidebar.header("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Asthma CSV Dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload the Asthma CSV dataset.")
    st.stop()


# =====================================================
# LOAD DATA
# =====================================================

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error("Could not read the CSV file.")
    st.write(e)
    st.stop()

df.columns = df.columns.str.strip()


# =====================================================
# DATASET OVERVIEW
# =====================================================

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", df.shape[0])
col2.metric("Total Features", df.shape[1])
col3.metric(
    "Missing Values",
    int(df.isnull().sum().sum())
)

with st.expander("🔍 View Dataset"):
    st.dataframe(
        df.head(15),
        use_container_width=True
    )


# =====================================================
# TARGET
# =====================================================

target = "Diagnosis"

if target not in df.columns:
    st.error("Diagnosis column was not found.")
    st.write("Available columns:")
    st.write(list(df.columns))
    st.stop()


df[target] = pd.to_numeric(
    df[target],
    errors="coerce"
)

df = df.dropna(
    subset=[target]
)


# =====================================================
# FEATURES
# =====================================================

possible_features = [
    "Age",
    "Gender",
    "Ethnicity",
    "EducationLevel",
    "BMI",
    "Smoking",
    "PhysicalActivity",
    "DietQuality",
    "SleepQuality",
    "PollutionExposure",
    "PollenExposure",
    "DustExposure",
    "PetAllergy",
    "FamilyHistoryAsthma",
    "HistoryOfAllergies",
    "Eczema",
    "HayFever",
    "GastroesophagealReflux",
    "LungFunctionFEV1",
    "LungFunctionFVC",
    "Wheezing",
    "ShortnessOfBreath",
    "ChestTightness",
    "Coughing",
    "NighttimeSymptoms",
    "ExerciseInduced",
    "BronchodilatorResponse",
    "OxygenSaturation",
    "Severity"
]

features = []

for feature in possible_features:
    if feature in df.columns:
        features.append(feature)


if len(features) < 2:
    st.error("Not enough usable features were found.")
    st.write(list(df.columns))
    st.stop()


# =====================================================
# CONVERT FEATURES
# =====================================================

for feature in features:
    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )


X = df[features]
y = df[target]


if y.nunique() < 2:
    st.error(
        "Dataset contains only one diagnosis category."
    )
    st.stop()


# =====================================================
# DIAGNOSIS DISTRIBUTION
# =====================================================

st.divider()

st.header("🎯 Asthma Diagnosis Distribution")

no_asthma = int((y == 0).sum())
asthma = int((y == 1).sum())

col1, col2 = st.columns(2)

col1.metric(
    "No Asthma Records",
    no_asthma
)

col2.metric(
    "Asthma Records",
    asthma
)


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

try:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

except ValueError:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )


# =====================================================
# MODELS
# =====================================================

models = {

    "Logistic Regression": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]),

    "Decision Tree": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            DecisionTreeClassifier(
                max_depth=8,
                random_state=42
            )
        )
    ]),

    "Random Forest": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]),

    "KNN": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ]),

    "SVM": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            SVC(
                probability=True,
                random_state=42
            )
        )
    ])
}


# =====================================================
# TRAIN ALL MODELS
# =====================================================

results = []
trained_models = {}

with st.spinner(
    "🤖 Training and comparing multiple ML models..."
):

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            prediction
        )

        precision = precision_score(
            y_test,
            prediction,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            prediction,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            prediction,
            zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        trained_models[name] = model


# =====================================================
# MODEL COMPARISON
# =====================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "F1 Score",
    ascending=False
)

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]


# =====================================================
# MODEL COMPARISON TABLE
# =====================================================

st.divider()

st.header(
    "🤖 Machine Learning Model Comparison"
)

display_df = results_df.copy()

for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]:

    display_df[column] = (
        display_df[column] * 100
    ).round(2).astype(str) + "%"


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# =====================================================
# BEST MODEL
# =====================================================

st.success(
    "🏆 Best Model: " + best_model_name
)

best_f1 = results_df.iloc[0]["F1 Score"]

st.metric(
    "Best Model F1 Score",
    f"{best_f1 * 100:.2f}%"
)


# =====================================================
# MODEL COMPARISON CHART
# =====================================================

st.subheader(
    "📊 Model Performance Comparison"
)

chart_df = results_df.set_index(
    "Model"
)[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
]

st.bar_chart(
    chart_df
)


# =====================================================
# BEST MODEL PREDICTION
# =====================================================

best_prediction = best_model.predict(
    X_test
)


# =====================================================
# CONFUSION MATRIX
# =====================================================

st.divider()

st.header(
    "🔲 Best Model Confusion Matrix"
)

cm = confusion_matrix(
    y_test,
    best_prediction,
    labels=[0, 1]
)

fig, ax = plt.subplots()

ax.imshow(cm)

ax.set_title(
    best_model_name +
    " - Confusion Matrix"
)

ax.set_xlabel(
    "Predicted Result"
)

ax.set_ylabel(
    "Actual Result"
)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(
    [
        "No Asthma",
        "Asthma"
    ]
)

ax.set_yticklabels(
    [
        "No Asthma",
        "Asthma"
    ]
)

for i in range(2):

    for j in range(2):

        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

st.pyplot(fig)

plt.close(fig)


# =====================================================
# SIMPLE FEATURE NAMES
# =====================================================

simple_names = {

    "Age": "👤 Age",
    "Gender": "👥 Gender",
    "Ethnicity": "🌍 Ethnicity",
    "EducationLevel": "🎓 Education Level",
    "BMI": "⚖️ Body Mass Index",
    "Smoking": "🚬 Smoking",
    "PhysicalActivity": "🏃 Physical Activity",
    "DietQuality": "🥗 Diet Quality",
    "SleepQuality": "😴 Sleep Quality",
    "PollutionExposure": "🌫️ Pollution Exposure",
    "PollenExposure": "🌼 Pollen Exposure",
    "DustExposure": "🌪️ Dust Exposure",
    "PetAllergy": "🐶 Pet Allergy",
    "FamilyHistoryAsthma": "👨‍👩‍👧 Family History of Asthma",
    "HistoryOfAllergies": "🤧 History of Allergies",
    "Eczema": "🩹 Eczema",
    "HayFever": "🌸 Hay Fever",
    "GastroesophagealReflux": "🍽️ Acid Reflux",
    "LungFunctionFEV1": "🫁 Lung Function",
    "LungFunctionFVC": "🫁 Lung Capacity",
    "Wheezing": "😮‍💨 Wheezing",
    "ShortnessOfBreath": "😮‍💨 Shortness of Breath",
    "ChestTightness": "🫁 Chest Tightness",
    "Coughing": "🤧 Coughing",
    "NighttimeSymptoms": "🌙 Nighttime Symptoms",
    "ExerciseInduced": "🏃 Exercise Related Symptoms",
    "BronchodilatorResponse": "💨 Response to Breathing Medicine",
    "OxygenSaturation": "🩸 Oxygen Level",
    "Severity": "📊 Symptom Severity"
}


# =====================================================
# INFORMATION USED
# =====================================================

st.divider()

st.header(
    "🔎 Information Used by AI"
)

for feature in features:

    st.write(
        simple_names.get(
            feature,
            feature
        )
    )


# =====================================================
# FEATURE IMPORTANCE
# =====================================================

st.divider()

st.header(
    "⭐ Important Factors"
)

try:

    final_model = best_model.named_steps["model"]

    if hasattr(
        final_model,
        "feature_importances_"
    ):

        importance_values = (
            final_model.feature_importances_
        )

    elif hasattr(
        final_model,
        "coef_"
    ):

        importance_values = np.abs(
            final_model.coef_[0]
        )

    else:

        importance_values = None


    if importance_values is not None:

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance_values
        })

        importance_df["Feature"] = (
            importance_df["Feature"]
            .map(simple_names)
        )

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False
        )

        st.bar_chart(
            importance_df.set_index(
                "Feature"
            ).head(10)
        )

        st.write(
            "The chart shows the factors that had the "
            "strongest influence on the selected model."
        )

    else:

        st.info(
            "Feature importance is not directly available "
            "for the selected model."
        )

except Exception:

    st.info(
        "Feature importance could not be calculated."
    )


# =====================================================
# PATIENT ASSESSMENT
# =====================================================

st.divider()

st.header(
    "🧑‍⚕️ Asthma Risk Assessment"
)

st.write(
    "Enter the person's information below."
)

values = {}


# AGE

if "Age" in features:

    values["Age"] = st.number_input(
        "👤 Age",
        min_value=1.0,
        max_value=120.0,
        value=30.0
    )


# GENDER

if "Gender" in features:

    gender = st.selectbox(
        "👥 Gender",
        [
            "Male",
            "Female"
        ]
    )

    values["Gender"] = (
        0 if gender == "Male" else 1
    )


# ETHNICITY

if "Ethnicity" in features:

    ethnicity = st.selectbox(
        "🌍 Ethnicity",
        [
            "Caucasian",
            "African American",
            "Asian",
            "Other"
        ]
    )

    ethnicity_values = {
        "Caucasian": 0,
        "African American": 1,
        "Asian": 2,
        "Other": 3
    }

    values["Ethnicity"] = (
        ethnicity_values[ethnicity]
    )


# EDUCATION

if "EducationLevel" in features:

    education = st.selectbox(
        "🎓 Education Level",
        [
            "High School",
            "Graduate",
            "Postgraduate",
            "PhD"
        ]
    )

    education_values = {
        "High School": 1,
        "Graduate": 2,
        "Postgraduate": 3,
        "PhD": 4
    }

    values["EducationLevel"] = (
        education_values[education]
    )


# BMI

if "BMI" in features:

    values["BMI"] = st.number_input(
        "⚖️ Body Mass Index",
        min_value=10.0,
        max_value=60.0,
        value=24.0
    )


# SMOKING

if "Smoking" in features:

    smoking = st.selectbox(
        "🚬 Smoking",
        [
            "No",
            "Yes"
        ]
    )

    values["Smoking"] = (
        1 if smoking == "Yes" else 0
    )


# PHYSICAL ACTIVITY

if "PhysicalActivity" in features:

    values["PhysicalActivity"] = st.number_input(
        "🏃 Physical Activity",
        min_value=0.0,
        max_value=20.0,
        value=5.0
    )


# DIET

if "DietQuality" in features:

    values["DietQuality"] = st.number_input(
        "🥗 Diet Quality",
        min_value=0.0,
        max_value=10.0,
        value=5.0
    )


# SLEEP

if "SleepQuality" in features:

    values["SleepQuality"] = st.number_input(
        "😴 Sleep Quality",
        min_value=0.0,
        max_value=10.0,
        value=6.0
    )


# EXPOSURES

for feature, label in [
    ("PollutionExposure", "🌫️ Pollution Exposure"),
    ("PollenExposure", "🌼 Pollen Exposure"),
    ("DustExposure", "🌪️ Dust Exposure")
]:

    if feature in features:

        values[feature] = st.number_input(
            label,
            min_value=0.0,
            max_value=10.0,
            value=5.0
        )


# YES / NO FEATURES

yes_no_features = [

    ("PetAllergy", "🐶 Pet Allergy"),

    (
        "FamilyHistoryAsthma",
        "👨‍👩‍👧 Family History of Asthma"
    ),

    (
        "HistoryOfAllergies",
        "🤧 History of Allergies"
    ),

    ("Eczema", "🩹 Eczema"),

    ("HayFever", "🌸 Hay Fever"),

    (
        "GastroesophagealReflux",
        "🍽️ Acid Reflux"
    ),

    ("Wheezing", "😮‍💨 Wheezing"),

    (
        "ShortnessOfBreath",
        "😮‍💨 Shortness of Breath"
    ),

    (
        "ChestTightness",
        "🫁 Chest Tightness"
    ),

    ("Coughing", "🤧 Coughing"),

    (
        "NighttimeSymptoms",
        "🌙 Nighttime Symptoms"
    ),

    (
        "ExerciseInduced",
        "🏃 Exercise Related Symptoms"
    ),

    (
        "BronchodilatorResponse",
        "💨 Response to Breathing Medicine"
    )
]


for feature, label in yes_no_features:

    if feature in features:

        answer = st.selectbox(
            label,
            [
                "No",
                "Yes"
            ]
        )

        values[feature] = (
            1 if answer == "Yes" else 0
        )


# LUNG FUNCTION

if "LungFunctionFEV1" in features:

    values["LungFunctionFEV1"] = st.number_input(
        "🫁 Lung Function Measurement",
        min_value=0.0,
        max_value=10.0,
        value=3.0
    )


if "LungFunctionFVC" in features:

    values["LungFunctionFVC"] = st.number_input(
        "🫁 Lung Capacity Measurement",
        min_value=0.0,
        max_value=10.0,
        value=4.0
    )


# OXYGEN

if "OxygenSaturation" in features:

    values["OxygenSaturation"] = st.number_input(
        "🩸 Oxygen Level (%)",
        min_value=70.0,
        max_value=100.0,
        value=97.0
    )


# SEVERITY

if "Severity" in features:

    severity = st.selectbox(
        "📊 Symptom Severity",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    severity_values = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    values["Severity"] = (
        severity_values[severity]
    )


# =====================================================
# PREDICTION
# =====================================================

if st.button(
    "🔮 Check Asthma Risk",
    use_container_width=True
):

    patient = pd.DataFrame(
        [values]
    )

    patient = patient.reindex(
        columns=features
    )

    prediction = best_model.predict(
        patient
    )[0]

    try:

        probability = (
            best_model.predict_proba(
                patient
            )[0][1]
        )

    except Exception:

        probability = None


    st.divider()

    st.header(
        "🎯 Asthma Risk Result"
    )


    if probability is not None:

        risk = probability * 100

        if risk < 30:

            level = "Lower"

            st.success(
                "🟢 Lower Asthma Risk"
            )

        elif risk < 60:

            level = "Moderate"

            st.warning(
                "🟡 Moderate Asthma Risk"
            )

        else:

            level = "Higher"

            st.error(
                "🔴 Higher Asthma Risk"
            )


        st.metric(
            "Estimated Asthma Risk",
            f"{risk:.2f}%"
        )

        st.progress(
            float(probability)
        )

    else:

        if prediction == 1:

            level = "Higher"

            st.error(
                "🔴 Higher Asthma Risk"
            )

        else:

            level = "Lower"

            st.success(
                "🟢 Lower Asthma Risk"
            )


    # =================================================
    # RESULT EXPLANATION
    # =================================================

    st.subheader(
        "💡 Understanding the Result"
    )

    if prediction == 1:

        st.write(
            "The selected machine-learning model found a pattern "
            "that is more similar to records with asthma in the "
            "training dataset."
        )

    else:

        st.write(
            "The selected machine-learning model found a pattern "
            "that is more similar to records without asthma in "
            "the training dataset."
        )


    # =================================================
    # SUMMARY
    # =================================================

    st.subheader(
        "📋 Assessment Summary"
    )

    summary = []

    for key in features:

        summary.append({
            "Information": simple_names.get(
                key,
                key
            ),
            "Entered Value": values[key]
        })


    summary_df = pd.DataFrame(
        summary
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )


    # =================================================
    # REPORT
    # =================================================

    report = (
        "ASTHMA RISK PREDICTION REPORT\n"
    )

    report += (
        "============================\n\n"
    )

    report += (
        "Best Model: "
        + str(best_model_name)
        + "\n"
    )

    report += (
        "Risk Level: "
        + str(level)
        + " Asthma Risk\n"
    )

    if probability is not None:

        report += (
            "Estimated Risk: "
            + f"{risk:.2f}%"
            + "\n"
        )


    report += "\nMODEL PERFORMANCE\n"

    best_row = results_df.iloc[0]

    report += (
        "Accuracy: "
        + f"{best_row['Accuracy'] * 100:.2f}%"
        + "\n"
    )

    report += (
        "Precision: "
        + f"{best_row['Precision'] * 100:.2f}%"
        + "\n"
    )

    report += (
        "Recall: "
        + f"{best_row['Recall'] * 100:.2f}%"
        + "\n"
    )

    report += (
        "F1 Score: "
        + f"{best_row['F1 Score'] * 100:.2f}%"
        + "\n"
    )


    report += "\nPATIENT INFORMATION\n"

    for key in features:

        report += (
            simple_names.get(
                key,
                key
            )
            + ": "
            + str(values[key])
            + "\n"
        )


    report += "\nDISCLAIMER\n"

    report += (
        "This project is for educational purposes only "
        "and is not a medical diagnosis."
    )


    st.download_button(
        "📥 Download Assessment Report",
        data=report,
        file_name="asthma_risk_report.txt",
        mime="text/plain",
        use_container_width=True
    )


# =====================================================
# FUTURE IMPROVEMENTS
# =====================================================

st.divider()

st.header(
    "🔮 Future Improvements"
)

st.write(
    "🤖 Deep Learning based asthma prediction"
)

st.write(
    "🔍 Explainable AI using SHAP"
)

st.write(
    "📊 More advanced health visualizations"
)

st.write(
    "📈 Cross-validation and hyperparameter tuning"
)

st.write(
    "🧠 More respiratory health features"
)

st.write(
    "🌐 Online deployment"
)

st.write(
    "📱 Mobile application"
)


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "🫁 Asthma Risk Prediction | "
    "Machine Learning Project | Educational Use Only"
)