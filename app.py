import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Predictive Maintenance System", layout="wide")

st.title("🚀 Predictive Maintenance System for Industrial Equipment")

st.write("""
This application predicts whether a machine will fail within the next 7 days 
based on industrial sensor data.
""")

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("machine_maintenance.csv")

    return df

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ----------------------------
# Feature Selection
# ----------------------------
features = [
    "temperature",
    "vibration",
    "pressure",
    "humidity",
    "runtime_hours",
    "load_percentage",
    "maintenance_history"
]

X = df[features]
y = df["failure_within_7days"]

# ----------------------------
# Train Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Scaling
# ----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------
# Model Training
# ----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# Evaluation
# ----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("📈 Model Performance")

st.write(f"**Accuracy:** {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
st.pyplot(fig)

# ----------------------------
# Predict for New Machine (Unknown Data)
# ----------------------------
st.subheader("🆕 Predict for New Machine Data")

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)", 0.0, 200.0, 80.0)
    vibration = st.number_input("Vibration Level", 0.0, 5.0, 0.5)
    pressure = st.number_input("Pressure", 0.0, 100.0, 30.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)

with col2:
    runtime_hours = st.number_input("Runtime Hours", 0.0, 5000.0, 1000.0)
    load_percentage = st.number_input("Load Percentage (%)", 0.0, 100.0, 70.0)
    maintenance_history = st.number_input("Maintenance History Count", 0, 20, 2)

if st.button("Predict Failure Risk"):

    new_data = np.array([[temperature,
                          vibration,
                          pressure,
                          humidity,
                          runtime_hours,
                          load_percentage,
                          maintenance_history]])

    new_data_scaled = scaler.transform(new_data)

    prediction = model.predict(new_data_scaled)[0]
    probability = model.predict_proba(new_data_scaled)[0][1]

    st.subheader("🔎 Prediction Result")

    if probability > 0.75:
        st.error(f"🚨 CRITICAL RISK ({probability:.2f}) — Immediate Maintenance Required")
    elif probability > 0.40:
        st.warning(f"⚠ Moderate Risk ({probability:.2f}) — Schedule Inspection")
    else:
        st.success(f"✅ Low Risk ({probability:.2f}) — Normal Operation")

# ----------------------------
# Feature Importance
# ----------------------------
st.subheader("📌 Feature Importance")

importance = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

fig2, ax2 = plt.subplots()
sns.barplot(x="Importance", y="Feature", data=importance_df)
st.pyplot(fig2)
