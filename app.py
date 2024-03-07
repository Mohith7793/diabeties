import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("/Users/mohith/Desktop/Diabetes1/diabetes.csv")

# Display title and dataset statistics
st.title('Diabetes Checkup')
st.subheader('Training Data Statistics')
st.write(df.describe())

# Split data into features and target
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Function to get user input
def user_input():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)
    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    features = pd.DataFrame(user_data, index=[0])
    return features

# Get user input
user_features = user_input()

# Display user input
st.subheader('Patient Data')
st.write(user_features)

# Train the model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Make predictions
prediction = rf_model.predict(user_features)

# Display prediction
st.subheader('Prediction')
if prediction[0] == 0:
    st.write('You are not Diabetic')
else:
    st.write('You are Diabetic')

# Calculate accuracy
accuracy = accuracy_score(y_test, rf_model.predict(X_test)) * 100
st.subheader('Accuracy')
st.write(f"{accuracy:.2f}%")

# Interactive plot between Glucose and Blood Pressure
st.subheader('Interactive Plot: Glucose vs Blood Pressure')
glucose_bp_fig = plt.figure()
sns.scatterplot(data=df, x='Glucose', y='BloodPressure', hue='Outcome', palette='coolwarm')
plt.xlabel('Glucose')
plt.ylabel('Blood Pressure')
st.pyplot(glucose_bp_fig)

# Interactive plot between BMI and Glucose
st.subheader('Interactive Plot: BMI vs Glucose')
bmi_glucose_fig = plt.figure()
sns.scatterplot(data=df, x='BMI', y='Glucose', hue='Outcome', palette='coolwarm')
plt.xlabel('BMI')
plt.ylabel('Glucose')
st.pyplot(bmi_glucose_fig)
