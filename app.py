import pickle
import pandas as pd
import streamlit as st

# Load the trained SVC model saved by the notebook
model = pickle.load(open("svc_trained_model.pkl", "rb"))

# Label encodings (same as the notebook's LabelEncoder: alphabetical order)
PCLASS = {"First": 0, "Second": 1, "Third": 2}
GENDER = {"Female": 0, "Male": 1}
SIBLING = {"One": 0, "Three": 1, "Two": 2, "Zero": 3}
EMBARKED = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}

st.title("Titanic Passenger Survival Prediction")
st.write("Enter the passenger's details and click Predict.")

pclass = st.selectbox("Passenger class", ["First", "Second", "Third"])
gender = st.selectbox("Gender", ["Male", "Female"])
sibling = st.selectbox("Siblings/spouses aboard", ["Zero", "One", "Two", "Three"])
embarked = st.selectbox("Port of embarkation", ["Southampton", "Cherbourg", "Queenstown"])

if st.button("Predict"):
    features = pd.DataFrame({
        "PClass": [PCLASS[pclass]],
        "Gender": [GENDER[gender]],
        "Sibling": [SIBLING[sibling]],
        "Embarked": [EMBARKED[embarked]],
    })
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.success("Prediction: SURVIVED")
    else:
        st.error("Prediction: NOT SURVIVED")
