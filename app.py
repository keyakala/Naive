import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Accuracy
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

# Streamlit UI
st.title("Iris Flower Classifier (Naive Bayes)")

st.write("Train Accuracy:", train_acc)
st.write("Test Accuracy:", test_acc)

st.subheader("Enter Flower Measurements")

sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

# Predict button
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)

    species = iris.target_names[prediction][0]

    st.success(f"Predicted Flower Species: {species}")

# Show confusion matrices
st.subheader("Confusion Matrix (Train)")
st.write(confusion_matrix(y_train, y_pred_train))

st.subheader("Confusion Matrix (Test)")
st.write(confusion_matrix(y_test, y_pred_test))