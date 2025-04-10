import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from rdkit import Chem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to display dataset info
def display_data_info(dataset):
    st.write("数据集概况：")
    st.write(dataset.describe())
    st.write("缺失数据情况：")
    st.write(dataset.isna().sum())
    st.write("数据分布：")
    fig, ax = plt.subplots()
    sns.histplot(dataset.iloc[:, 0], kde=True, ax=ax)  # Example: first column of dataset
    st.pyplot(fig)

# Function to train and save a model
def train_model(dataset, label_column):
    X = dataset.drop(columns=[label_column])
    y = dataset[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"模型准确率：{acc:.4f}")
    
    # Save the model
    joblib.dump(model, 'tox21_model.pkl')
    st.success("模型已保存")

# Function to predict using the saved model
def predict_new_molecule(smiles):
    # Load the pre-trained model
    model = joblib.load('tox21_model.pkl')
    
    # Convert SMILES to features (use RDKit or other methods)
    mol = Chem.MolFromSmiles(smiles)
    # Here you should include code for feature extraction from SMILES
    # For simplicity, we assume we have a feature vector ready
    features = np.array([0])  # Dummy features
    
    prediction = model.predict(features.reshape(1, -1))
    st.write(f"预测结果：{prediction}")
    
    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    shap.summary_plot(shap_values, features)

# Streamlit UI
st.title("Tox21数据集建模与预测应用")

dataset_choice = st.selectbox("选择数据集", ["tox21", "其他数据集"])  # Example choice
if dataset_choice == "tox21":
    data = pd.read_csv("tox21.csv")  # Replace with actual data loading code
    display_data_info(data)

label_column = st.text_input("输入标签列名", "tox21_label")
if st.button("训练模型"):
    train_model(data, label_column)

smiles_input = st.text_input("输入分子SMILES")
if st.button("进行预测"):
    predict_new_molecule(smiles_input)
