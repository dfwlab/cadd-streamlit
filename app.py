import streamlit as st
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from rdkit import Chem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 设置页面标题和图标
st.set_page_config(page_title="Tox21数据集应用", page_icon="🔬")

# 添加CSS样式以美化界面
st.markdown("""
    <style>
        .css-1d391kg {background-color: #f0f4f7;}
        .sidebar .sidebar-content {background-color: #e1f5fe;}
        .sidebar .sidebar-title {font-size: 20px; font-weight: bold; color: #00796b;}
        .stButton>button {background-color: #00796b; color: white; border-radius: 10px; padding: 10px;}
        .stSelectbox>div {font-size: 16px; color: #00796b;}
    </style>
    """, unsafe_allow_html=True)

# 自动查找 ./data 目录下的所有 CSV 文件
def get_csv_files(directory="./data"):
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    return csv_files

# Function to display dataset info
def display_data_info(dataset):
    st.subheader("数据集概况")
    st.write(dataset.describe())
    st.subheader("缺失数据情况")
    st.write(dataset.isna().sum())
    st.subheader("数据分布")
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

# 获取 ./data 目录下的所有CSV文件
csv_files = get_csv_files()

# 检查是否有CSV文件
if not csv_files:
    st.error("没有找到CSV文件，请确保 './data' 目录下有CSV文件")
else:
    # 左侧边栏选择功能
    sidebar_option = st.sidebar.selectbox(
        "选择功能",
        ["数据展示", "模型训练", "活性预测"]
    )

    # 功能1：展示数据
    if sidebar_option == "数据展示":
        # 动态加载CSV文件
        dataset_choice = st.sidebar.selectbox("选择数据集", [os.path.basename(file) for file in csv_files])  # 获取文件名

        # 加载选定的数据集
        selected_file = csv_files[[os.path.basename(file) for file in csv_files].index(dataset_choice)]
        data = pd.read_csv(selected_file)
        
        # 显示数据集概况
        display_data_info(data)

    # 功能2：训练模型
    elif sidebar_option == "模型训练":
        # 动态加载CSV文件
        dataset_choice = st.sidebar.selectbox("选择数据集", [os.path.basename(file) for file in csv_files])  # 获取文件名

        # 加载选定的数据集
        selected_file = csv_files[[os.path.basename(file) for file in csv_files].index(dataset_choice)]
        data = pd.read_csv(selected_file)
        
        # 用户输入标签列名
        label_column = st.sidebar.text_input("输入标签列名", "tox21_label")
        
        if st.sidebar.button("训练模型"):
            train_model(data, label_column)

    # 功能3：进行预测
    elif sidebar_option == "活性预测":
        smiles_input = st.sidebar.text_input("输入分子SMILES")
        if st.sidebar.button("进行预测"):
            predict_new_molecule(smiles_input)
