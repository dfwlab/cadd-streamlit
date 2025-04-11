import streamlit as st
import pandas as pd
import os
import random
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import shap
from Bio import Entrez
from openai import OpenAI
from io import StringIO

# Configure page title and icon
st.set_page_config(page_title="2025CADD课程实践", page_icon="🔬")

# Utility functions
def generate_project_dir():
    """Generate a unique project directory."""
    project_name = datetime.now().strftime("%Y-%m-%d-%H-%M") + "_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    project_dir = os.path.join("./projects", project_name)
    os.makedirs(project_dir, exist_ok=True)
    return project_dir

def create_fingerprint(smiles):
    """Convert SMILES to molecular fingerprints."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp = fpgen.GetFingerprint(mol)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    return [None] * 2048  # Return a fingerprint array of zeros for invalid SMILES

def preprocess_and_save(data, project_dir, label_column):
    """Process input data and save fingerprint with labels to CSV."""
    smiles_col = 'smiles' if 'smiles' in data.columns else 'SMILES'
    if smiles_col not in data.columns:
        st.error('No valid SMILES column found.')
        return None
    fingerprints = data[smiles_col].apply(create_fingerprint)
    fp_df = pd.DataFrame(fingerprints.tolist())
    fp_df['label'] = data[label_column]
    file_path = os.path.join(project_dir, "input.csv")
    fp_df.to_csv(file_path, index=False)
    return file_path

def train_rf_model(fp_file, project_dir, rf_params):
    """Train and evaluate RandomForest model."""
    data = pd.read_csv(fp_file).dropna()
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(**rf_params)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(project_dir, "model.pkl"))

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    # Save and display evaluation plots
    save_plots(project_dir, fpr, tpr, roc_auc, confusion_matrix(y_test, y_pred), model.feature_importances_)

    return model, acc, roc_auc

def save_plots(project_dir, fpr, tpr, roc_auc, confusion, feature_importances):
    """Generate and save plots."""
    # ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(project_dir, "roc_curve.png"))
    st.image(os.path.join(project_dir, "roc_curve.png"))

    # Confusion matrix
    plt.figure()
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(project_dir, "confusion_matrix.png"))
    st.image(os.path.join(project_dir, "confusion_matrix.png"))

    # Feature importance
    plt.figure()
    sns.barplot(x=list(X.columns), y=feature_importances)
    plt.title("Feature Importance")
    plt.savefig(os.path.join(project_dir, "feature_importance.png"))
    st.image(os.path.join(project_dir, "feature_importance.png"))

# Streamlit UI functions
def display_data_summary(data):
    """Display basic dataset summary and plots."""
    st.subheader("数据集概况")
    st.write(data.info())
    st.write(data.describe())
    for col in data.select_dtypes(include=['number']).columns[:3]:
        st.write(f"{col} 的分布：")
        sns.histplot(data[col], kde=True)
        st.pyplot()

# Main Streamlit app layout
st.title("2025CADD课程实践")
sidebar_option = st.sidebar.selectbox("选择功能", ["数据展示", "模型训练", "活性预测", "查看已有项目", "知识获取"])

# 数据展示
if sidebar_option == "数据展示":
    csv_files = glob.glob("./data/*.csv")
    dataset_choice = st.sidebar.selectbox("选择数据集", [os.path.basename(f) for f in csv_files])
    selected_file = pd.read_csv(csv_files[[os.path.basename(f) for f in csv_files].index(dataset_choice)])
    display_data_summary(selected_file)

# 模型训练
elif sidebar_option == "模型训练":
    csv_files = glob.glob("./data/*.csv")
    dataset_choice = st.sidebar.selectbox("选择数据集", [os.path.basename(f) for f in csv_files])
    selected_file = pd.read_csv(csv_files[[os.path.basename(f) for f in csv_files].index(dataset_choice)])

    label_column = st.sidebar.selectbox("选择标签列", selected_file.columns)
    rf_params = {
        'n_estimators': st.sidebar.slider("n_estimators", 50, 500, 100),
        'max_depth': st.sidebar.slider("max_depth", 1, 30, 3),
        'max_features': st.sidebar.slider("max_features", 0.1, 1.0, 0.2)
    }
    if st.sidebar.button("开始训练模型"):
        project_dir = generate_project_dir()
        fp_file = preprocess_and_save(selected_file, project_dir, label_column)
        model, acc, roc_auc = train_rf_model(fp_file, project_dir, rf_params)
        st.write(f"训练完成，准确率: {acc:.4f}, AUC: {roc_auc:.4f}")

# 活性预测
elif sidebar_option == "活性预测":
    projects = glob.glob('./projects/*')
    if projects:
        project_name = st.selectbox("选择一个项目进行预测", [os.path.basename(p) for p in projects])
        model_file = os.path.join("./projects", project_name, "model.pkl")
        if os.path.exists(model_file):
            model = joblib.load(model_file)
            smiles_input = st.text_input("输入SMILES")
            if smiles_input:
                fingerprint = create_fingerprint(smiles_input)
                prediction = model.predict([fingerprint])
                prob = model.predict_proba([fingerprint])[:, -1]
                st.write(f"预测结果: {prediction[0]}, 概率: {prob[0]}")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(fingerprint)
                shap.summary_plot(shap_values, features=fingerprint)
                st.pyplot()

# 查看已有项目
elif sidebar_option == "查看已有项目":
    projects = glob.glob('./projects/*')
    if projects:
        project_name = st.selectbox("选择一个项目查看", [os.path.basename(p) for p in projects])
        selected_project_dir = os.path.join("./projects", project_name)
        if os.path.exists(os.path.join(selected_project_dir, "input.csv")):
            data = pd.read_csv(os.path.join(selected_project_dir, "input.csv"))
            st.write("数据预览：")
            st.dataframe(data.head())
        if os.path.exists(os.path.join(selected_project_dir, "confusion_matrix.png")):
            st.image(os.path.join(selected_project_dir, "confusion_matrix.png"))

# 知识获取
elif sidebar_option == "知识获取":
    keyword = '"Clinical Toxicology" and "Chemical"'
    pmcid_list = search_pmc(keyword)
    st.write(f"相关文献：{pmcid_list}")
    pmcid = '11966747'
    article_details = fetch_article_details(pmcid)
    st.write(f'文章摘要: {article_details[0]["front"]["article-meta"]["abstract"][0]["p"][1]}')

    key = st.text_input("输入OpenAI API密钥", "")
    if key:
        os.environ["OPENAI_API_KEY"] = key
        client = OpenAI()
        query = f"从以下文献中提取与毒副作用相关的化合物信息: {article_details[0]['front']['article-meta']['abstract'][0]['p'][1]}"
        response = client.responses.create(model="gpt-4", input=query)
        st.write(response.output_text)
