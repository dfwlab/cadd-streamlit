import streamlit as st
import pandas as pd
import os
import re
import glob
import joblib
import random
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import shap
from Bio import Entrez
from openai import OpenAI
from io import StringIO

# Streamlit page configuration
st.set_page_config(page_title="2025CADD课程实践", page_icon="🔬")

# Helper functions
def create_project_directory():
    project_name = datetime.now().strftime("%Y-%m-%d-%H-%M") + "_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    project_dir = os.path.join("./projects", project_name)
    os.makedirs(project_dir, exist_ok=True)
    return project_dir

def mol_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    st.warning(f"无法解析SMILES: {smiles}")
    return [None] * 2048

def display_data_summary(data):
    st.subheader("数据集概况")
    st.write("数据的基本信息：")
    st.write(data.info())
    st.write("描述性统计：")
    st.write(data.describe())

    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
    st.subheader("数值型特征的分布")
    for col in numeric_columns[:3]:
        st.write(f"{col} 的分布：")
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, ax=ax)
        ax.set_title(f"{col}")
        st.pyplot(fig)

def preprocess_data(fp_file):
    data = pd.read_csv(fp_file).dropna()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data.dropna()

def save_input_data_with_fingerprint(data, project_dir, label_column):
    columns_name = next((col for col in ['smiles', 'SMILES'] if col in data.columns), None)
    if not columns_name:
        st.write('Cannot find column named "smiles" or "SMILES" in the dataset!')
        return
    fingerprints = data[columns_name].apply(mol_to_fp)
    fingerprint_df = pd.DataFrame(fingerprints.tolist())
    fingerprint_df['label'] = data[label_column]
    output_file = os.path.join(project_dir, "input.csv")
    fingerprint_df.to_csv(output_file, index=False)
    st.write(f"Fingerprint data saved to {output_file}")
    return output_file

def train_and_save_model(fp_file, project_dir, rf_params):
    data = preprocess_data(fp_file)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    st.write("特征数据(X)形状：", X.shape)
    st.write("标签数据(y)形状：", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(**rf_params)
    model.fit(X_train, y_train)

    model_filename = "model.pkl"
    joblib.dump(model, os.path.join(project_dir, model_filename))
    st.write(f"模型已保存到：{os.path.join(project_dir, model_filename)}")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    save_and_display_roc_curve(fpr, tpr, roc_auc, project_dir)
    save_and_display_confusion_matrix(y_test, y_pred, project_dir)
    save_and_display_feature_importance(model, X.columns, project_dir)

    return model, acc, roc_auc

def save_and_display_roc_curve(fpr, tpr, roc_auc, project_dir):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    plt.savefig(os.path.join(project_dir, "roc_curve.png"))
    st.image(os.path.join(project_dir, "roc_curve.png"))

def save_and_display_confusion_matrix(y_test, y_pred, project_dir):
    confusion = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    plt.savefig(os.path.join(project_dir, "confusion_matrix.png"))
    st.image(os.path.join(project_dir, "confusion_matrix.png"))

def save_and_display_feature_importance(model, features, project_dir):
    feature_importances = model.feature_importances_
    fig, ax = plt.subplots()
    sns.barplot(x=list(features), y=feature_importances, ax=ax)
    ax.set_title("Feature Importance")
    plt.savefig(os.path.join(project_dir, "feature_importance.png"))
    st.image(os.path.join(project_dir, "feature_importance.png"))

def search_pmc(keyword):
    handle = Entrez.esearch(db="pmc", term=keyword, retmode="xml", retmax=5)
    return Entrez.read(handle)["IdList"]

def fetch_article_details(pmcid):
    handle = Entrez.efetch(db="pmc", id=pmcid, retmode="text")
    return Entrez.read(handle)

# Streamlit UI setup
st.title("2025CADD课程实践")
sidebar_option = st.sidebar.selectbox("选择功能", ["数据展示", "模型训练", "活性预测", "查看已有项目", "知识获取"])

# Data visualization
if sidebar_option == "数据展示":
    csv_files = glob.glob("./data/*.csv")
    dataset_choice = st.sidebar.selectbox("选择数据集", [os.path.basename(file) for file in csv_files])
    selected_file = csv_files[[os.path.basename(file) for file in csv_files].index(dataset_choice)]
    data = pd.read_csv(selected_file)
    display_data_summary(data)

# Model training
elif sidebar_option == "模型训练":
    csv_files = glob.glob("./data/*.csv")
    dataset_choice = st.sidebar.selectbox("选择数据集", [os.path.basename(file) for file in csv_files])
    selected_file = csv_files[[os.path.basename(file) for file in csv_files].index(dataset_choice)]
    data = pd.read_csv(selected_file)

    label_column = st.sidebar.selectbox("选择标签列", data.columns.tolist())
    rf_params = {
        'n_estimators': st.sidebar.slider("随机森林 n_estimators", 50, 500, 100),
        'max_depth': st.sidebar.slider("随机森林 max_depth", 1, 30, 3),
        'max_features': st.sidebar.slider("随机森林 max_features", 0.1, 1.0, 0.2)
    }

    if st.sidebar.button("开始训练模型"):
        project_dir = create_project_directory()
        fp_file = save_input_data_with_fingerprint(data, project_dir, label_column)
        model, acc, roc_auc = train_and_save_model(fp_file, project_dir, rf_params)
        st.write(f"训练完成，准确率: {acc:.4f}, AUC: {roc_auc:.4f}")

# Predictive model
elif sidebar_option == "活性预测":
    projects = glob.glob('./projects/*')
    if not projects:
        st.write("没有找到已训练的项目")
    else:
        project_names = [os.path.basename(project) for project in projects]
        project_name = st.selectbox("选择一个项目进行预测", project_names)
        selected_project_dir = os.path.join("./projects", project_name)
        model_filename = os.path.join(selected_project_dir, "model.pkl")
        
        if os.path.exists(model_filename):
            model = joblib.load(model_filename)
            st.write(f"加载模型：{model_filename}")

            smiles_input = st.text_input("输入分子SMILES")
            if smiles_input:
                fingerprint = mol_to_fp(smiles_input)
                if fingerprint is not None:
                    prediction = model.predict([fingerprint])
                    prob = model.predict_proba([fingerprint])[:, -1]
                    st.write(f"预测结果: {prediction[0]}, 概率: {prob[0]}")

                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(fingerprint)
                    fig, ax = plt.subplots()
                    shap.summary_plot(shap_values, features=fingerprint, show=False)
                    st.pyplot(fig)
        else:
            st.write("没有找到模型文件，请确保该项目已训练并保存模型。")

# Existing projects
elif sidebar_option == "查看已有项目":
    display_existing_projects()

# Knowledge retrieval from PubMed
elif sidebar_option == "知识获取":
    Entrez.email = "your_email@example.com"
    keyword = '"Clinical Toxicology" and "Chemical"'
    pmcid_list = search_pmc(keyword)
    st.write(f"关键词: {keyword}")
    st.write(f'搜索到的相关文献(前五篇): {list(pmcid_list)}')

    pmcid = '11966747'
    article_details = fetch_article_details(pmcid)
    st.write(f'从PMC获取文献"{pmcid}"全文: ')
    title = article_details[0]['front']['article-meta']['title-group']['article-title'].replace('\n', '')
    abstract = article_details[0]['front']['article-meta']['abstract'][0]['p'][1].replace('\n', '')
    st.info(f'题目: {title}')
    st.info(f'摘要: {abstract}')
    full_text = ""
    for i in article_details[0]['body']['sec']:
        for j in i['p']:
            full_text += re.sub(r'<.*?>', '', j.replace('\n', ''))+'\n'
    st.text_area("全文", full_text, height=300)

    key = st.text_input("请输入您的OpenAI Key用于解析文献知识", "")
    if key:
        os.environ["OPENAI_API_KEY"] = key
        client = OpenAI()
        query = """请从以下文献中提取与毒副作用相关的化合物信息，包括名字，类型和毒副作用描述：\n""" + abstract
        response = client.responses.create(model="gpt-4", input=query)
        st.write(response.output_text)
