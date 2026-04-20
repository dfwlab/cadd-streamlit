import streamlit as st
from streamlit_ketcher import st_ketcher
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

# Set page configuration
st.set_page_config(page_title="CADD课程实践", page_icon="🔬")


# --- Helper Functions ---
# Display basic data summary
def display_data_summary(data):
    st.subheader("数据集概况")
    st.write("数据的基本信息：", data.info())
    st.write("描述性统计：", data.describe())

    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
    st.subheader("数值型特征的分布")
    for col in numeric_columns[:3]:
        st.write(f"{col} 的分布：")
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, ax=ax)
        ax.set_title(f"{col}")
        st.pyplot(fig)


# Create project directory with a unique name
def create_project_directory():
    project_name = datetime.now().strftime("%Y-%m-%d-%H-%M") + "_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    project_dir = os.path.join("./projects", project_name)
    os.makedirs(project_dir, exist_ok=True)
    return project_dir


# Generate fingerprint for a molecule
def mol_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp = fpgen.GetFingerprint(mol)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    else:
        st.warning(f"无法解析SMILES: {smiles}")
        return [None] * 2048


# Save fingerprint data to CSV
def save_input_data_with_fingerprint(data, project_dir, label_column):
    columns_name = 'smiles' if 'smiles' in data.columns else ('SMILES' if 'SMILES' in data.columns else None)
    if columns_name is None:
        st.write('无法找到名为 "smiles" 或 "SMILES" 的列!')
        return
    fingerprints = data[columns_name].apply(mol_to_fp)
    fingerprint_df = pd.DataFrame(fingerprints.tolist())
    fingerprint_df['label'] = data[label_column]
    output_file = os.path.join(project_dir, "input.csv")
    fingerprint_df.to_csv(output_file, index=False)
    st.write(f"Fingerprint data saved to {output_file}")
    return output_file


# Preprocess data by removing missing values and converting to numeric
def preprocess_data(fp_file):
    data = pd.read_csv(fp_file).dropna()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data.dropna()


# Train and save model, also evaluate and plot metrics
def train_and_save_model(fp_file, project_dir, rf_params):
    data = preprocess_data(fp_file)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError as e:
        st.error(f"train_test_split 出错：{e}")
        return None, None

    model = RandomForestClassifier(n_estimators=rf_params['n_estimators'], max_depth=rf_params['max_depth'], max_features=rf_params['max_features'], random_state=42)
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        st.error(f"模型训练失败：{e}")
        return None, None

    model_filename = "model.pkl"
    joblib.dump(model, os.path.join(project_dir, model_filename))
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plot_and_save(fpr, tpr, "ROC Curve", roc_auc, project_dir, "roc_curve.png")
    
    # Plot Confusion Matrix
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), project_dir)

    # Plot Feature Importance
    plot_feature_importance(model.feature_importances_, X.columns, project_dir)

    return model, acc, roc_auc


def plot_and_save(x, y, title, auc_score, project_dir, filename):
    fig, ax = plt.subplots()
    ax.plot(x, y, color='blue', lw=2, label=f'{title} (AUC = {auc_score:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    plt.savefig(os.path.join(project_dir, filename))
    st.image(os.path.join(project_dir, filename))


def plot_confusion_matrix(confusion, project_dir):
    fig, ax = plt.subplots()
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    plt.savefig(os.path.join(project_dir, "confusion_matrix.png"))
    st.image(os.path.join(project_dir, "confusion_matrix.png"))


def plot_feature_importance(importance, features, project_dir):
    fig, ax = plt.subplots()
    sns.barplot(x=list(features), y=importance, ax=ax)
    ax.set_title("Feature Importance")
    plt.savefig(os.path.join(project_dir, "feature_importance.png"))
    st.image(os.path.join(project_dir, "feature_importance.png"))

# 查询PubMed Central (PMC) 数据库
def search_pmc(keyword):
    search_term = keyword  # 输入搜索关键词
    handle = Entrez.esearch(db="pmc", term=search_term, retmode="xml", retmax=5)  # 限制返回5篇文章
    record = Entrez.read(handle)
    return record["IdList"]

# 获取文章详细信息
def fetch_article_details(pmcid):
    handle = Entrez.efetch(db="pmc", id=pmcid, retmode="text")
    record = Entrez.read(handle)
    return record


# --- Streamlit UI ---
sidebar_option = st.sidebar.selectbox("选择功能", ["首页", "数据展示", "模型训练", "活性预测", "查看已有项目", "知识获取"])

# 首页
if sidebar_option == "首页":
    # Set header
    st.markdown("""
        <h1 style="text-align: center; color: #4CAF50;">CADD课程实践</h1>
        <p style="text-align: center; font-size: 18px; color: #555;">欢迎来到我们的计算机辅助药物设计平台！选择您感兴趣的功能开始使用。</p>
    """, unsafe_allow_html=True)
    # Add some styling
    st.markdown("""
        <style>
            .card {
                background-color: #f9f9f9;
                border: 2px solid #d1d1d1;
                border-radius: 10px;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin-top: 20px;
                text-align: center;
                font-size: 16px;
            }
            .card:hover {
                background-color: #e8f4f8;
                cursor: pointer;
            }
            .card-title {
                font-size: 20px;
                font-weight: bold;
                color: #4CAF50;
            }
            .card-description {
                color: #666;
                font-size: 14px;
                margin-top: 10px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Add columns for a cleaner layout
    col1, col2, col3 = st.columns(3)
    
    # Define the clickable cards (functionality links)
    with col1:
        st.markdown("""
            <div class="card" onclick="window.location.href='#'">
                <div class="card-title">数据展示</div>
                <div class="card-description">查看数据集概况并生成相关的统计图表。</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="card" onclick="window.location.href='#'">
                <div class="card-title">模型训练</div>
                <div class="card-description">训练机器学习模型并评估性能(AUC曲线等)。</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="card" onclick="window.location.href='#'">
                <div class="card-title">活性预测</div>
                <div class="card-description">输入SMILES并进行化合物活性预测，进行SHAP解释。</div>
            </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="card" onclick="window.location.href='#'">
                <div class="card-title">查看已有项目</div>
                <div class="card-description">查看您之前创建的项目和模型评估结果。</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="card" onclick="window.location.href='#'">
                <div class="card-title">知识获取</div>
                <div class="card-description">获取文献中的毒副作用信息，支持文献摘要提取。</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="card" onclick="window.location.href='#'">
                <div class="card-title">其他功能</div>
                <div class="card-description">补充其他计算机辅助药物设计相关功能。</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <footer style="text-align: center; margin-top: 50px;">
            <p style="font-size: 14px; color: #888;">© 2025 计算机辅助药物设计课程实践平台 | 由TJCADD团队开发</p>
        </footer>
    """, unsafe_allow_html=True)

# 功能1: 展示数据
elif sidebar_option == "数据展示":
    st.title("数据展示")
    csv_files = glob.glob("./data/*.csv")
    dataset_choice = st.sidebar.selectbox("选择数据集", [os.path.basename(file) for file in csv_files])
    selected_file = csv_files[[os.path.basename(file) for file in csv_files].index(dataset_choice)]
    data = pd.read_csv(selected_file)
    display_data_summary(data)

# 功能2: 模型训练
elif sidebar_option == "模型训练":
    st.title("模型训练")
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
        st.write(f"训练完成，模型准确率(Accuracy): {acc:.4f}; 模型AUC: {roc_auc:.4f}")
        st.success(f"模型已保存到：{os.path.join(project_dir, 'model.pkl')}")

# 功能3: 活性预测
elif sidebar_option == "活性预测":
    st.title("活性预测")
    # List trained projects
    projects = glob.glob('./projects/*')
    if not projects:
        st.write("没有找到已训练的项目")
    else:
        project_names = [os.path.basename(project) for project in projects]
        project_name = st.selectbox("选择一个项目进行预测", project_names)
        selected_project_dir = os.path.join("./projects", project_name)

        # Load model for prediction
        model_filename = os.path.join(selected_project_dir, "model.pkl")
        if os.path.exists(model_filename):
            model = joblib.load(model_filename)
            st.write(f"加载模型：{model_filename}")

            # Input SMILES (by ketcher) for prediction
            molecule = st.text_input("输入分子SMILES", r"C1C=CC(C)=C(CC2C=C(CCC)C=C2)C=1")
            smile_code = st_ketcher(molecule)
            st.markdown(f"Smile code: ``{smile_code}``")
            
            if smile_code:
                fingerprint = mol_to_fp(smile_code)
                if fingerprint is not None:
                    prediction = model.predict([fingerprint])
                    prob = model.predict_proba([fingerprint])[:, -1]
                    st.write(f"预测结果: {prediction[0]}, 概率: {prob[0]}")

                    # SHAP explanation
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(fingerprint)
                    fig, ax = plt.subplots()
                    shap.summary_plot(shap_values, features=fingerprint, show=False)
                    st.pyplot(fig)
                else:
                    st.write("无法解析该SMILES字符串，请输入有效的SMILES。")
        else:
            st.write("没有找到模型文件，请确保该项目已训练并保存模型。")

# 功能4: 查看已有项目
elif sidebar_option == "查看已有项目":
    st.title("查看已有项目")
    projects = glob.glob('./projects/*')
    if not projects:
        st.write("没有找到项目")
    else:
        project_names = [os.path.basename(project) for project in projects]
        project_name = st.selectbox("选择一个项目查看", project_names)
        selected_project_dir = os.path.join("./projects", project_name)

        # Show files within project
        if os.path.exists(os.path.join(selected_project_dir, "input.csv")):
            data = pd.read_csv(os.path.join(selected_project_dir, "input.csv"))
            st.write("数据预览：")
            st.dataframe(data.head())

        # Display evaluation charts
        if os.path.exists(os.path.join(selected_project_dir, "confusion_matrix.png")):
            st.image(os.path.join(selected_project_dir, "confusion_matrix.png"))

        if os.path.exists(os.path.join(selected_project_dir, "feature_importance.png")):
            st.image(os.path.join(selected_project_dir, "feature_importance.png"))

# 功能5: 知识获取
elif sidebar_option == "知识获取":
    st.title("知识获取")
    # Set up Entrez email for PubMed search
    Entrez.email = "your_email@example.com"
    keyword = '"Clinical Toxicology" and "Chemical"'  # Search term
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
            full_text += re.sub(r'<.*?>', '', j.replace('\n', '')) + '\n'
    st.text_area("全文", full_text, height=300)

    key = st.text_input("请输入您的OpenAI Key用于解析文献知识", "")
    if key:
        os.environ["OPENAI_API_KEY"] = key
        client = OpenAI()

        # Query model for compound toxicity information
        st.write("常规提问:")
        query = f"""请从以下文献中提取与毒副作用相关的化合物信息，包括名字，类型和毒副作用描述：\n{abstract}"""
        response = client.responses.create(
            model="gpt-4",
            input=query
        )
        st.write(response.output_text)

        st.write("提示词工程:")
        query = f"""请从文献中提取与毒副作用相关的化合物信息,要求如下：
        1. 仅输出获取的信息，不要输出额外的文字，英文回复;
        2. 按照TSV格式输出结果，格式为："化合物\t类型\t毒副作用";
        3. 仅输出能从本文中得到的信息，本文缺失的信息输出为空;
        Examples:
        cocaine\tDrug\tDevelopmental toxicity and female reproductive toxicity
        Amphetamines\tDrug class\t
        End examples
        文献信息为:\n{abstract}"""
        response = client.responses.create(
            model="gpt-4",
            input=query
        )
        st.write(response.output_text)

        try:
            data = StringIO("化合物\t类型\t毒副作用\n" + response.output_text)
            df = pd.read_csv(data, sep='\t')
            st.dataframe(df)
        except:
            st.write("输出格式错误，无法解析为csv表格")
