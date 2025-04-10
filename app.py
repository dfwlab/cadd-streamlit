import streamlit as st
import pandas as pd
import os
import glob
import joblib
import random
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem, DataStructs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import shap
from Bio import Entrez
from openai import OpenAI

# 设置页面标题和图标
st.set_page_config(page_title="2025CADD课程实践", page_icon="🔬")

# 显示数据集概况的函数
def display_data_summary(data):
    st.subheader("数据集概况")

    # 显示数据的基本信息和描述性统计
    st.write("数据的基本信息：")
    st.write(data.info())

    st.write("描述性统计：")
    st.write(data.describe())

    # 选择数值型列进行分布绘图
    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()

    # 绘制每个数值型特征的直方图
    st.subheader("数值型特征的分布")
    for col in numeric_columns[:3]:
        st.write(f"{col} 的分布：")
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, ax=ax)
        ax.set_title(f"{col}")
        st.pyplot(fig)
        

# 创建项目目录并命名
def create_project_directory():
    project_name = datetime.now().strftime("%Y-%m-%d-%H-%M") + "_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    project_dir = os.path.join("./projects", project_name)
    os.makedirs(project_dir, exist_ok=True)
    return project_dir

# 初始化Fingerprint生成器
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
# 定义新的Fingerprint转换函数
def mol_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = fpgen.GetFingerprint(mol)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    else:
        st.warning(f"无法解析SMILES: {smiles}")
        return [None] * 2048  # 返回一个全为None的fingerprint，表示解析失败

# 将fingerprint数据保存为csv
def save_input_data_with_fingerprint(data, project_dir, label_column):
    # 检查SMILES列（考虑大小写不一致）
    columns_name = 'smiles' if 'smiles' in data.columns else ('SMILES' if 'SMILES' in data.columns else None)
    if columns_name is None:
        st.write('Cannot find column named "smiles" or "SMILES" in the dataset!')
        return
    # 计算fingerprints并合并到数据集中
    fingerprints = data[columns_name].apply(mol_to_fp)
    # 创建DataFrame并添加标签列
    fingerprint_df = pd.DataFrame(fingerprints.tolist())
    fingerprint_df['label'] = data[label_column]
    # 保存结果为CSV文件
    output_file = os.path.join(project_dir, "input.csv")
    fingerprint_df.to_csv(output_file, index=False)
    # 返回保存的文件路径，方便后续操作
    st.write(f"Fingerprint data saved to {output_file}")
    return output_file

# 删除缺失数据并确保数据是数值型
def preprocess_data(fp_file):
    data = pd.read_csv(fp_file)
    # 删除包含缺失值的行
    data = data.dropna()
    # 确保数据都是数值型
    # 对于非数值型列，可以进行转换，例如使用OneHotEncoder等方法
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')  # 转换成数值型，如果无法转换则置为NaN
    # 删除包含NaN值的行（再次确保没有NaN）
    data = data.dropna()
    return data

# 训练并保存模型
def train_and_save_model(fp_file, project_dir, rf_params):
    # 预处理数据
    data = preprocess_data(fp_file)
    # 特征与标签分离
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    # 检查X和y的维度
    st.write("特征数据(X)形状：", X.shape)
    st.write("标签数据(y)形状：", y.shape)
    # 划分训练集和测试集
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError as e:
        st.error(f"train_test_split 出错：{e}")
        return None, None
    # 初始化模型
    model = RandomForestClassifier(n_estimators=rf_params['n_estimators'], max_depth=rf_params['max_depth'], max_features=rf_params['max_features'], random_state=42)
    # 训练模型
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        st.error(f"模型训练失败：{e}")
        return None, None
    # 保存模型
    model_filename = "model.pkl"
    joblib.dump(model, os.path.join(project_dir, model_filename))
    # 评估模型
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 计算AUC
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # 绘制AUC曲线
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    plt.savefig(os.path.join(project_dir, "roc_curve.png"))
    st.image(os.path.join(project_dir, "roc_curve.png"))

    # 保存评估结果
    confusion = confusion_matrix(y_test, y_pred)
    # 保存混淆矩阵图
    fig, ax = plt.subplots()
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    plt.savefig(os.path.join(project_dir, "confusion_matrix.png"))
    st.image(os.path.join(project_dir, "confusion_matrix.png"))
    
    # 特征重要性图
    feature_importances = model.feature_importances_
    fig, ax = plt.subplots()
    sns.barplot(x=list(X.columns), y=feature_importances, ax=ax)
    ax.set_title("Feature Importance")
    plt.savefig(os.path.join(project_dir, "feature_importance.png"))
    st.image(os.path.join(project_dir, "feature_importance.png"))
    
    return model, acc, roc_auc

# 查看已有的项目
def display_existing_projects():
    projects = glob.glob('./projects/*')
    if not projects:
        st.write("没有找到项目")
    else:
        project_names = [os.path.basename(project) for project in projects]
        project_name = st.selectbox("选择一个项目查看", project_names)
        selected_project_dir = os.path.join("./projects", project_name)
        
        # 展示项目中的文件
        if os.path.exists(os.path.join(selected_project_dir, "input.csv")):
            data = pd.read_csv(os.path.join(selected_project_dir, "input.csv"))
            st.write("数据预览：")
            st.dataframe(data.head())
        
        # 展示评估图表
        if os.path.exists(os.path.join(selected_project_dir, "confusion_matrix.png")):
            st.image(os.path.join(selected_project_dir, "confusion_matrix.png"))
        
        if os.path.exists(os.path.join(selected_project_dir, "feature_importance.png")):
            st.image(os.path.join(selected_project_dir, "feature_importance.png"))

# 查询PubMed Central (PMC) 数据库
def search_pmc(keyword):
    search_term = keyword  # 输入搜索关键词
    handle = Entrez.esearch(db="pmc", term=search_term, retmode="xml", retmax=5)  # 限制返回5篇文章
    record = Entrez.read(handle)
    return record["IdList"]

# 获取文章详细信息
def fetch_article_details(pmcid):
    handle = Entrez.efetch(db="pmc", id=pmcid, retmode="xml")
    record = Entrez.read(handle)
    return record

# 获取全文链接
def extract_full_text(article_record):
    # 根据具体的XML结构提取文本部分
    full_text = ""
    for article in article_record:
        for body in article.get("body", []):
            full_text += body.get("text", "")
    return full_text



# Streamlit UI
st.title("2025CADD课程实践")

# 左侧边栏选择功能
sidebar_option = st.sidebar.selectbox(
    "选择功能",
    ["数据展示", "模型训练", "活性预测", "查看已有项目", "知识获取"]
)

# 功能1：展示数据
if sidebar_option == "数据展示":
    # 动态加载CSV文件
    csv_files = glob.glob("./data/*.csv")
    dataset_choice = st.sidebar.selectbox("选择数据集", [os.path.basename(file) for file in csv_files])

    # 加载选定的数据集
    selected_file = csv_files[[os.path.basename(file) for file in csv_files].index(dataset_choice)]
    data = pd.read_csv(selected_file)
    
    # 显示数据集概况
    display_data_summary(data)

# 功能2：训练模型
elif sidebar_option == "模型训练":
    # 动态加载CSV文件
    csv_files = glob.glob("./data/*.csv")
    dataset_choice = st.sidebar.selectbox("选择数据集", [os.path.basename(file) for file in csv_files])

    # 加载选定的数据集
    selected_file = csv_files[[os.path.basename(file) for file in csv_files].index(dataset_choice)]
    data = pd.read_csv(selected_file)
    
    # 动态获取数据集的列名，并让用户选择标签列
    label_column = st.sidebar.selectbox("选择标签列", data.columns.tolist())
    
    # 设置RandomForest参数
    rf_params = {
        'n_estimators': st.sidebar.slider("随机森林 n_estimators", 50, 500, 100),
        'max_depth': st.sidebar.slider("随机森林 max_depth", 1, 30, 3),
        'max_features': st.sidebar.slider("随机森林 max_features", 0.1, 1.0, 0.2)
    }

    # 开始建模
    if st.sidebar.button("开始训练模型"):
        # 创建项目目录
        project_dir = create_project_directory()

        # 计算fingerprint并保存
        fp_file = save_input_data_with_fingerprint(data, project_dir, label_column)
        
        # 训练模型并保存结果
        model, acc, roc_auc = train_and_save_model(fp_file, project_dir, rf_params)
        
        # 显示训练结果
        st.write(f"训练完成，模型准确率(Accuracy): {acc:.4f}; 模型AUC: {roc_auc:.4f}")
        st.success(f"模型已保存到：{os.path.join(project_dir, 'model.pkl')}")

# 功能3：进行预测
elif sidebar_option == "活性预测":
    # 列出已训练的项目
    projects = glob.glob('./projects/*')
    if not projects:
        st.write("没有找到已训练的项目")
    else:
        project_names = [os.path.basename(project) for project in projects]
        project_name = st.selectbox("选择一个项目进行预测", project_names)
        selected_project_dir = os.path.join("./projects", project_name)
        
        # 加载选择的项目中的模型
        model_filename = os.path.join(selected_project_dir, "model.pkl")
        if os.path.exists(model_filename):
            model = joblib.load(model_filename)
            st.write(f"加载模型：{model_filename}")
            
            # 输入SMILES并进行预测
            smiles_input = st.text_input("输入分子SMILES")
            if smiles_input:
                fingerprint = mol_to_fp(smiles_input)
                if fingerprint is not None:
                    prediction = model.predict([fingerprint])
                    prob = model.predict_proba([fingerprint])[:, -1]
                    st.write(f"预测结果: {prediction[0]}, 概率: {prob[0]}")
                    
                    # SHAP解释
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(fingerprint)
                    fig, ax = plt.subplots()
                    shap.summary_plot(shap_values, features=fingerprint, show=False)
                    st.pyplot(fig)
                else:
                    st.write("无法解析该SMILES字符串，请输入有效的SMILES。")
        else:
            st.write("没有找到模型文件，请确保该项目已训练并保存模型。")

# 功能4：查看已有项目
elif sidebar_option == "查看已有项目":
    display_existing_projects()

# 功能5:知识获取
elif sidebar_option == "知识获取":
    key = st.text_input("请输入您的OpenAI Key", "")
    
    # 设置Entrez邮箱
    Entrez.email = "your_email@example.com"
    keyword = '"Clinical Toxicology" and "Chemical"'  # 搜索关键词
    pmcid_list = search_pmc(keyword)
    st.write(f"关键词: {keyword}")
    st.write(pmcid_list)

    pmcid = pmcid_list[0]
    article_details = fetch_article_details(pmcid)
    st.write(f'Fecth article : {pmcid}')
    st.write(article_details[0]['body']['sec'])
    #full_text = extract_full_text(article_details)

    if key:
        os.environ["OPENAI_API_KEY"] = key
        client = OpenAI()
        # 文献内容，假设已经以字符串形式提取
        document_text = str(article_details[0]['body']['sec'])
        # 提问模型以获取化合物的毒副作用信息
        query = """请从以下文献中提取与毒副作用相关的化合物名字，别名或其结构等信息：\n""" + document_text.replace('\n', '').replace('\n', '')[:10000]
        st.write(query)

        response = client.responses.create(
            model="gpt-4",
            input=query
        )
        st.write(response.output_text)
