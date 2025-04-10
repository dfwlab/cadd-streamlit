import streamlit as st
import pandas as pd
import os
import glob
import joblib
import random
import string
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import shap

# 设置页面标题和图标
st.set_page_config(page_title="Tox21数据集应用", page_icon="🔬")

# 创建项目目录并命名
def create_project_directory():
    project_name = datetime.now().strftime("%Y-%m-%d-%H-%M") + "_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    project_dir = os.path.join("./projects", project_name)
    os.makedirs(project_dir, exist_ok=True)
    return project_dir

# 计算fingerprint
def calculate_fingerprint(smiles):
    # 尝试生成分子对象
    mol = Chem.MolFromSmiles(smiles)
    
    # 检查分子对象是否有效
    if mol is None:
        st.warning(f"无法解析SMILES: {smiles}")
        return [None] * 2048  # 返回一个全为None的fingerprint，表示解析失败
    
    # 计算fingerprint（Morgan Fingerprint）
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return list(fingerprint)

# 将fingerprint数据保存为csv
def save_input_data_with_fingerprint(data, project_dir, label_column):
    # 检查SMILES列（考虑大小写不一致）
    columns_name = 'smiles' if 'smiles' in data.columns else ('SMILES' if 'SMILES' in data.columns else None)
    
    if columns_name is None:
        st.write('Cannot find column named "smiles" or "SMILES" in the dataset!')
        return
    
    # 计算fingerprints并合并到数据集中
    fingerprints = data[columns_name].apply(calculate_fingerprint)
    
    # 删除所有fingerprint为None的行（即SMILES解析失败的行）
    valid_fingerprints = fingerprints.apply(lambda x: x != [None] * 2048)
    data = data[valid_fingerprints]
    fingerprints = fingerprints[valid_fingerprints]
    
    # 创建DataFrame并添加标签列
    fingerprint_df = pd.DataFrame(fingerprints.tolist())
    fingerprint_df['label'] = data[label_column]
    
    # 保存结果为CSV文件
    output_file = os.path.join(project_dir, "input.csv")
    fingerprint_df.to_csv(output_file, index=False)
    
    # 返回保存的文件路径，方便后续操作
    st.write(f"Fingerprint data saved to {output_file}")
    return output_file

# 训练并保存模型
def train_and_save_model(data, label_column, project_dir, rf_params):
    X = data.drop(columns=[label_column])
    y = data[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestClassifier(n_estimators=rf_params['n_estimators'], max_depth=rf_params['max_depth'])
    model.fit(X_train, y_train)
    
    # 保存模型
    model_filename = "model.pkl"
    joblib.dump(model, os.path.join(project_dir, model_filename))
    
    # 评估模型
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # 保存评估结果
    confusion = confusion_matrix(y_test, y_pred)
    st.write(f"模型准确率：{acc:.4f}")
    
    # 保存混淆矩阵图
    fig, ax = plt.subplots()
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    plt.savefig(os.path.join(project_dir, "confusion_matrix.png"))
    
    # 特征重要性图
    feature_importances = model.feature_importances_
    fig, ax = plt.subplots()
    sns.barplot(x=list(X.columns), y=feature_importances, ax=ax)
    ax.set_title("Feature Importance")
    plt.savefig(os.path.join(project_dir, "feature_importance.png"))
    
    return model, acc

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

# Streamlit UI
st.title("Tox21数据集建模与预测应用")

# 左侧边栏选择功能
sidebar_option = st.sidebar.selectbox(
    "选择功能",
    ["数据展示", "模型训练", "活性预测", "查看已有项目"]
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
    st.subheader("数据集概况")
    st.write(data.describe())

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
        'max_depth': st.sidebar.slider("随机森林 max_depth", 3, 30, 10)
    }

    # 开始建模
    if st.sidebar.button("开始训练模型"):
        # 创建项目目录
        project_dir = create_project_directory()

        # 计算fingerprint并保存
        save_input_data_with_fingerprint(data, project_dir, label_column)
        
        # 训练模型并保存结果
        model, acc = train_and_save_model(data, label_column, project_dir, rf_params)
        
        # 显示训练结果
        st.write(f"训练完成，模型准确率：{acc:.4f}")
        st.success(f"模型已保存到：{os.path.join(project_dir, 'model.pkl')}")

# 功能3：进行预测
elif sidebar_option == "活性预测":
    smiles_input = st.sidebar.text_input("输入分子SMILES")
    if st.sidebar.button("进行预测"):
        st.write("待实现预测功能")

# 功能4：查看已有项目
elif sidebar_option == "查看已有项目":
    display_existing_projects()
