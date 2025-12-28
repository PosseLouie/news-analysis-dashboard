import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image

# ==========================================
# 1. 页面配置 (必须是第一行)
# ==========================================
st.set_page_config(
    page_title="英俄语虚假新闻计量特征研究",
    page_icon="📰",
    layout="wide"
)

# ==========================================
# 2. 侧边栏：项目介绍与导航
# ==========================================
st.sidebar.title("🔬 研究控制台")
st.sidebar.info("本系统用于复现《英俄语虚假新闻共性计量特征挖掘》一文的核心实验。")
st.sidebar.markdown("---")
# 你可以根据实际情况修改下面的名字
st.sidebar.markdown("**汇报人：** [你的名字]") 
st.sidebar.markdown("**指导老师：** [老师名字]")

# 菜单选择
option = st.sidebar.radio(
    "选择展示模块：",
    ("1. 数据概览 & 特征筛选", "2. PCA 三维空间分布", "3. 跨语言自动聚类")
)

# ==========================================
# 3. 加载数据函数
# ==========================================
@st.cache_data
def load_data():
    # 读取你之前生成的筛选后特征表
    # 确保 step1_selected_features.xlsx 在同一目录下
    try:
        df = pd.read_excel("step1_selected_features.xlsx")
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("❌ 数据加载失败！请确保目录中包含 'step1_selected_features.xlsx' 文件。")
    st.stop()

# ==========================================
# 4. 模块一：数据概览
# ==========================================
if option == "1. 数据概览 & 特征筛选":
    st.title("📊 数据概览与共性特征")
    st.markdown("### 1.1 实验数据集")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("总样本数", len(df))
    col2.metric("语言种类", f"{df['Language'].nunique()} (English, Russian)")
    col3.metric("类别标签", f"{df['Label'].nunique()} (Fake, Real)")
    
    st.markdown("#### 数据样本预览")
    st.dataframe(df.head(), use_container_width=True)
    
    st.markdown("### 1.2 经 T检验 筛选出的共性特征")
    st.info("以下特征在英俄双语中均通过了显著性检验 (P < 0.05)，被认定为跨语言共性特征：")
    
    # 提取特征列名 (排除非特征列)
    feature_cols = [c for c in df.columns if c not in ['Language', 'Label', 'Set_Type']]
    
    # 动态展示特征列表
    cols = st.columns(3)
    for i, feature in enumerate(feature_cols):
        cols[i % 3].success(f"✅ {feature}")

    st.markdown("""
    **特征说明：**
    * `Avg_Dep_Dist`: 平均依存距离 (句法复杂度核心指标)
    * `Type_Token_Ratio`: 词汇丰富度
    * `LIX_Index` / `ARI_Index`: 可读性指标
    """)

# ==========================================
# 5. 模块二：PCA 3D 可视化 (交互式)
# ==========================================
elif option == "2. PCA 三维空间分布":
    st.title("🧊 多维特征降维可视化 (PCA)")
    st.markdown("> **操作提示**：请使用鼠标**拖动旋转**下方的立方体，观察不同语言真假新闻的分布差异。滚轮可缩放。")
    
    # --- PCA 计算逻辑 (实时计算，保证与你代码逻辑一致) ---
    feature_cols = [c for c in df.columns if c not in ['Language', 'Label', 'Set_Type']]
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    # PCA 降维
    pca = PCA(n_components=3)
    components = pca.fit_transform(X_scaled)
    
    # 将结果存回 DataFrame 用于绘图
    df_pca = df.copy()
    df_pca['PC1'] = components[:, 0]
    df_pca['PC2'] = components[:, 1]
    df_pca['PC3'] = components[:, 2]
    df_pca['Group'] = df_pca['Language'] + '-' + df_pca['Label']
    
    # 计算解释方差比
    var_ratio = pca.explained_variance_ratio_
    
    # --- Plotly 交互式绘图 ---
    # 定义颜色映射 (对应你 matplotlib 代码中的配色)
    color_map = {
        'English-Fake': '#d62728', # 红色
        'English-Real': '#1f77b4', # 蓝色
        'Russian-Fake': '#ff7f0e', # 橙色
        'Russian-Real': '#2ca02c'  # 绿色
    }
    
    # 定义形状映射 (对应你代码中的 markers: English='o', Russian='^')
    # Plotly 中 'circle' 对应圆圈, 'diamond' 对应菱形/三角类
    symbol_map = {
        'English': 'circle',
        'Russian': 'diamond'
    }
    
    fig = px.scatter_3d(
        df_pca, 
        x='PC1', y='PC2', z='PC3',
        color='Group', 
        symbol='Language',
        color_discrete_map=color_map,
        symbol_map=symbol_map,
        opacity=0.8,
        height=700,
        hover_data=['Language', 'Label'],
        labels={'PC1': f'PC1 ({var_ratio[0]:.1%})', 
                'PC2': f'PC2 ({var_ratio[1]:.1%})', 
                'PC3': f'PC3 ({var_ratio[2]:.1%})'},
        title="英俄语真假新闻共性特征 PCA 3D 分布"
    )
    
    # 调整点的尺寸和边框
    fig.update_traces(marker=dict(size=6, line=dict(width=1, color='White')))
    
    # 在网页上展示图表
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📈 观察结论")
    st.markdown("""
    * 🔶 **俄语 (橙/绿)**：
        * 真假新闻在空间中**分离度非常高**，几乎处于两个不同的区域。
        * 这说明俄语假新闻的特征非常明显，容易被识别。
    * 🔴 **英语 (红/蓝)**：
        * 真假新闻点位**相互交织 (Overlapping)**，没有明显的分界线。
        * 这验证了论文结论：英语假新闻的伪装性更强，仅靠计量特征难以简单区分。
    """)

# ==========================================
# 6. 模块三：聚类分析
# ==========================================
elif option == "3. 跨语言自动聚类":
    st.title("🧬 层次聚类分析 (Dendrogram)")
    st.markdown("机器在不知道文章“真假”标签的情况下，是如何根据特征自动对它们进行分类的？")
    
    # 加载之前保存的图片
    try:
        # 使用你上传的文件名 clustering_dendrogram.png
        image = Image.open('clustering_dendrogram.png')
        st.image(image, caption='跨语言层次聚类树状图 (Hierarchical Clustering)', use_column_width=True)
    except FileNotFoundError:
        st.warning("⚠️ 未找到 'clustering_dendrogram.png' 图片。请确保该图片在同一目录下。")
        
    st.markdown("### 🔬 聚类结果解读")
    st.info("""
    **图表解读指南：**
    这是一棵“家谱树”，树下方的每一个叶子节点代表一篇新闻。被同一条颜色的线连接在一起的节点，表示机器认为它们是一类。
    
    1. **第一层分裂 (语言差异)**：
       树状图顶端首先将样本分成了两大枝。左侧主要是俄语 (Ru)，右侧主要是英语 (En)。说明**语言本身的差异是最大的**。
    
    2. **俄语区 (左侧分支)**：
       * 观察底部的标签，可以发现 `Ru-Fake` (俄语假新闻) 和 `Ru-Real` (俄语真新闻) 即使在同一语言分支下，也被分到了**不同的子群**。
       * 机器能很清楚地把它们分开。
    
    3. **英语区 (右侧分支)**：
       * 观察右侧的标签，`En-Fake` 和 `En-Real` 是**完全交错**排列的。
       * 机器“困惑”了，无法有效区分英语的真假新闻。
    """)