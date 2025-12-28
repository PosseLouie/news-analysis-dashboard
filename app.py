import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit.components.v1 as components  # 新增这行
from PIL import Image

# ==========================================
# 0. 中文名称映射字典 (配置区)
# ==========================================
NAME_MAP = {
    # 基础信息
    "Language": "语言",
    "Label": "真实性标签",
    "Filename": "文件名",
    "Set_Type": "数据集类型",
    "Group": "分组",
    
    # 词特征
    "Avg_Word_Len": "平均词长",
    "Type_Token_Ratio": "类型符比 (TTR)",
    "NOUN_Ratio": "名词占比",
    "VERB_Ratio": "动词占比",
    "ADJ_Ratio": "形容词占比",
    "PRON_Ratio": "代词占比",
    
    # 句特征
    "Avg_Sent_Len": "平均句长",
    "Avg_Dep_Dist": "平均依存距离 (DD)",
    "Nsubj_Ratio": "名词主语占比",
    "Amod_Ratio": "形容词修饰占比",
    "Advmod_Ratio": "状语修饰占比",
    
    # 可读性特征
    "LIX_Index": "LIX可读性指数",
    "ARI_Index": "ARI自动化可读性指数"
}

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(
    page_title="英俄语虚假新闻计量特征研究",
    page_icon="📰",
    layout="wide"
)

# ==========================================
# 2. 侧边栏
# ==========================================
st.sidebar.title("🔬 研究控制台")
# 去除"本系统用于..."的说明书语气，改为项目背景描述
st.sidebar.info("英俄语虚假新闻共性计量特征挖掘Method复现")

# 选项文案微调，去除序号，显得更现代
option = st.sidebar.radio(
    "导航：",
    ("数据概览与特征", "PCA 降维分析", "跨语言聚类验证")
)

# ==========================================
# 3. 加载数据
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("step1_selected_features.xlsx")
        return df
    except FileNotFoundError:
        return None

df_raw = load_data()

if df_raw is None:
    st.error("⚠️ 数据文件缺失：请确认 'step1_selected_features.xlsx' 已上传至根目录。")
    st.stop()

def scroll_to_top():
    # 这段 JS 代码会找到 Streamlit 的主滚动容器并将其卷动到顶部
    js_code = """
        <script>
            var body = window.parent.document.querySelector(".main");
            console.log(body);
            if (body) {
                body.scrollTop = 0;
            }
        </script>
    """
    components.html(js_code, height=0)

# ==========================================
# 4. 模块一：数据概览
# ==========================================
if option == "数据概览与特征":
    st.title("📊 数据概览与共性特征")
    
    # --- 1.1 数据集统计 ---
    # 去除 "1.1 实验数据集" 这种大纲式标题，直接用自然段落
    st.markdown("本次实验涵盖英俄双语的真实与虚假新闻样本，基础分布如下：")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("总样本量", len(df_raw))
    col2.metric("涵盖语言", "英语 / 俄语")
    col3.metric("类别划分", "真实 / 虚假")
    
    st.markdown("##### 原始数据预览")
    
    df_display = df_raw.rename(columns=NAME_MAP)
    if '真实性标签' in df_display.columns:
        df_display['真实性标签'] = df_display['真实性标签'].replace({'Fake': '虚假', 'Real': '真实'})
    if '语言' in df_display.columns:
        df_display['语言'] = df_display['语言'].replace({'English': '英语', 'Russian': '俄语'})
        
    st.dataframe(df_display, use_container_width=True)
    
    # --- 1.2 共性特征展示 ---
    st.markdown("---")
    st.subheader("显著性特征筛选")
    
    # 将原来的列表说明改为一段连贯的文字
    st.write("""
    通过对原始特征进行跨语言的独立样本 T 检验，我们筛选出了一组在英俄双语环境中均表现出显著差异（P < 0.05）的共性特征。
    这些特征被认为是识别虚假新闻的关键计量指标。
    """)
    
    feature_cols = [c for c in df_raw.columns if c not in ['Language', 'Label', 'Set_Type']]
    
    cols = st.columns(3)
    for i, feature_en in enumerate(feature_cols):
        feature_cn = NAME_MAP.get(feature_en, feature_en)
        cols[i % 3].success(f"{feature_cn}")

    # 关键修改：将原来的“指标说明列表”改为自然段落解释
    st.info("""
    **指标解读：** 这些特征主要反映了文本的复杂度与丰富度。例如，**平均依存距离**可以有效衡量句法结构的复杂程度，数值越小结构越简单；
    **类型符比 (TTR)** 则直观反映了词汇运用的丰富性，较低的 TTR 通常意味着文本用词重复、贫乏。
    此外，引入 **LIX 和 ARI 指数** 则是为了量化文本的可读性，数值越低代表文本越易于阅读，这往往是虚假新闻为了迎合受众而呈现的典型特征。
    """)

# ==========================================
# 5. 模块二：PCA 3D 可视化
# ==========================================
elif option == "PCA 降维分析":
    scroll_to_top()
    st.title("🧊 多维特征空间的降维观测")
    # 用叙述性语言替代功能介绍
    st.markdown("""
    为了直观观测真假新闻在高维特征空间中的分布形态，我们利用 **主成分分析 (PCA)** 将多维计量特征压缩至低维空间。
    这种可视化手段能帮助我们快速识别不同类别文本之间的边界效应。
    """)

    # --- 计算逻辑 ---
    feature_cols = [c for c in df_raw.columns if c not in ['Language', 'Label', 'Set_Type']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_raw[feature_cols])
    pca = PCA(n_components=3)
    components = pca.fit_transform(X_scaled)
    
    df_pca = df_raw.copy()
    df_pca['PC1'], df_pca['PC2'], df_pca['PC3'] = components[:, 0], components[:, 1], components[:, 2]
    df_pca['Group_CN'] = df_pca['Language'].replace({'English':'英语', 'Russian':'俄语'}) + '-' + \
                         df_pca['Label'].replace({'Fake':'虚假', 'Real':'真实'})
    
    color_map = {'英语-虚假': '#d62728', '英语-真实': '#1f77b4', 
                 '俄语-虚假': '#ff7f0e', '俄语-真实': '#2ca02c'}
    symbol_map = {'English': 'circle', 'Russian': 'diamond'}

    # --- 选项卡布局 ---
    tab1, tab2, tab3 = st.tabs(["3D 空间交互", "2D 平面投影", "原文复现对比"])

    with tab1:
        fig_3d = px.scatter_3d(
            df_pca, x='PC1', y='PC2', z='PC3',
            color='Group_CN', symbol='Language',
            color_discrete_map=color_map, symbol_map=symbol_map,
            opacity=0.8, height=600,
            title="PCA 空间分布复现",
            labels={'PC1': '主成分 1', 'PC2': '主成分 2', 'PC3': '主成分 3', 'Group_CN': '类别'}
        )
        fig_3d.update_traces(marker=dict(size=5, line=dict(width=1, color='White')))
        # 去除明显的交互提示，现代UI通常默认用户知道怎么操作，或者写得隐晦点
        st.caption("注：支持拖拽旋转与缩放视角")
        st.plotly_chart(fig_3d, use_container_width=True, key="pca_3d_main")

    with tab2:
        st.markdown("2D 投影能更清晰地展示类簇之间的重叠情况：")
        fig_2d = px.scatter(
            df_pca, x='PC1', y='PC2',
            color='Group_CN', symbol='Language',
            color_discrete_map=color_map, symbol_map=symbol_map,
            opacity=0.8, height=600,
            labels={'PC1': '主成分 1', 'PC2': '主成分 2', 'Group_CN': '类别'}
        )
        fig_2d.update_traces(marker=dict(size=10, line=dict(width=1, color='White')))
        st.plotly_chart(fig_2d, use_container_width=True)

    with tab3:
        st.subheader("复现效果验证")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**📄 原文图表 (Fig. 2)**")
            try:
                img_paper = Image.open("paper_fig2.png") 
                st.image(img_paper, use_container_width=True)
            except:
                st.warning("待上传：paper_fig2.png")
        with col_b:
            st.markdown("**💻 复现结果**")
            st.plotly_chart(fig_3d, use_container_width=True, key="pca_3d_compare")
            
        # 关键修改：用连贯的段落替代 "1. 2. 结论"
        st.success("""
        **对比分析结论**
        
        复现结果与原论文呈现出高度一致的拓扑结构。从空间分布来看，**俄语样本**（橙色/绿色）展现出了清晰的分类边界，说明其真假新闻在计量特征上存在本质差异；
        反观**英语样本**（红色/蓝色），真假两类在空间中呈交织状态，重叠度极高。这一现象佐证了英语虚假新闻在文体风格上具有更高的伪装性，仅凭文体特征难以进行有效区分。
        """)

# ==========================================
# 6. 模块三：聚类分析
# ==========================================
elif option == "跨语言聚类验证":
    st.title("🧬 层次聚类分析")
    st.markdown("""
    本模块采用**无监督学习**方法，在完全剔除标签信息的前提下，考察机器如何依据文本特征对样本进行自发分组。
    这能从侧面验证特征的区分效度。
    """)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📄 原文图表 (Fig. 3)")
        try:
            img_paper_3 = Image.open("paper_fig3.png") 
            st.image(img_paper_3, use_container_width=True)
        except:
            st.warning("待上传：paper_fig3.png")
            
    with col2:
        st.markdown("##### 💻 复现树状图")
        try:
            image_my = Image.open('clustering_dendrogram.png')
            st.image(image_my, use_container_width=True)
        except FileNotFoundError:
            st.warning("计算结果未生成 (clustering_dendrogram.png)")

    st.markdown("---")
    st.subheader("图谱解读")
    
    # 关键修改：彻底重写，用比较自然的学术/分析口吻，不使用列表
    st.write("""
    观察复现生成的树状图，我们可以发现样本首先依据**语言种类**发生了宏观层面的分裂，这表明在计量特征维度上，语言本身的自然差异远大于内容真实性带来的差异。
    
    深入观察子层级可以发现一个有趣的非对称现象：在**俄语分支**下，机器能够较好地将虚假新闻与真实新闻归入不同的子簇，暗示两者在文体上泾渭分明；
    而在**英语分支**下，真假新闻的叶子节点呈现出混沌的混杂状态。这一结果再次印证了PCA分析中的发现，即英语虚假新闻的写作者可能采用了更高级的模仿策略，使其在句法和词汇层面与真实新闻高度趋同。
    """)