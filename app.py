import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud
from collections import Counter
import re
import os

# Sahifa sozlamalari
st.set_page_config(
    page_title="Yangiliklar Tahlili",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, modern dashboard look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #659961;
        margin-top: -15px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #659961;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        text-align: center;
        border-top: 4px solid #F2930B;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #ffffff;
    }
    .metric-label {
        font-size: 1rem;
        color: #ffffff;
        font-weight: 500;
        margin-top: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        width: 100%;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1;
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0;
        height: 48px;
        white-space: nowrap;
        background-color: #ffffff;
        border: 1px solid #659961;
        border-bottom: none;
        border-radius: 4px 4px 0px 0px;
        padding-top: 8px;
        padding-bottom: 8px;
        color: #659961;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab"] p {
        font-weight: bold !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #659961;
        border: none !important;
        border-bottom-color: transparent !important;
        color: #ffffff;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"]:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Data loading function

@st.cache_data(show_spinner=False)
def process_nlp_words(text_list, stopwords_tuple):
    text_corpus = " ".join(text_list).lower()
    text_corpus = re.sub(r"[\`\‘\’\ʻ\ʼ\”\“\02BB\02BC\02B9]", "'", text_corpus)
    words = re.findall(r"[a-z0-9']+", text_corpus)
    
    exceptions = {
        "orqali": "orqali", "qurboni": "qurbon", "qurbonni": "qurbon", "qurbonlari": "qurbon",
        "aholi": "aholi", "aholini": "aholi", "jinoyati": "jinoyat", "jinoyatni": "jinoyat",
        "ayoli": "ayol", "ayolni": "ayol", "bolasi": "bola", "bolani": "bola", "shaxsi": "shaxs"
    }
    suffixes = ['lar', 'ning', 'dagi', 'gacha', 'miz', 'siz', 'imiz', 'ingiz', 'lari']
    
    def q_stemmer(word):
        word = word.strip().strip("'")
        word = re.sub(r"[^a-z0-9']+", "", word)
        if word in exceptions: return exceptions[word]
            
        for _ in range(2): 
            for suff in suffixes:
                if word.endswith(suff) and len(word) > len(suff) + 3:
                    word = word[:-len(suff)]
                    break
        if word == "qurbo": return "qurbon"
        if word == "orqal": return "orqali"
        return word

    filtered_words = []
    for w in words:
        w_clean = q_stemmer(w)
        if len(w_clean) > 3 and w_clean not in stopwords_tuple:
            filtered_words.append(w_clean)
            
    return filtered_words

@st.cache_data
def load_data():
    final_file = "analyzed_merged_news.csv"
    intermediate_classify_violence = os.path.join("gabriel_runs", "classify_violence", "classify_violence_cleaned.csv")
    intermediate_extract = os.path.join("gabriel_runs", "extract_runs", "extraction_results_cleaned.csv")
    raw_file = "merged_news.csv"
    
    df = None
    if os.path.exists(final_file):
        df = pd.read_csv(final_file)
    elif os.path.exists(intermediate_classify_violence):
        st.warning("Yakuniy fayl hali yaratilmagan. Hozircha oraliq ma'lumotlar (2-bosqich yakuni) ko'rsatilmoqda.")
        df = pd.read_csv(intermediate_classify_violence)
    elif os.path.exists(intermediate_extract):
        st.warning("Hozircha faqat 1-bosqich (Extract) tahlili tugallangan.")
        df = pd.read_csv(intermediate_extract)
    elif os.path.exists(raw_file):
        st.warning("Tahlil qilingan fayllar umuman topilmadi. Boshlang'ich ma'lumotlar ko'rsatilmoqda.")
        df = pd.read_csv(raw_file)
    else:
        st.error("Fayl topilmadi. Skriptni ishlating.")
        return None
        
    if df is not None and 'parsed_date' in df.columns:
        df['parsed_date'] = pd.to_datetime(df['parsed_date'], errors='coerce')
        
    if df is not None and 'content length' not in df.columns and 'Content' in df.columns:
        df['content length'] = df['Content'].astype(str).str.len()
        
    return df

col_header_1, col_header_2 = st.columns([5, 2])
with col_header_1:
    st.markdown('<div class="main-header">Yangiliklar tahlili dashboardi</div>', unsafe_allow_html=True)
with col_header_2:
    if os.path.exists("logotip.png"):
        import base64
        with open("logotip.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end; margin-top: -50px;"><img src="data:image/png;base64,{encoded_string}" width="300"></div>',
            unsafe_allow_html=True
        )

df = load_data()

if df is not None:
    has_sentiment = 'title sentiment analyse' in df.columns
    has_content_sentiment = 'content sentiment analyse' in df.columns
    has_violence = 'violence analyse' in df.columns
    has_violence_victim = 'violence victim type' in df.columns
    has_perp = 'violence_perpetrator' in df.columns
    has_news_type = 'news type' in df.columns
    has_content = 'Content' in df.columns

    def clean_sentiment(text):
        t = str(text).lower()
        if 'positive' in t: return 'Positive'
        if 'negative' in t: return 'Negative'
        if 'neutral' in t: return 'Neutral'
        return "Noma'lum"
        
    def clean_array_brackets(text):
        if pd.isna(text): return text
        t = str(text)
        # O'rab turgan list qavslari va qo'shtirnoqlarni olib tashlaymiz
        match = re.search(r"^\[['\"](.*?)['\"]\]$", t.strip())
        if match:
            return match.group(1)
        return t

    if has_sentiment:
        df['title sentiment analyse'] = df['title sentiment analyse'].apply(clean_sentiment)
    if has_content_sentiment:
        df['content sentiment analyse'] = df['content sentiment analyse'].apply(clean_sentiment)
        
    if has_violence_victim:
        df['violence victim type'] = df['violence victim type'].apply(clean_array_brackets)
    if 'violence location' in df.columns:
        df['violence location'] = df['violence location'].apply(clean_array_brackets)
    if has_news_type:
        df['news type'] = df['news type'].apply(clean_array_brackets)
    if has_violence:
        df['violence analyse'] = df['violence analyse'].apply(clean_array_brackets)

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("🔍 Filtrlash Sozlamalari")
    
    # Yuklab olish formati uchun sozlama
    export_format = st.sidebar.selectbox(
        "Rasmlar Yuklash yuklash formati (Barcha grafiklar uchun):", 
        ["png", "jpeg", "svg", "webp"], 
        index=1,
        help="O'zgartirsangiz, barcha grafiklarning yuqori o'ng burchagidagi yuklab olish tugmasi mos ravishda ishlaydi."
    )
    
    # Barcha grafiklar yuklab olinishi uchun konfigratsiya ro'yxati
    dl_config = {
        'toImageButtonOptions': {
            'format': export_format,
            'filename': f'diagramma_grafik_{export_format}',
            'height': 800,
            'width': 1200,
            'scale': 2
        },
        'displayModeBar': True
    }
    st.sidebar.markdown("*(Yuklab olish uchun har bir grafikdagi Kamera 📷 tugmasiga bosing)*")
    st.sidebar.markdown("---")

    if 'Source' in df.columns:
        sources = df['Source'].dropna().unique().tolist()
        sources.insert(0, "Barchasi")
        selected_sources = st.sidebar.multiselect("📰 Manba (Source)", options=sources, default=["Barchasi"])
        if "Barchasi" not in selected_sources and selected_sources:
            df = df[df['Source'].isin(selected_sources)]

    if 'parsed_date' in df.columns and not df['parsed_date'].isna().all():
        min_date = df['parsed_date'].min().date()
        max_date = df['parsed_date'].min() if pd.isna(df['parsed_date'].max()) else df['parsed_date'].max().date()
        date_range = st.sidebar.date_input(
            "📅 Sana oralig'i", value=(min_date, max_date), min_value=min_date, max_value=max_date
        )
        if len(date_range) == 2:
            df = df[(df['parsed_date'].dt.date >= date_range[0]) & (df['parsed_date'].dt.date <= date_range[1])]

    if 'content length' in df.columns and not df['content length'].isna().all():
        min_len = int(df['content length'].min())
        max_len = int(df['content length'].max())
        if max_len > min_len:
            len_range = st.sidebar.slider(
                "Matn Uzunligi (Belgilar soni)",
                min_value=min_len,
                max_value=max_len,
                value=(min_len, max_len)
            )
            df = df[(df['content length'] >= len_range[0]) & (df['content length'] <= len_range[1])]

    if has_sentiment:
        sentiments = df['title sentiment analyse'].dropna().unique().tolist()
        if "Barchasi" not in sentiments: sentiments.insert(0, "Barchasi")
        selected_sentiments = st.sidebar.multiselect("🎭 Sarlavha Hissiyoti", options=sentiments, default=["Barchasi"])
        if "Barchasi" not in selected_sentiments and selected_sentiments:
            df = df[df['title sentiment analyse'].isin(selected_sentiments)]
            
    if has_violence:
        violence_status = df['violence analyse'].dropna().unique().tolist()
        if "Barchasi" not in violence_status: violence_status.insert(0, "Barchasi")
        selected_violence = st.sidebar.multiselect("Zo'ravonlik Holati", options=violence_status, default=["Barchasi"])
        if "Barchasi" not in selected_violence and selected_violence:
            df = df[df['violence analyse'].isin(selected_violence)]
            
    if has_news_type:
        news_types = df['news type'].dropna().unique().tolist()
        if "Barchasi" not in news_types: news_types.insert(0, "Barchasi")
        selected_news_types = st.sidebar.multiselect("Yangilik Turi", options=news_types, default=["Barchasi"])
        if "Barchasi" not in selected_news_types and selected_news_types:
            df = df[df['news type'].isin(selected_news_types)]
            
    if 'violence location' in df.columns:
        locations = df['violence location'].dropna().unique().tolist()
        if "Barchasi" not in locations: locations.insert(0, "Barchasi")
        selected_locations = st.sidebar.multiselect("Mintaqa (Hudud)", options=locations, default=["Barchasi"])
        if "Barchasi" not in selected_locations and selected_locations:
            df = df[df['violence location'].isin(selected_locations)]

    st.sidebar.markdown('---')
    st.sidebar.info(f"Ko'rsatilayotgan yangiliklar: **{len(df)}** ta")

    # --- MAIN CONTENT METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Jami yangiliklar</div></div>', unsafe_allow_html=True)
    with col2:
        sources_count = df['Source'].nunique() if 'Source' in df.columns else 0
        st.markdown(f'<div class="metric-card"><div class="metric-value">{sources_count}</div><div class="metric-label">Faol manbalar</div></div>', unsafe_allow_html=True)
    with col3:
        if has_violence:
            v_count = len(df[df['violence analyse'].astype(str).str.lower().str.contains('bor', na=False)])
        else: v_count = "-"
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ef4444;">{v_count}</div><div class="metric-label">Zo\'ravonlik qayd etilgan yangiliklar</div></div>', unsafe_allow_html=True)
    with col4:
        avg_len = f"{int(df['content length'].mean())} " if 'content length' in df.columns and not df['content length'].isna().all() else "-"
        st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_len}</div><div class="metric-label">O\'rtacha matn uzunligi (belgi)</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)

    if 'parsed_date' in df.columns and 'Source' in df.columns and not df['parsed_date'].isna().all():
        st.subheader("Manbalar bo'yicha davriy yangiliklar yuklanish grafigi")
        main_ts_df = df.groupby([df['parsed_date'].dt.date, 'Source']).size().reset_index(name='Soni')
        main_ts_df.rename(columns={'parsed_date': 'Sana'}, inplace=True)
        fig_main_ts = px.line(
            main_ts_df, x='Sana', y='Soni', color='Source',
            title="Kunlik chop etilgan yangiliklar soni (Kanal va Saytlar kesimida)",
            markers=True, line_shape="spline",
            color_discrete_sequence=['#659961', '#F2930B', '#A6A6A6', '#3475B5', '#D2A14E', '#8E6C8A', '#528E8C', '#B56B5D']
        )
        st.plotly_chart(fig_main_ts, use_container_width=True, config=dl_config)
        st.markdown("---")

    tabs = st.tabs([
        "Umumiy hissiyotlar (Sentiment analyse)", 
        "Ayollar va bolalarga nisbatan zo‘ravonlik", 
        "Yangiliklar klassifikatsiyasi", 
        "Vaqt dinamikasi & statistika",
        "NLP so'zlar bog'lanishi"
    ])
    
    # -----------------------------
    # TAB 1: UMUMIY HISSIYOTLAR
    # -----------------------------
    with tabs[0]:
        st.subheader("Hissiyot Tahlili (Sentiment Analysis)")
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            if has_sentiment:
                sc = df['title sentiment analyse'].value_counts().reset_index()
                sc.columns = ['Hissiyot', 'Soni']
                fig_s = px.pie(sc, values='Soni', names='Hissiyot', title="Sarlavhalar Hissiyoti",
                               color='Hissiyot', color_discrete_map={'Positive': '#659961', 'Neutral': '#A6A6A6', 'Negative': '#F2930B'}, hole=0.4)
                fig_s.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_s, use_container_width=True, config=dl_config)
            else: st.info("Sarlavha hissiyoti tahlili mavjud emas.")
        with fig_col2:
            if has_content_sentiment:
                cc = df['content sentiment analyse'].value_counts().reset_index()
                cc.columns = ['Hissiyot', 'Soni']
                cc['Ulush (%)'] = (cc['Soni'] / cc['Soni'].sum() * 100).round(1)
                fig_c = px.bar(cc, x='Hissiyot', y='Soni', title="Matn / Content Hissiyoti",
                               color='Hissiyot', color_discrete_map={'Positive': '#659961', 'Neutral': '#A6A6A6', 'Negative': '#F2930B'}, 
                               text=cc.apply(lambda row: f"{row['Soni']} ta ({row['Ulush (%)']}%)", axis=1))
                st.plotly_chart(fig_c, use_container_width=True, config=dl_config)
            else: st.info("Matn hissiyoti tahlili mavjud emas.")

    # -----------------------------
    # TAB 2: ZO'RAVONLIK HOLATLARI
    # -----------------------------
    with tabs[1]:
        st.subheader("Zo'ravonlik holatlari")
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            if has_violence:
                vc = df['violence analyse'].value_counts().reset_index()
                vc.columns = ['Holat', 'Soni']
                fig_v = px.bar(vc, x='Holat', y='Soni', title="Zo'ravonlik Holati Darajasi", color='Holat', color_discrete_map={'Bor': '#F2930B', 'Yo\'q': '#659961'}, text='Soni')
                st.plotly_chart(fig_v, use_container_width=True, config=dl_config)
        with vcol2:
            if has_violence_victim:
                v_df = df[~df['violence victim type'].astype(str).str.contains('Yo\'q', case=False, na=False)]
                if not v_df.empty:
                    vic = v_df['violence victim type'].value_counts().reset_index()
                    vic.columns = ['Qurbon', 'Soni']
                    fig_vic = px.pie(vic, values='Soni', names='Qurbon', title="Qurbonlar Toifasi", hole=0.3, color_discrete_sequence=['#F2930B', '#A6A6A6', '#659961', '#3475B5', '#D2A14E'])
                    st.plotly_chart(fig_vic, use_container_width=True, config=dl_config)
                else: st.info("Qurbonlar haqida ma'lumot topilmadi.")
        if has_perp:
            p_df = df[~df['violence_perpetrator'].astype(str).str.contains('Yo\'q', case=False, na=False)]
            if not p_df.empty:
                perp = p_df['violence_perpetrator'].value_counts().reset_index().head(12)
                perp.columns = ['Aybdor', 'Soni']
                fig_p = px.bar(perp, y='Aybdor', x='Soni', orientation='h', title="Asosiy Aybdorlar/Jinoyatchilar (Top-12)", color='Soni', color_continuous_scale=['#A6A6A6', '#F2930B'])
                fig_p.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_p, use_container_width=True, config=dl_config)



    # -----------------------------
    # TAB 3: KLASSIFIKATSIYA
    # -----------------------------
    with tabs[2]:
        st.subheader("Yangilik Maqolalari Toifalari (News Types)")
        ncol1, ncol2 = st.columns(2)
        with ncol1:
            if has_news_type:
                nc = df['news type'].value_counts().reset_index()
                nc.columns = ['Tur', 'Soni']
                fig_n = px.pie(nc, values='Soni', names='Tur', title="Yangiliklarning Umumiy Taqsimoti", hole=0.5, color_discrete_sequence=['#659961', '#F2930B', '#A6A6A6', '#3475B5', '#D2A14E'])
                fig_n.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_n, use_container_width=True, config=dl_config)
        with ncol2:
            if has_news_type and has_sentiment:
                cross_df = pd.crosstab(df['news type'], df['title sentiment analyse'], normalize='index').reset_index()
                if not cross_df.empty:
                    melted_pct = cross_df.melt(id_vars='news type', var_name='Hissiyot', value_name='Ulush (%)')
                    melted_pct['Ulush (%)'] = (melted_pct['Ulush (%)'] * 100).round(1)
                    
                    counts_df = pd.crosstab(df['news type'], df['title sentiment analyse']).reset_index().melt(id_vars='news type', var_name='Hissiyot', value_name='Soni')
                    melted_pct['Soni'] = counts_df['Soni']
                    
                    fig_cross = px.bar(melted_pct, x='news type', y='Ulush (%)', color='Hissiyot', 
                                       title="Yo'nalishlar bo'yicha Hissiyotlar (100% Ulush)", 
                                       color_discrete_map={'Positive': '#659961', 'Neutral': '#A6A6A6', 'Negative': '#F2930B'}, 
                                       text=melted_pct.apply(lambda row: f"{row['Ulush (%)']}%" if row['Ulush (%)'] >= 3 else "", axis=1),
                                       hover_data={'Soni': True, 'Ulush (%)': True, 'news type': False})
                    fig_cross.update_traces(textposition='inside')
                    fig_cross.update_layout(barmode='stack', uniformtext_minsize=10, uniformtext_mode='hide')
                    st.plotly_chart(fig_cross, use_container_width=True, config=dl_config)

        # SANKEY DIAGRAM (QAYTADAN TEKSHIRILGAN)
        if 'Source' in df.columns and has_news_type and has_sentiment:
            st.markdown("---")
            st.subheader("Ma'lumotlar oqimi: Manba → Yo'nalish → Hissiyot (Sankey Diagram)")
            st.write("Qaysi manbalardagi xabarlar qaysi yo'nalish bo'yicha e'lon qilinib, qanday kayfiyatga o'tayotganligini ko'rsatuvchi interaktiv oqim.")
            
            # Xavfsiz nusxa olish va tozalash
            s_df = df.copy()
            s_df['Source'] = s_df['Source'].fillna("Noma'lum Manba").astype(str)
            s_df['news type'] = s_df['news type'].fillna("Noma'lum Tur").astype(str)
            s_df['title sentiment analyse'] = s_df['title sentiment analyse'].fillna("Noma'lum Hissiyot").astype(str)
            
            # 1. Source to News Type
            src_nt = s_df.groupby(['Source', 'news type']).size().reset_index(name='value')
            src_nt['source_col'] = src_nt['Source'] + ' (Manba)'
            src_nt['target_col'] = src_nt['news type'] + ' (Tur)'
            
            # 2. News Type to Sentiment
            nt_sent = s_df.groupby(['news type', 'title sentiment analyse']).size().reset_index(name='value')
            nt_sent['source_col'] = nt_sent['news type'] + ' (Tur)'
            nt_sent['target_col'] = nt_sent['title sentiment analyse'] + ' (Hissiyot)'
            
            links_df = pd.concat([
                src_nt[['source_col', 'target_col', 'value']], 
                nt_sent[['source_col', 'target_col', 'value']]
            ], ignore_index=True)
            
            # Unikal tugunlarni ajratish
            all_nodes = list(pd.unique(links_df[['source_col', 'target_col']].values.ravel('K')))
            node_mapping = {node: i for i, node in enumerate(all_nodes)}
            
            source_indices = links_df['source_col'].map(node_mapping).tolist()
            target_indices = links_df['target_col'].map(node_mapping).tolist()
            values = links_df['value'].tolist()
            
            manba_nodes = [n for n in all_nodes if ' (Manba)' in n]
            tur_nodes = [n for n in all_nodes if ' (Tur)' in n]
            
            # Ranglarni va tartibini belgilash (x, y koordinatalari bilan)
            node_colors = []
            x_pos = []
            y_pos = []
            for n in all_nodes:
                if 'Positive' in n: 
                    node_colors.append('#659961')
                    x_pos.append(0.99); y_pos.append(0.1)
                elif 'Neutral' in n: 
                    node_colors.append('#A6A6A6')
                    x_pos.append(0.99); y_pos.append(0.5)
                elif 'Negative' in n: 
                    node_colors.append('#F2930B')
                    x_pos.append(0.99); y_pos.append(0.9)
                elif ' (Manba)' in n: 
                    node_colors.append('#3475B5')
                    x_pos.append(0.01)
                    if len(manba_nodes) == 1: y_pos.append(0.5)
                    else: y_pos.append(0.01 + 0.98 * (manba_nodes.index(n) / (len(manba_nodes) - 1)))
                elif ' (Tur)' in n:
                    node_colors.append('#3475B5')
                    x_pos.append(0.5)
                    if len(tur_nodes) == 1: y_pos.append(0.5)
                    else: y_pos.append(0.01 + 0.98 * (tur_nodes.index(n) / (len(tur_nodes) - 1)))
                else: 
                    node_colors.append('#3475B5')
                    x_pos.append(0.5); y_pos.append(0.5)

            try:
                fig_sankey = go.Figure(data=[go.Sankey(
                    arrangement="snap",
                    textfont=dict(color="black", size=13),
                    node = dict(
                      pad = 20,
                      thickness = 25,
                      line = dict(color = "white", width = 0.5),
                      label = [f"<b>{n.replace(' (Manba)', '').replace(' (Tur)', '').replace(' (Hissiyot)', '')}</b>" for n in all_nodes],
                      color = node_colors,
                      x = x_pos,
                      y = y_pos
                    ),
                    link = dict(
                      source = source_indices,
                      target = target_indices,
                      value = values,
                      color = "rgba(166, 166, 166, 0.4)" 
                    ))])
                
                fig_sankey.update_layout(title_text="Axborot tarqalish hamda kayfiyat o'zgarish oqimi", font_size=12, height=500)
                st.plotly_chart(fig_sankey, use_container_width=True, config=dl_config)
            except Exception as e:
                st.error("Diagrammani chizishda ichki ma'lumotlar bilan bog'liq xatolik yuz berdi. Iltimos tekshiring.")

    # -----------------------------
  # TAB 4: DINAMIKA
    # -----------------------------
    with tabs[3]:
        st.subheader("Statistika")
        if 'parsed_date' in df.columns and not df['parsed_date'].isna().all():
            if has_news_type:
                time_df = df.groupby([df['parsed_date'].dt.date, 'news type']).size().reset_index(name='Soni')
                fig_time = px.line(time_df, x='parsed_date', y='Soni', color='news type', title="Turkumlar Bo'yicha O'sish", markers=True, color_discrete_sequence=['#659961', '#F2930B', '#A6A6A6', '#3475B5', '#D2A14E'])
                st.plotly_chart(fig_time, use_container_width=True, config=dl_config)
        if 'content length' in df.columns and 'Source' in df.columns and has_content_sentiment:
            st.markdown("---")
            st.subheader("Maqolalar hajmi dinamikasi: Manba → Hajm → Matn Hissiyoti")
            
            l_df = df.copy()
            l_df['Source'] = l_df['Source'].fillna("Noma'lum Manba").astype(str)
            l_df['content sentiment analyse'] = l_df['content sentiment analyse'].fillna("Noma'lum Hissiyot").astype(str)
            
            bins = [-1, 500, 1500, 3000, float('inf')]
            labels = ['Qisqa (0-500)', "O'rtacha (500-1500)", "Uzun (1500-3000)", "Juda Uzun (3000+)"]
            l_df['Uzunlik'] = pd.cut(l_df['content length'], bins=bins, labels=labels).astype(str)
            
            # 1. Source to Uzunlik
            src_len = l_df.groupby(['Source', 'Uzunlik']).size().reset_index(name='value')
            src_len['source_col'] = src_len['Source'] + ' (Manba)'
            src_len['target_col'] = src_len['Uzunlik'] + ' (Hajm)'
            
            # 2. Uzunlik to Sentiment
            len_sent = l_df.groupby(['Uzunlik', 'content sentiment analyse']).size().reset_index(name='value')
            len_sent['source_col'] = len_sent['Uzunlik'] + ' (Hajm)'
            len_sent['target_col'] = len_sent['content sentiment analyse'] + ' (Hissiyot)'
            
            links_l_df = pd.concat([
                src_len[['source_col', 'target_col', 'value']], 
                len_sent[['source_col', 'target_col', 'value']]
            ], ignore_index=True)
            
            l_nodes = list(pd.unique(links_l_df[['source_col', 'target_col']].values.ravel('K')))
            l_node_mapping = {node: i for i, node in enumerate(l_nodes)}
            
            l_source_indices = links_l_df['source_col'].map(l_node_mapping).tolist()
            l_target_indices = links_l_df['target_col'].map(l_node_mapping).tolist()
            l_values = links_l_df['value'].tolist()
            
            l_manba_nodes = [n for n in l_nodes if ' (Manba)' in n]
            
            l_node_colors = []
            l_x_pos, l_y_pos = [], []
            for n in l_nodes:
                if 'Positive' in n: 
                    l_node_colors.append('#659961')
                    l_x_pos.append(0.99); l_y_pos.append(0.1)
                elif 'Neutral' in n: 
                    l_node_colors.append('#A6A6A6')
                    l_x_pos.append(0.99); l_y_pos.append(0.5)
                elif 'Negative' in n: 
                    l_node_colors.append('#F2930B')
                    l_x_pos.append(0.99); l_y_pos.append(0.9)
                elif ' (Manba)' in n: 
                    l_node_colors.append('#3475B5')
                    l_x_pos.append(0.01)
                    if len(l_manba_nodes) == 1: l_y_pos.append(0.5)
                    else: l_y_pos.append(0.01 + 0.98 * (l_manba_nodes.index(n) / (len(l_manba_nodes) - 1)))
                elif 'Qisqa' in n:
                    l_node_colors.append('#D2A14E')
                    l_x_pos.append(0.5); l_y_pos.append(0.1)
                elif "O'rtacha" in n:
                    l_node_colors.append('#D2A14E')
                    l_x_pos.append(0.5); l_y_pos.append(0.4)
                elif 'Uzun' in n and 'Juda' not in n:
                    l_node_colors.append('#D2A14E')
                    l_x_pos.append(0.5); l_y_pos.append(0.7)
                elif 'Juda Uzun' in n:
                    l_node_colors.append('#D2A14E')
                    l_x_pos.append(0.5); l_y_pos.append(0.95)
                else: 
                    l_node_colors.append('#D2A14E')
                    l_x_pos.append(0.5); l_y_pos.append(0.5)
            
            try:
                fig_sankey_len = go.Figure(data=[go.Sankey(
                    arrangement="snap",
                    textfont=dict(color="black", size=13),
                    node = dict(
                      pad = 20, thickness = 25,
                      line = dict(color = "white", width = 0.5),
                      label = [f"<b>{n.replace(' (Manba)', '').replace(' (Hajm)', '').replace(' (Hissiyot)', '')}</b>" for n in l_nodes],
                      color = l_node_colors,
                      x = l_x_pos, y = l_y_pos
                    ),
                    link = dict(
                      source = l_source_indices, target = l_target_indices, value = l_values,
                      color = "rgba(166, 166, 166, 0.4)" 
                    )
                )])
                fig_sankey_len.update_layout(title_text="Yangilik hajmining manbalar va hissiyotlar bo'yicha tarqalishi", font_size=12, height=500)
                st.plotly_chart(fig_sankey_len, use_container_width=True, config=dl_config)
            except Exception as e:
                st.error("Dinamika diagrammasini tuzishda xatolik yuz berdi.")
        elif 'content length' in df.columns:
            fig_len = px.histogram(df, x='content length', nbins=50, title="Maqolalar Uzunligi Taqsimoti (Belgilar Soni)", color_discrete_sequence=['#659961'])
            st.plotly_chart(fig_len, use_container_width=True, config=dl_config)

    # -----------------------------
    # TAB 5: SO'ZLAR (WORD CLOUD, TREE, NETWORK) -> FULLY REVISED
    # -----------------------------
    with tabs[4]:
        st.subheader("🗣️ Matn So'zlarini Chuqur Tahlil Qilish (NLP)")
        st.write("Bu yerda O'zbek tilidagi qoshimcha kelishiklar yoki yordamchi so'zlar avtomatik tozalangan holda ko'rsatiladi.")
        
        if has_content:
            # STOP WORDS KENGAYTIRILDI
            uzbek_stopwords = {
                "va", "bilan", "uchun", "ham", "shu", "bu", "esa", "qildi", "deb", "o'z", "bo'ldi", 
                "haqida", "kerak", "edi", "yo'q", "bo'yicha", "undan", "mumkin", "kabi", "orqali", 
                "ega", "ular", "hamda", "yoki", "ammo", "lekin", "biroq", "ning", "ni", "ga", "da", "dan",
                "faqat", "hatto", "chunki", "agar", "garchi", "shuning", "balki", "bo'lsa", "o'zi", 
                "ikki", "bir", "deb", "ekan", "biz",'va', 'bu', 'bilan', 'uchun', 'ham', 'deb', 'o‘z', 'bir', 'bor', 'edi', 'esa', 'kabi', 
                'shu', 'u', 'ushbu', 'lekin', 'ammo', 'chunki', 'bo‘lib', 'bo‘lgan', 'yili', 'so‘ng', 
                'yil', 'faqat', 'qildi', 'qilish', 'mumkin', 'kerak', 'qilingan', 'bo‘ladi', 'etildi', 
                'etadi', 'ekan', 'avval', 'keyin', 'ular', 'biz', 'siz', 'men', 'u', 'bu', 'o‘sha',
                'haqida', 'tomonidan', 'bo‘yicha', 'hamda', 'uni', 'uning', 'unga', 'unda', 'undan',
                'eng', 'juda', 'yana', 'endi', 'mana', 'barcha', 'boshqa', 'har', 'hamma', 'edi', 'edi',
                'yoki', 'jumladan', 'bo‘lsa', 'ko‘ra', 'ta', 'ya\'ni', 'yani', 'balki', 'holatda', 'sababli',
                'tufayli', 'biri', 'ko‘p', 'yilgi', 'yangi', 'kun', 'oy', 'bo‘yi', 'o‘zi', 
                'o‘zining', 'o‘ziga', 'o‘zini', 'o‘zidan', 'haq', 'hali', 'endi', 'o', 'g', 'n', 's', 'm', 'd', 'k', 'z',
                'esa', 'emas', 'aholda', 'aynan', 'demak', 'gohi', 'gohida', 'kerakli', 'keraksiz', 'aniq', 'agar', 'shuningdek',
                'biroq','buni','haqda','orqali','degan', 'nisbatan','qilib','dedi','bo‘l',"berish","qila","etil",'bo‘la','olib',
                'etgan','olgan','oldi','etdi','bo‘lish','qilish','ko‘ra','tashqari','bundan','bo‘ladi','etish','bunday','hech',
                'berdi','mazkur','qilgan','o‘sha',"bo'lgan",'yuzasidan',"ko'ra","o'sha","so'ng","bo'lib","ma'lum", 'soidr', 'olish'
            }
                               
            with st.spinner("Matnlar o'zbek tilida chuqur tahlil qilinib tozalanmoqda. Kuting..."):
                text_list = df['Content'].dropna().astype(str).tolist()[:3000]
                filtered_words = process_nlp_words(text_list, tuple(uzbek_stopwords))
                
                # To keep uzbek_stemmer available for single inputs in Word Tree/Network Graphs
                def uzbek_stemmer(word):
                    word = word.strip().strip("'")
                    word = re.sub(r"[^a-z0-9']+", "", word)
                    exceptions = {"orqali": "orqali", "qurboni": "qurbon", "qurbonni": "qurbon", "qurbonlari": "qurbon",
                                  "aholi": "aholi", "aholini": "aholi", "jinoyati": "jinoyat", "jinoyatni": "jinoyat",
                                  "ayoli": "ayol", "ayolni": "ayol", "bolasi": "bola", "bolani": "bola", "shaxsi": "shaxs"}
                    if word in exceptions: return exceptions[word]
                    suffixes = ['lar', 'ning', 'dagi', 'gacha', 'miz', 'siz', 'imiz', 'ingiz', 'lari']
                    for _ in range(2): 
                        for suff in suffixes:
                            if word.endswith(suff) and len(word) > len(suff) + 3:
                                word = word[:-len(suff)]
                                break
                    if word == "qurbo": return "qurbon"
                    if word == "orqal": return "orqali"
                    return word
                
                # ------ WORD TREE & TABLE YUKLASH ------
                st.markdown("### Interaktiv So'zlar Bog'lanishi (Word Tree)")
                st.write("Qaysidir asosiy so'zni yozing o'sha so'z va uning atrofidagi ulanishlar Daraxt (Tree) shaklida topiladi.")
                
                wtree_col1, wtree_col2 = st.columns([1, 2])
                with wtree_col1:
                    target_word_raw = st.text_input("Izlanayotgan so'zni kiriting (masalan: zo'ravonlik):", "zo'ravonlik")
                    tree_depth = st.slider("Daraxt chuqurligi (Ulangan so'zlar zanjiri uzunligi):", 1, 3, 1)
                    branch_count = st.slider("Shoxobchalar (Yon tarmoqlar) miqdori:", 3, 20, 10)
                    
                    target_word_clean = re.sub(r"[\`\‘\’\ʻ\ʼ\”\“\02BB\02BC\02B9]", "'", target_word_raw.lower())
                    target_word = uzbek_stemmer(target_word_clean)
                    
                if len(filtered_words) > 1 and target_word:
                    # Gorizontal Tree yaratish mantiqi
                    class PathTree:
                        def __init__(self, name):
                            self.name = name
                            self.children = {}
                            self.count = 0
                            self.x = 0
                            self.y = 0

                    root = PathTree(target_word)
                    for i in range(len(filtered_words) - tree_depth):
                        if filtered_words[i] == target_word:
                            root.count += 1
                            curr_node = root
                            for d in range(1, tree_depth + 1):
                                nxt_word = filtered_words[i+d]
                                if nxt_word not in curr_node.children:
                                    curr_node.children[nxt_word] = PathTree(nxt_word)
                                curr_node.children[nxt_word].count += 1
                                curr_node = curr_node.children[nxt_word]

                    def prune_tree(node, max_branches):
                        if not node.children:
                            return
                        sorted_children = sorted(node.children.values(), key=lambda x: x.count, reverse=True)[:max_branches]
                        node.children = {c.name: c for c in sorted_children}
                        for c in node.children.values():
                            prune_tree(c, max_branches)

                    if root.count > 0:
                        prune_tree(root, branch_count)

                        x_coords, y_coords, node_texts, node_sizes = [], [], [], []
                        edge_x, edge_y, flat_edges = [], [], []

                        def set_y(node, level, current_y):
                            node.x = level
                            if not node.children:
                                node.y = current_y
                                return current_y + 1
                            start_y = current_y
                            for c in node.children.values():
                                current_y = set_y(c, level + 1, current_y)
                            node.y = (start_y + current_y - 1) / 2.0
                            return current_y

                        set_y(root, 0, 0)

                        def extract_trace_data(node, parent_name=None):
                            x_coords.append(node.x)
                            y_coords.append(node.y)
                            node_texts.append(f"{node.name} ({node.count})")
                            node_sizes.append(20 + min(node.count * 1.5, 40))
                            
                            if parent_name:
                                flat_edges.append((parent_name, node.name, node.count, node.x))

                            for c in node.children.values():
                                edge_x.extend([node.x, c.x, None])
                                edge_y.extend([node.y, c.y, None])
                                extract_trace_data(c, node.name)

                        extract_trace_data(root)

                        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color='#adb5bd'), hoverinfo='none', mode='lines')
                        node_trace = go.Scatter(
                            x=x_coords, y=y_coords, mode='markers+text',
                            textposition="middle right", text=node_texts, hoverinfo='text',
                            textfont=dict(color='#0b7285', size=13),
                            marker=dict(showscale=False, color='#e3fafc', size=node_sizes, line=dict(width=2, color='#15aabf'))
                        )

                        fig_tree = go.Figure(data=[edge_trace, node_trace],
                                          layout=go.Layout(
                                            title=f"'{target_word_raw}' so'zining Gorizontal Daraxti (Chuqurlik: {tree_depth})",
                                            showlegend=False, hovermode='closest', plot_bgcolor='white',
                                            margin=dict(b=20, l=20, r=100, t=40),
                                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, tree_depth + 1.2]),
                                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                                          ))
                        st.plotly_chart(fig_tree, use_container_width=True, config=dl_config)

                        # JADVAL YUKLAB OLISH QISMI
                        st.markdown(f"**Daraxt bo'g'inlari ulanish joriy jadvali:**")
                        bg_df = pd.DataFrame(flat_edges, columns=['Asosiy Soz', 'Ulanuvchi Soz', 'Uchrashish Soni', 'Daraxt Bosqichi'])
                        if not bg_df.empty:
                            st.dataframe(bg_df, use_container_width=True)
                            csv_table = bg_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Daraxt Jadvalini Yuklab Olish (CSV)",
                                data=csv_table,
                                file_name=f"{target_word_clean}_treedata.csv",
                                mime="text/csv",
                            )
                    else:
                        st.warning(f"Matn ichida ushbu '{target_word_raw}' so'zi yetarli ulanishlar bilan topilmadi.")
                
                st.markdown("---")
                # ------ MULTIPLE WORDS NETWORK GRAPH ------
                st.markdown("### So'zlarning o'zaro Tarmoq Chiziqli Grafiki (Network Graph)")
                st.write("Bir nechta alohida so'zlarni tanlang. Ularning qanchalik ko'p birga uchrashgani chiziqlar ustida raqamlangan bo'ladi.")
                
                # Eng ko'p uchragan 200 ta so'zni tanlash (Network bog'lanishining chuqurligini ushlab turish uchun kerak)
                top_words_all = [w[0] for w in Counter(filtered_words).most_common(200)]
                
                net_words_input = st.text_input(
                    "Grafikka joylash uchun so'zlarni vergul bilan ajratib yozing (ixtiyoriy so'zlar):", 
                    value="sud, jinoyat, sudya, qamoq, jazo"
                )
                
                net_words = []
                if net_words_input:
                    raw_words = [w.strip() for w in net_words_input.split(',')]
                    for rw in raw_words:
                        if rw:
                            # Kiritilgan so'zni xuddi matndek tozalab o'chirgichdan o'tkazamiz
                            cw = re.sub(r"[\`\‘\’\ʻ\ʼ\”\“\02BB\02BC\02B9]", "'", rw.lower())
                            cw = uzbek_stemmer(cw)
                            if cw:
                                net_words.append(cw)
                
                if net_words:
                    # Collect connections where both words exist in context of each other (window of 5 words)
                    nw_bigrams = []
                    window_size = 15
                    for i in range(len(filtered_words)):
                        if filtered_words[i] in net_words:
                            for j in range(1, window_size + 1):
                                if i + j < len(filtered_words) and filtered_words[i + j] in net_words:
                                    pair = tuple(sorted([filtered_words[i], filtered_words[i + j]]))
                                    nw_bigrams.append(pair)
                             
                    nw_counts = Counter(nw_bigrams).most_common(50)
                    
                    G = nx.Graph()
                    for pair, count in nw_counts:
                        if count > 1: # at least 2 co-occurences
                            G.add_edge(pair[0], pair[1], weight=count)
                            
                    # Graphviz or NetworkX Plotly drawing
                    if len(G.nodes) > 0:
                        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
                        
                        edge_x, edge_y, m_x, m_y, lines_text = [], [], [], [], []
                        for edge in G.edges(data=True):
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                            
                            m_x.append((x0 + x1) / 2)
                            m_y.append((y0 + y1) / 2)
                            lines_text.append(f"{edge[0]}-{edge[1]}: {edge[2]['weight']} marta")
                            
                        # Chiziqlar 
                        edge_trace = go.Scatter(
                            x=edge_x, y=edge_y, line=dict(width=1, color='#aaa'),
                            hoverinfo='none', mode='lines'
                        )
                        
                        # Chiziqlar ustida soni
                        edge_label_trace = go.Scatter(
                            x=m_x, y=m_y, mode='text',
                            text=[str(e[2]['weight']) for e in G.edges(data=True)],
                            textposition='top center',
                            hovertext=lines_text,
                            hoverinfo='text',
                            textfont=dict(color='darkred', size=11)
                        )

                        node_x, node_y, node_text, node_size = [], [], [], []
                        for node in G.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                            node_text.append(node)
                            node_size.append( G.degree(node)*5 + 15 )

                        # Tugun (Pufakcha) lar tasviri
                        node_trace = go.Scatter(
                            x=node_x, y=node_y, mode='markers+text',
                            textposition="top center",
                            hoverinfo='text',
                            marker=dict(
                                showscale=True, colorscale='Viridis', color=[G.degree(n) for n in G.nodes()],
                                size=node_size, line_width=2),
                            text=node_text
                        )
                        
                        fig_net = go.Figure(data=[edge_trace, edge_label_trace, node_trace],
                                         layout=go.Layout(
                                            title='', showlegend=False, hovermode='closest',
                                            margin=dict(b=20,l=5,r=5,t=40),
                                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                                         )
                        st.plotly_chart(fig_net, use_container_width=True, config=dl_config)
                    else:
                        st.info("Tanlangan so'zlar guruhining matinda aniq birga kelgan (kesishish) xolati aniqlanmadi. Boshqa so'zlar bilan sinab ko'rsangiz bo'ladi.")
                
                st.markdown("---")
                
                # WORD CLOUD SO'NGIDA QOLGANI
                st.markdown("#### Hamma matndan yig'ilgan so'zlar buluti (Word Cloud)")
                if filtered_words:
                    word_freq = Counter(filtered_words)
                    wc = WordCloud(width=800, height=350, background_color='white', colormap='tab20', max_words=100)
                    wc.generate_from_frequencies(word_freq)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)

        else:
            st.info("Bu bo'lim matnlarni o'qish uchun yaratilgan. Bizada matn topilmadi.")

    # --- DATA TABLE ---
    st.markdown("---")
    st.subheader("Batafsil ma'lumotlar bazasini ko'rish")
    display_cols = list(df.columns)
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True, height=400)

else:
    st.warning("Ma'lumotlar bazasi bilan ishlashda kamchilik. Iltimos jadvallar mavjudligini tekshiring.")
