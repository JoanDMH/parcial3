import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import os

# Directorio base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(BASE_DIR) == 'app':
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
else:
    PROJECT_ROOT = BASE_DIR

# ── CONFIGURACIÓN DE LA PÁGINA ───────────────────────────────────────────────
st.set_page_config(
    page_title="Recomendador de Medicamentos — DrugLib CF",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado para un look premium oscuro y moderno
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f1219 0%, #151922 100%);
        color: #e2e8f0;
    }
    h1, h2, h3 {
        color: #00f2fe;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 5px solid #00f2fe;
        border-radius: 8px;
        padding: 18px;
        backdrop-filter: blur(10px);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #ffffff;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    /* Estilos para dropdowns y inputs en Streamlit */
    .stTextInput>div>div>input, .stMultiSelect>div {
        background-color: #1e293b !important;
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# ── CARGA DE DATOS (CON CACHÉ) ───────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    # Cargar y combinar datasets
    train = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'drugLibTrain_raw.tsv'), sep='\t', on_bad_lines='skip')
    test  = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'drugLibTest_raw.tsv'),  sep='\t', on_bad_lines='skip')
    df = pd.concat([train, test], ignore_index=True)
    
    # Limpiar columnas
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    elif '' in df.columns:
        df = df.rename(columns={'': 'id_original'})
        
    df = df.rename(columns={'urlDrugName': 'drugName'})
    df = df.dropna(subset=['drugName', 'condition', 'rating'])
    
    # Normalizar textos
    df['drugName']  = df['drugName'].str.lower().str.strip()
    df['condition'] = df['condition'].str.lower().str.strip()
    df['rating']    = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])
    
    # Filtrar condiciones no legibles
    df = df[~df['condition'].str.contains(r'^\d+$', na=True)]
    return df

@st.cache_resource
def construir_similitud(df):
    # Intentar cargar la matriz pre-calculada
    matriz_path = os.path.join(PROJECT_ROOT, 'models', 'similarity_matrix.pkl')
    if os.path.exists(matriz_path):
        try:
            with open(matriz_path, 'rb') as f:
                return pickle.load(f)
        except:
            pass
            
    # Si no existe, construirla
    pivot = df.pivot_table(
        index='condition', columns='drugName',
        values='rating', aggfunc='mean'
    ).fillna(0)
    pivot_drug = pivot.T
    scaler = StandardScaler()
    pivot_scaled = scaler.fit_transform(pivot_drug)
    sim_matrix = cosine_similarity(pivot_scaled)
    return pd.DataFrame(sim_matrix, index=pivot_drug.index, columns=pivot_drug.index)

df = cargar_datos()
drug_sim_df = construir_similitud(df)

# Listas de condiciones y medicamentos únicos para autocompletar
list_conditions = sorted(df['condition'].unique().tolist())
list_drugs = sorted(df['drugName'].unique().tolist())

# ── MODELO DE RECOMENDACIÓN ──────────────────────────────────────────────────
def mejor_medicamento(condition, df, min_reviews=3):
    subset = df[df['condition'] == condition.lower().strip()]
    if subset.empty:
        return None
    res = subset.groupby('drugName').agg(
        avg_rating=('rating','mean'),
        n_reviews=('rating','count')
    ).reset_index()
    
    # Filtrar por reseñas
    res_filtered = res[res['n_reviews'] >= min_reviews]
    if res_filtered.empty:
        res_filtered = res
    return res_filtered.nlargest(10, 'avg_rating')

def top5_cond(condition, df, min_reviews=3):
    subset = df[df['condition'] == condition.lower().strip()]
    if subset.empty:
        return None
    res = subset.groupby('drugName').agg(
        avg_rating=('rating','mean'),
        n_reviews=('rating','count')
    ).reset_index()
    res = res[res['n_reviews'] >= min_reviews]
    return res.nlargest(5, 'avg_rating').reset_index(drop=True)

def condiciones_med(drug, df, top_n=5):
    subset = df[df['drugName'] == drug.lower().strip()]
    if subset.empty:
        return None
    return subset.groupby('condition').agg(
        n=('condition','count'),
        avg_rating=('rating','mean')
    ).reset_index().nlargest(top_n, 'n')

# ── CABECERA / HERO SECTION ──────────────────────────────────────────────────
st.title("Sistema de Recomendación de Medicamentos")
st.subheader("Joan David Martínez Hernández")
st.markdown("""
Esta aplicación permite realizar un **filtrado colaborativo basado en ítems** para recomendar medicamentos 
y predecir el comportamiento y satisfacción del paciente basado en la base de datos **DrugLib**.
""")

# Métricas rápidas del dataset en el Sidebar
st.sidebar.markdown("### Estadísticas Generales")
st.sidebar.markdown(f"**Total Reseñas:** `{len(df):,}`")
st.sidebar.markdown(f"**Medicamentos:** `{df['drugName'].nunique():,}`")
st.sidebar.markdown(f"**Condiciones:** `{df['condition'].nunique():,}`")

# ── PESTAÑAS PRINCIPALES ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Mejor Medicamento",
    "Top 5 por Condición",
    "Similitud entre Fármacos",
    "Condiciones por Medicamento",
    "Exploración General y NLP"
])

# ── TAB 1: MEJOR MEDICAMENTO POR CONDICIÓN ───────────────────────────────────
with tab1:
    st.header("Medicamento Recomendado por Mayor Satisfacción")
    st.markdown("Busca qué medicamento tiene el mayor promedio de satisfacción para una condición específica.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        condition_input = st.selectbox("Selecciona una Condición Médica:", list_conditions, index=list_conditions.index("depression") if "depression" in list_conditions else 0, key="tab1_cond")
    with col2:
        min_rev = st.slider("Mínimo de reseñas requeridas:", 1, 20, 5, key="tab1_slider")
        
    if st.button("Buscar Recomendación", key="btn1", type="primary"):
        result = mejor_medicamento(condition_input, df, min_rev)
        if result is None or result.empty:
            st.error(f"No se encontraron datos de medicamentos para la condición: '{condition_input}'")
        else:
            best = result.iloc[0]
            
            # Tarjeta premium de recomendación
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Medicamento Recomendado</div>
                <div class="metric-value">{best['drugName'].upper()}</div>
                <div style="margin-top: 10px; font-size: 1.1rem;">
                    Calificación de Satisfacción: <b>{best['avg_rating']:.2f} / 10</b> | Reseñas: <b>{best['n_reviews']}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            
            # Gráfico Plotly
            fig = px.bar(
                result, x='drugName', y='avg_rating',
                color='avg_rating', color_continuous_scale='tealgrn',
                text='avg_rating',
                title=f'Medicamentos mejor calificados para: {condition_input.title()}',
                labels={'drugName': 'Medicamento', 'avg_rating': 'Rating Promedio'}
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(
                yaxis_range=[0, 11], 
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ffffff'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabla detallada
            st.subheader("Datos Detallados")
            st.dataframe(result.rename(columns={
                'drugName': 'Medicamento',
                'avg_rating': 'Rating Promedio',
                'n_reviews': 'N° Reseñas'
            }), use_container_width=True)

# ── TAB 2: TOP 5 POR CONDICIÓN ───────────────────────────────────────────────
with tab2:
    st.header("Top 5 Medicamentos Mejor Valorados")
    st.markdown("Muestra un listado ordenado de los 5 mejores medicamentos recomendados para tratar una condición.")
    
    col_input = st.selectbox("Selecciona la Condición Médica:", list_conditions, index=list_conditions.index("anxiety") if "anxiety" in list_conditions else 0, key="tab2_cond")
    
    if st.button("Generar Ranking Top 5", key="btn2", type="primary"):
        result2 = top5_cond(col_input, df)
        if result2 is None or result2.empty:
            st.error(f"No se encontraron suficientes datos de medicamentos para la condición: '{col_input}'")
        else:
            col_left, col_right = st.columns([3, 2])
            
            with col_left:
                fig2 = px.bar(
                    result2, x='avg_rating', y='drugName',
                    orientation='h', color='avg_rating',
                    color_continuous_scale='Blues',
                    text='avg_rating',
                    title=f'Top 5 Medicamentos para: {col_input.title()}',
                    labels={'drugName': 'Medicamento', 'avg_rating': 'Rating Promedio'}
                )
                fig2.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig2.update_layout(
                    xaxis_range=[0, 11],
                    yaxis={'categoryorder': 'total ascending'},
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#ffffff'
                )
                st.plotly_chart(fig2, use_container_width=True)
                
            with col_right:
                st.subheader("Tabla de Ranking")
                result2.index = range(1, len(result2) + 1)
                st.dataframe(result2[['drugName', 'avg_rating', 'n_reviews']].rename(columns={
                    'drugName': 'Medicamento',
                    'avg_rating': 'Satisfacción Promedio',
                    'n_reviews': 'N° Reseñas'
                }), use_container_width=True)

# ── TAB 3: SIMILITUD ENTRE FÁRMACOS ──────────────────────────────────────────
with tab3:
    st.header("Similitud de Medicamentos (Filtrado Colaborativo)")
    st.markdown("Compara dos medicamentos o descubre cuáles son los más parecidos en base a la respuesta de pacientes para las mismas condiciones.")
    
    col1, col2 = st.columns(2)
    with col1:
        drug_a = st.selectbox("Medicamento A:", list_drugs, index=list_drugs.index("lyrica") if "lyrica" in list_drugs else 0, key="drug_a")
    with col2:
        drug_b = st.selectbox("Medicamento B:", list_drugs, index=list_drugs.index("gabapentin") if "gabapentin" in list_drugs else 0, key="drug_b")
        
    if st.button("Calcular Similitud Coseno", key="btn3", type="primary"):
        da, db = drug_a.lower().strip(), drug_b.lower().strip()
        if da not in drug_sim_df.index:
            st.error(f"El medicamento '{da}' no está registrado en la matriz de similitud.")
        elif db not in drug_sim_df.index:
            st.error(f"El medicamento '{db}' no está registrado en la matriz de similitud.")
        else:
            sim = drug_sim_df.loc[da, db]
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Similitud Coseno entre {da.upper()} y {db.upper()}</div>
                <div class="metric-value">{sim:.4f}</div>
                <div style="margin-top: 10px; font-size: 0.95rem; color: #94a3b8;">
                    *Valores cercanos a 1.00 indican perfiles de prescripción y calificaciones de satisfacción muy similares.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            
            # Medicamentos más similares
            col_sim1, col_sim2 = st.columns(2)
            with col_sim1:
                st.subheader(f"Top 10 similares a {da.title()}")
                top_sim_a = drug_sim_df[da].drop(index=da).sort_values(ascending=False).head(10).reset_index()
                top_sim_a.columns = ['Medicamento', 'Similitud']
                fig_sim_a = px.bar(
                    top_sim_a, x='Similitud', y='Medicamento',
                    orientation='h', color='Similitud', color_continuous_scale='teal',
                    title=f'Recomendaciones similares a: {da.title()}'
                )
                fig_sim_a.update_layout(yaxis={'categoryorder': 'total ascending'}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_sim_a, use_container_width=True)
                
            with col_sim2:
                st.subheader(f"Top 10 similares a {db.title()}")
                top_sim_b = drug_sim_df[db].drop(index=db).sort_values(ascending=False).head(10).reset_index()
                top_sim_b.columns = ['Medicamento', 'Similitud']
                fig_sim_b = px.bar(
                    top_sim_b, x='Similitud', y='Medicamento',
                    orientation='h', color='Similitud', color_continuous_scale='teal',
                    title=f'Recomendaciones similares a: {db.title()}'
                )
                fig_sim_b.update_layout(yaxis={'categoryorder': 'total ascending'}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_sim_b, use_container_width=True)

    st.write("")
    st.subheader("Mapa de Calor (Heatmap) Personalizado")
    st.markdown("Genera una matriz visual de similitud para un grupo personalizado de medicamentos.")
    
    lista_default = "lyrica, prozac, zoloft, lexapro, cymbalta, gabapentin, xanax, wellbutrin, effexor"
    lista_drugs_input = st.text_input("Ingresa medicamentos separados por coma:", value=lista_default)
    
    if st.button("Generar Heatmap de Similitud", key="btn_heat"):
        drugs_list = [d.strip().lower() for d in lista_drugs_input.split(',')]
        valid_drugs = [d for d in drugs_list if d in drug_sim_df.index]
        
        if len(valid_drugs) < 2:
            st.error("Por favor, ingresa al menos 2 medicamentos válidos que existan en el sistema.")
        else:
            sub = drug_sim_df.loc[valid_drugs, valid_drugs]
            fig_heat = go.Figure(data=go.Heatmap(
                z=sub.values, x=sub.columns.tolist(), y=sub.index.tolist(),
                colorscale='RdBu', zmid=0,
                text=np.round(sub.values, 2),
                texttemplate='%{text}',
                colorbar=dict(title='Similitud')
            ))
            fig_heat.update_layout(
                title='Matriz de Similitud Coseno',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ffffff',
                width=700, height=600
            )
            st.plotly_chart(fig_heat, use_container_width=True)

# ── TAB 4: CONDICIONES POR MEDICAMENTO ────────────────────────────────────────
with tab4:
    st.header("Condiciones Clínicas más Comunes por Medicamento")
    st.markdown("Analiza para qué patologías o diagnósticos se prescribe con mayor frecuencia un medicamento seleccionado.")
    
    col_drug_input = st.selectbox("Selecciona un Medicamento:", list_drugs, index=list_drugs.index("lyrica") if "lyrica" in list_drugs else 0, key="tab4_drug")
    top_n_cond = st.slider("Condiciones a mostrar:", 3, 10, 5, key="tab4_slider")
    
    if st.button("Buscar Usos Clínicos", key="btn4", type="primary"):
        result4 = condiciones_med(col_drug_input, df, top_n_cond)
        if result4 is None or result4.empty:
            st.error(f"No se encontraron datos para el medicamento: '{col_drug_input}'")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                fig4a = px.pie(
                    result4, names='condition', values='n',
                    title=f'Distribución de Usos para: {col_drug_input.upper()}',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig4a.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
                st.plotly_chart(fig4a, use_container_width=True)
                
            with col_b:
                fig4b = px.bar(
                    result4, x='n', y='condition', orientation='h',
                    color='avg_rating', color_continuous_scale='deep',
                    title='Frecuencia de Uso y Calificación de Satisfacción',
                    labels={'n': 'Reseñas/Prescripciones', 'condition': 'Condición Médica', 'avg_rating': 'Satisfacción'}
                )
                fig4b.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#ffffff'
                )
                st.plotly_chart(fig4b, use_container_width=True)
                
            st.dataframe(result4.rename(columns={
                'condition': 'Condición Médica',
                'n': 'Cantidad de Prescripciones (Reseñas)',
                'avg_rating': 'Satisfacción Promedio'
            }), use_container_width=True)

# ── TAB 5: EXPLORACIÓN GENERAL Y NLP ──────────────────────────────────────────
with tab5:
    st.header("Análisis Exploratorio y Procesamiento de Lenguaje Natural (NLP)")
    
    # Sección 1: Distribución General de Ratings
    st.subheader("Calificación de Satisfacción de Pacientes")
    fig_dist = px.histogram(
        df, x='rating', nbins=10, 
        color_discrete_sequence=['#4facfe'],
        labels={'rating': 'Rating de Satisfacción (1-10)'},
        title='Distribución de Calificaciones en toda la Plataforma'
    )
    fig_dist.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#ffffff',
        yaxis_title='Cantidad de Opiniones'
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    col_x, col_y = st.columns(2)
    with col_x:
        top_cond_list = df['condition'].value_counts().head(10).reset_index()
        top_cond_list.columns = ['condition', 'count']
        fig_cond = px.bar(
            top_cond_list, x='count', y='condition', orientation='h',
            color='count', color_continuous_scale='teal',
            title='Top 10 Condiciones con Mayor Cantidad de Reseñas'
        )
        fig_cond.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff'
        )
        st.plotly_chart(fig_cond, use_container_width=True)
        
    with col_y:
        top_drugs_list = df['drugName'].value_counts().head(10).reset_index()
        top_drugs_list.columns = ['drugName', 'count']
        fig_drugs = px.bar(
            top_drugs_list, x='count', y='drugName', orientation='h',
            color='count', color_continuous_scale='sunset',
            title='Top 10 Medicamentos con Mayor Cantidad de Reseñas'
        )
        fig_drugs.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff'
        )
        st.plotly_chart(fig_drugs, use_container_width=True)

    # Sección 2: Nubes de Palabras (Word Clouds) interactivas
    st.markdown("---")
    st.subheader("Nube de Palabras Interactiva por Condición Médica")
    st.markdown("Visualiza las palabras clave y expresiones más repetidas por los pacientes en sus opiniones de texto.")
    
    col_wc1, col_wc2 = st.columns(2)
    with col_wc1:
        cond_wc = st.selectbox("Selecciona la Condición para la Nube:", list_conditions, index=list_conditions.index("depression") if "depression" in list_conditions else 0, key="wc_cond")
    with col_wc2:
        rev_col = st.selectbox("Selecciona la Reseña a Procesar:", ['benefitsReview', 'sideEffectsReview', 'commentsReview'], index=0, key="wc_col")
        
    if st.button("Generar Nube de Palabras", key="btn_wc"):
        subset_wc = df[df['condition'] == cond_wc.lower().strip()]
        if subset_wc.empty:
            st.error("No hay suficientes opiniones para esta condición.")
        else:
            texto = ' '.join(subset_wc[rev_col].dropna().astype(str).tolist())
            
            # Limpiar HTML entities y símbolos
            texto = re.sub(r'&#\d+;', ' ', texto)
            texto = re.sub(r'[^a-zA-Z\s]', ' ', texto)
            
            if len(texto.strip()) < 10:
                st.warning("No hay suficiente texto para construir la nube de palabras.")
            else:
                with st.spinner("Generando nube de palabras..."):
                    # Crear WordCloud
                    wc = WordCloud(
                        width=800, height=450, 
                        background_color='#0f1219', 
                        max_words=80, 
                        colormap='cyan'
                    ).generate(texto)
                    
                    fig_wc, ax = plt.subplots(figsize=(10, 5), facecolor='#0f1219')
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    plt.tight_layout()
                    st.pyplot(fig_wc)
