# Plan de Desarrollo: Sistema de Recomendación de Medicamentos
**Ciencia de Datos y Minería de Texto — Filtrado Colaborativo**

---

## 0. Contexto y Objetivos

Se construirá un sistema de recomendación de medicamentos basado en **filtrado colaborativo y análisis de texto**, usando un dataset de reseñas farmacéuticas (4.143 instancias: 3.107 train / 1.036 test, formato `.tsv`).

**Preguntas a responder:**
- **a.** Dado una condición médica → medicamento con mayor rating.
- **b.** Top-5 medicamentos mejor valorados para una condición.
- **c.** Similitud entre medicamentos basada en ratings y conteo de utilidad.
- **d.** Dado un medicamento → condiciones más comunes para las que es formulado.
- **e.** Aplicación interactiva (Streamlit) con gráficas y modelo integrado.

---

## 1. Estructura del Proyecto

```
streamlit-project/
│
├── data/
│   ├── drugsComTrain_raw.tsv          # Datos de entrenamiento (75%)
│   └── drugsComTest_raw.tsv           # Datos de prueba (25%)
│
├── notebooks/
│   └── sistema_recomendacion.ipynb    # Notebook completo de análisis
│
├── app/
│   └── app.py                         # Aplicación Streamlit
│
├── models/
│   └── similarity_matrix.pkl          # Matriz de similitud serializada
│
├── outputs/
│   └── reporte_final.pdf              # PDF con resultados y gráficas
│
└── requirements.txt
```

---

## 2. Dependencias y Entorno

```python
# requirements.txt
pandas
numpy
scikit-learn
scipy
matplotlib
seaborn
plotly
streamlit
wordcloud
fpdf2
nbconvert
```

**Instalación:**
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn plotly streamlit wordcloud fpdf2 nbconvert
```

---

## 3. Carga y Exploración de Datos (EDA)

### 3.1 Carga de archivos TSV

```python
import pandas as pd

# Columnas esperadas: uniqueID, drugName, condition, review, rating,
#                    date, usefulCount, sideEffects (texto), effectiveness
train_df = pd.read_csv('data/drugsComTrain_raw.tsv', sep='\t', on_bad_lines='skip')
test_df  = pd.read_csv('data/drugsComTest_raw.tsv',  sep='\t', on_bad_lines='skip')

df = pd.concat([train_df, test_df], ignore_index=True)
print(df.shape)        # Esperado: (4143, 8)
print(df.columns.tolist())
print(df.dtypes)
```

> **⚠️ AGENTE:** Imprimir `df.head()` y `df.info()`. Identificar nombres exactos de columnas antes de continuar. Ajustar nombres de columnas en todos los pasos siguientes según los nombres reales del TSV.

### 3.2 Inspección de columnas clave

```python
# Columnas esperadas (ajustar si difieren):
# 'drugName', 'condition', 'rating', 'usefulCount'
# 'sideEffects' (cualitativa 5 niveles), 'effectiveness' (cualitativa 5 niveles)
# 'review' o columnas de reseñas: benefitsReview, sideEffectsReview, commentsReview

print(df['rating'].describe())
print(df['condition'].value_counts().head(10))
print(df['drugName'].nunique())
```

### 3.3 Limpieza de datos

```python
# 1. Eliminar filas sin condición o sin medicamento
df = df.dropna(subset=['drugName', 'condition', 'rating'])

# 2. Normalizar texto: minúsculas, strip
df['drugName']  = df['drugName'].str.lower().str.strip()
df['condition'] = df['condition'].str.lower().str.strip()

# 3. Convertir rating a numérico
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna(subset=['rating'])

# 4. Convertir usefulCount a numérico
df['usefulCount'] = pd.to_numeric(df['usefulCount'], errors='coerce').fillna(0)

# 5. Filtrar condiciones con texto ilegible (ej: dígitos, "not listed", etc.)
df = df[~df['condition'].str.contains(r'\d', na=True)]

print(f"Datos limpios: {df.shape}")
```

### 3.4 Estadísticas descriptivas y visualizaciones EDA

```python
import matplotlib.pyplot as plt
import seaborn as sns

# a) Distribución de ratings
plt.figure(figsize=(8,4))
sns.histplot(df['rating'], bins=10, kde=True)
plt.title('Distribución de Ratings')
plt.xlabel('Rating (1-10)')
plt.savefig('outputs/eda_rating_dist.png', dpi=150, bbox_inches='tight')
plt.show()

# b) Top 15 condiciones más frecuentes
top_conditions = df['condition'].value_counts().head(15)
plt.figure(figsize=(10,5))
sns.barplot(x=top_conditions.values, y=top_conditions.index, palette='viridis')
plt.title('Top 15 Condiciones Médicas')
plt.xlabel('Cantidad de Reseñas')
plt.savefig('outputs/eda_top_conditions.png', dpi=150, bbox_inches='tight')
plt.show()

# c) Top 15 medicamentos más reseñados
top_drugs = df['drugName'].value_counts().head(15)
plt.figure(figsize=(10,5))
sns.barplot(x=top_drugs.values, y=top_drugs.index, palette='magma')
plt.title('Top 15 Medicamentos Más Reseñados')
plt.savefig('outputs/eda_top_drugs.png', dpi=150, bbox_inches='tight')
plt.show()

# d) Boxplot de rating por efectividad (si existe columna)
if 'effectiveness' in df.columns:
    plt.figure(figsize=(10,5))
    sns.boxplot(data=df, x='effectiveness', y='rating')
    plt.title('Rating vs Efectividad')
    plt.savefig('outputs/eda_rating_effectiveness.png', dpi=150, bbox_inches='tight')
    plt.show()
```

---

## 4. Construcción del Sistema de Recomendación (Filtrado Colaborativo)

El enfoque es **filtrado colaborativo basado en ítems (Item-Based CF)**, usando la matriz usuario-ítem donde:
- **Filas:** condiciones médicas
- **Columnas:** medicamentos
- **Valores:** rating promedio ponderado + usefulCount

### 4.1 Construcción de la Matriz Condición–Medicamento

```python
import numpy as np

# Rating ponderado: combinar rating con usefulCount como peso de confianza
df['weighted_rating'] = df['rating'] * np.log1p(df['usefulCount'] + 1)

# Pivot: condición × medicamento → promedio de weighted_rating
pivot = df.pivot_table(
    index='condition',
    columns='drugName',
    values='weighted_rating',
    aggfunc='mean'
)

# También guardar pivot de rating puro para preguntas a y b
pivot_raw = df.pivot_table(
    index='condition',
    columns='drugName',
    values='rating',
    aggfunc='mean'
)

print(f"Matriz Condición×Medicamento: {pivot.shape}")
```

### 4.2 Similitud entre Medicamentos (Pregunta c)

```python
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Transponer: medicamentos como filas, condiciones como columnas
pivot_drug = pivot.T.fillna(0)

# Normalizar
scaler = StandardScaler()
pivot_scaled = scaler.fit_transform(pivot_drug)

# Matriz de similitud coseno entre medicamentos
drug_sim_matrix = cosine_similarity(pivot_scaled)
drug_sim_df = pd.DataFrame(
    drug_sim_matrix,
    index=pivot_drug.index,
    columns=pivot_drug.index
)

print("Matriz de similitud de medicamentos creada.")
print(drug_sim_df.shape)

# Serializar para uso en la app
import pickle
with open('models/similarity_matrix.pkl', 'wb') as f:
    pickle.dump(drug_sim_df, f)
```

---

## 5. Respuestas a las Preguntas del Sistema

### Pregunta a — Medicamento con mayor rating para una condición

```python
def recomendar_mejor_medicamento(condition, df, min_reviews=5):
    """
    Dado una condición, retorna el medicamento con el mayor rating promedio.
    Se filtran medicamentos con menos de min_reviews reseñas para robustez.
    """
    condition = condition.lower().strip()
    subset = df[df['condition'] == condition]

    if subset.empty:
        return f"No se encontraron datos para la condición: '{condition}'"

    # Agrupar: promedio de rating y conteo de reseñas
    resumen = subset.groupby('drugName').agg(
        avg_rating=('rating', 'mean'),
        n_reviews=('rating', 'count'),
        avg_useful=('usefulCount', 'mean')
    ).reset_index()

    # Filtrar por mínimo de reseñas
    resumen = resumen[resumen['n_reviews'] >= min_reviews]

    if resumen.empty:
        resumen = subset.groupby('drugName').agg(
            avg_rating=('rating', 'mean'),
            n_reviews=('rating', 'count')
        ).reset_index()

    best = resumen.loc[resumen['avg_rating'].idxmax()]
    return best

# PRUEBA a.1: depression
print("=== CONDICIÓN: depression ===")
resultado = recomendar_mejor_medicamento('depression', df)
print(resultado)

# PRUEBA a.2: breast cancer
print("\n=== CONDICIÓN: breast cancer ===")
resultado = recomendar_mejor_medicamento('breast cancer', df)
print(resultado)
```

**Visualización para pregunta a:**
```python
def plot_top_medicamentos_condicion(condition, df, top_n=10, min_reviews=3):
    condition = condition.lower().strip()
    subset = df[df['condition'] == condition]
    resumen = subset.groupby('drugName').agg(
        avg_rating=('rating', 'mean'),
        n_reviews=('rating', 'count')
    ).reset_index()
    resumen = resumen[resumen['n_reviews'] >= min_reviews].nlargest(top_n, 'avg_rating')

    plt.figure(figsize=(10, 5))
    bars = plt.barh(resumen['drugName'], resumen['avg_rating'], color='steelblue')
    plt.xlabel('Rating Promedio')
    plt.title(f'Top {top_n} Medicamentos para: {condition.title()}')
    plt.xlim(0, 10)
    for bar, n in zip(bars, resumen['n_reviews']):
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f'n={n}', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'outputs/top_drugs_{condition.replace(" ","_")}.png', dpi=150)
    plt.show()

plot_top_medicamentos_condicion('depression', df)
plot_top_medicamentos_condicion('breast cancer', df)
```

---

### Pregunta b — Top-5 medicamentos para una condición

```python
def top5_medicamentos(condition, df, min_reviews=3):
    """
    Retorna los 5 medicamentos mejor valorados para una condición dada.
    Ordena por rating promedio descendente; desempata por usefulCount.
    """
    condition = condition.lower().strip()
    subset = df[df['condition'] == condition]

    if subset.empty:
        return f"No se encontraron datos para: '{condition}'"

    resumen = subset.groupby('drugName').agg(
        avg_rating=('rating', 'mean'),
        n_reviews=('rating', 'count'),
        avg_useful=('usefulCount', 'mean')
    ).reset_index()

    resumen = resumen[resumen['n_reviews'] >= min_reviews]
    top5 = resumen.nlargest(5, ['avg_rating', 'avg_useful'])
    top5 = top5.reset_index(drop=True)
    top5.index += 1  # Ranking desde 1
    return top5[['drugName', 'avg_rating', 'n_reviews', 'avg_useful']]

# PRUEBA b.1: allergies
print("=== TOP 5 PARA: allergies ===")
print(top5_medicamentos('allergies', df).to_string())

# PRUEBA b.2: anxiety
print("\n=== TOP 5 PARA: anxiety ===")
print(top5_medicamentos('anxiety', df).to_string())
```

**Visualización para pregunta b:**
```python
import plotly.express as px

def plot_top5_plotly(condition, df):
    top5 = top5_medicamentos(condition, df)
    if isinstance(top5, str):
        print(top5)
        return
    fig = px.bar(
        top5.reset_index(),
        x='drugName', y='avg_rating',
        color='avg_rating',
        color_continuous_scale='Blues',
        text='avg_rating',
        title=f'Top 5 Medicamentos — {condition.title()}',
        labels={'drugName': 'Medicamento', 'avg_rating': 'Rating Promedio'}
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis_range=[0, 10])
    fig.show()
    fig.write_image(f'outputs/top5_{condition.replace(" ","_")}.png')

plot_top5_plotly('allergies', df)
plot_top5_plotly('anxiety', df)
```

---

### Pregunta c — Similitud entre Medicamentos

```python
def similitud_drogas(drug1, drug2, drug_sim_df):
    """
    Retorna la similitud coseno entre dos medicamentos.
    """
    drug1 = drug1.lower().strip()
    drug2 = drug2.lower().strip()

    if drug1 not in drug_sim_df.index:
        return f"Medicamento '{drug1}' no encontrado en la matriz."
    if drug2 not in drug_sim_df.index:
        return f"Medicamento '{drug2}' no encontrado en la matriz."

    sim = drug_sim_df.loc[drug1, drug2]
    return f"Similitud entre '{drug1}' y '{drug2}': {sim:.4f}"

def top_similares(drug, drug_sim_df, top_n=10):
    """
    Retorna los top_n medicamentos más similares a uno dado.
    """
    drug = drug.lower().strip()
    if drug not in drug_sim_df.index:
        return f"'{drug}' no encontrado."
    sim_series = drug_sim_df[drug].drop(index=drug).sort_values(ascending=False)
    return sim_series.head(top_n).reset_index()

# PRUEBA c: comparar medicamentos representativos
print(similitud_drogas('lyrica', 'gabapentin', drug_sim_df))
print(similitud_drogas('prozac', 'zoloft', drug_sim_df))

print("\n=== Medicamentos más similares a Lyrica ===")
print(top_similares('lyrica', drug_sim_df).to_string())
```

**Visualización similitud — Heatmap:**
```python
def plot_heatmap_similitud(drugs_list, drug_sim_df):
    """
    Heatmap de similitud entre una lista de medicamentos.
    """
    drugs_list = [d.lower().strip() for d in drugs_list if d.lower().strip() in drug_sim_df.index]
    sub = drug_sim_df.loc[drugs_list, drugs_list]

    plt.figure(figsize=(10, 8))
    sns.heatmap(sub, annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Heatmap de Similitud entre Medicamentos')
    plt.tight_layout()
    plt.savefig('outputs/heatmap_similitud.png', dpi=150)
    plt.show()

# Lista representativa de medicamentos para el heatmap
sample_drugs = ['lyrica', 'prozac', 'zoloft', 'lexapro', 'cymbalta',
                'gabapentin', 'xanax', 'wellbutrin', 'effexor', 'celexa']
plot_heatmap_similitud(sample_drugs, drug_sim_df)
```

---

### Pregunta d — Condiciones para un medicamento dado

```python
def condiciones_por_medicamento(drug, df, top_n=5):
    """
    Dado un medicamento, lista las top_n condiciones para las que es formulado.
    Ordena por frecuencia de prescripción (conteo de instancias).
    """
    drug = drug.lower().strip()
    subset = df[df['drugName'] == drug]

    if subset.empty:
        return f"No se encontraron datos para el medicamento: '{drug}'"

    resumen = subset.groupby('condition').agg(
        n_prescripciones=('condition', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index().nlargest(top_n, 'n_prescripciones')

    resumen.index = range(1, len(resumen) + 1)
    return resumen

# PRUEBA d.1: lyrica
print("=== CONDICIONES PARA: lyrica ===")
print(condiciones_por_medicamento('lyrica', df).to_string())

# PRUEBA d.2: prozac
print("\n=== CONDICIONES PARA: prozac ===")
print(condiciones_por_medicamento('prozac', df).to_string())
```

**Visualización pregunta d:**
```python
def plot_condiciones_medicamento(drug, df):
    result = condiciones_por_medicamento(drug, df)
    if isinstance(result, str):
        print(result)
        return

    fig = px.pie(
        result,
        names='condition',
        values='n_prescripciones',
        title=f'Condiciones para las que se prescribe: {drug.title()}',
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    fig.show()
    fig.write_image(f'outputs/condiciones_{drug}.png')

plot_condiciones_medicamento('lyrica', df)
plot_condiciones_medicamento('prozac', df)
```

---

## 6. Análisis de Texto (Word Clouds y NLP básico)

```python
from wordcloud import WordCloud

def generar_wordcloud(condition, df, columna_review='review'):
    """
    Genera una nube de palabras con las reseñas de una condición específica.
    """
    condition = condition.lower().strip()
    subset = df[df['condition'] == condition]

    if subset.empty or columna_review not in df.columns:
        print(f"Sin datos de reseñas para: {condition}")
        return

    texto = ' '.join(subset[columna_review].dropna().astype(str).tolist())

    # Limpiar HTML entities comunes
    import re
    texto = re.sub(r'&#\d+;', ' ', texto)
    texto = re.sub(r'[^a-zA-Z\s]', ' ', texto)

    wc = WordCloud(width=800, height=400, background_color='white',
                   max_words=100, colormap='viridis').generate(texto)

    plt.figure(figsize=(12, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Palabras Clave en Reseñas — {condition.title()}')
    plt.tight_layout()
    plt.savefig(f'outputs/wordcloud_{condition.replace(" ","_")}.png', dpi=150)
    plt.show()

# Ajustar columna según el nombre real en el dataset
# Posibles nombres: 'review', 'benefitsReview', 'sideEffectsReview', 'commentsReview'
# Probar con el que corresponda:
generar_wordcloud('depression', df)
generar_wordcloud('anxiety', df)
```

---

## 7. Aplicación Streamlit (`app/app.py`)

### Estructura de la App

La aplicación tendrá **4 pestañas** correspondientes a las preguntas a–d, más una pestaña de exploración general.

```python
# app/app.py
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

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Sistema de Recomendación de Medicamentos",
    page_icon="💊",
    layout="wide"
)

# ── Carga de datos (con caché) ───────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    train = pd.read_csv('data/drugsComTrain_raw.tsv', sep='\t', on_bad_lines='skip')
    test  = pd.read_csv('data/drugsComTest_raw.tsv',  sep='\t', on_bad_lines='skip')
    df = pd.concat([train, test], ignore_index=True)

    df['drugName']  = df['drugName'].str.lower().str.strip()
    df['condition'] = df['condition'].str.lower().str.strip()
    df['rating']    = pd.to_numeric(df['rating'], errors='coerce')
    df['usefulCount'] = pd.to_numeric(df['usefulCount'], errors='coerce').fillna(0)
    df = df.dropna(subset=['drugName', 'condition', 'rating'])
    df = df[~df['condition'].str.contains(r'\d', na=True)]
    df['weighted_rating'] = df['rating'] * np.log1p(df['usefulCount'] + 1)
    return df

@st.cache_resource
def construir_similitud(df):
    pivot = df.pivot_table(
        index='condition', columns='drugName',
        values='weighted_rating', aggfunc='mean'
    ).fillna(0)
    pivot_drug = pivot.T
    scaler = StandardScaler()
    pivot_scaled = scaler.fit_transform(pivot_drug)
    sim_matrix = cosine_similarity(pivot_scaled)
    return pd.DataFrame(sim_matrix, index=pivot_drug.index, columns=pivot_drug.index)

df = cargar_datos()
drug_sim_df = construir_similitud(df)

# ── Funciones del modelo ─────────────────────────────────────────────────────
def mejor_medicamento(condition, df, min_reviews=3):
    subset = df[df['condition'] == condition.lower().strip()]
    if subset.empty:
        return None
    res = subset.groupby('drugName').agg(
        avg_rating=('rating','mean'), n=('rating','count'),
        avg_useful=('usefulCount','mean')
    ).reset_index()
    res = res[res['n'] >= min_reviews]
    if res.empty:
        res = subset.groupby('drugName').agg(
            avg_rating=('rating','mean'), n=('rating','count')
        ).reset_index()
    return res.nlargest(10, 'avg_rating')

def top5_cond(condition, df, min_reviews=3):
    subset = df[df['condition'] == condition.lower().strip()]
    if subset.empty:
        return None
    res = subset.groupby('drugName').agg(
        avg_rating=('rating','mean'), n=('rating','count'),
        avg_useful=('usefulCount','mean')
    ).reset_index()
    res = res[res['n'] >= min_reviews]
    return res.nlargest(5, 'avg_rating').reset_index(drop=True)

def condiciones_med(drug, df, top_n=5):
    subset = df[df['drugName'] == drug.lower().strip()]
    if subset.empty:
        return None
    return subset.groupby('condition').agg(
        n=('condition','count'), avg_rating=('rating','mean')
    ).reset_index().nlargest(top_n, 'n')

# ── Header ───────────────────────────────────────────────────────────────────
st.title("💊 Sistema de Recomendación de Medicamentos")
st.markdown("**Filtrado Colaborativo basado en reseñas farmacéuticas**")
st.markdown(f"📊 Dataset: **{len(df):,}** reseñas | **{df['drugName'].nunique():,}** medicamentos | **{df['condition'].nunique():,}** condiciones")

# ── Pestañas ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Mejor Medicamento",
    "📋 Top 5 por Condición",
    "🔗 Similitud entre Drogas",
    "🩺 Condiciones por Medicamento",
    "📊 Exploración General"
])

# ── TAB 1: Mejor medicamento por condición ───────────────────────────────────
with tab1:
    st.header("Medicamento con Mayor Rating para una Condición")
    condition_input = st.text_input("Ingrese la condición médica:", value="depression", key="tab1_cond")
    min_rev = st.slider("Mínimo de reseñas requeridas:", 1, 20, 5)

    if st.button("Recomendar", key="btn1"):
        result = mejor_medicamento(condition_input, df, min_rev)
        if result is None:
            st.error(f"No se encontraron datos para: '{condition_input}'")
        else:
            best = result.iloc[0]
            st.success(f"✅ Mejor medicamento: **{best['drugName'].title()}**  |  Rating: **{best['avg_rating']:.2f}/10**")

            fig = px.bar(
                result, x='drugName', y='avg_rating',
                color='avg_rating', color_continuous_scale='Blues',
                text='avg_rating',
                title=f'Top Medicamentos — {condition_input.title()}',
                labels={'drugName':'Medicamento','avg_rating':'Rating Promedio'}
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(yaxis_range=[0,10], xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(result.rename(columns={
                'drugName':'Medicamento','avg_rating':'Rating Prom.',
                'n':'N° Reseñas','avg_useful':'Utilidad Prom.'
            }), use_container_width=True)

# ── TAB 2: Top 5 por condición ───────────────────────────────────────────────
with tab2:
    st.header("Top 5 Medicamentos para una Condición")
    cond2 = st.text_input("Condición médica:", value="anxiety", key="tab2_cond")

    if st.button("Obtener Top 5", key="btn2"):
        result2 = top5_cond(cond2, df)
        if result2 is None:
            st.error(f"No hay datos para: '{cond2}'")
        else:
            st.subheader(f"Top 5 para: {cond2.title()}")
            fig2 = px.bar(
                result2, x='avg_rating', y='drugName',
                orientation='h', color='avg_rating',
                color_continuous_scale='Viridis',
                text='avg_rating',
                title=f'Top 5 Medicamentos — {cond2.title()}',
                labels={'drugName':'Medicamento','avg_rating':'Rating Promedio'}
            )
            fig2.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig2.update_layout(xaxis_range=[0,10], yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig2, use_container_width=True)

            result2.index = range(1, 6)
            st.dataframe(result2[['drugName','avg_rating','n','avg_useful']].rename(
                columns={'drugName':'Medicamento','avg_rating':'Rating Prom.',
                         'n':'N° Reseñas','avg_useful':'Utilidad Prom.'}
            ), use_container_width=True)

# ── TAB 3: Similitud entre Drogas ────────────────────────────────────────────
with tab3:
    st.header("Similitud entre Medicamentos")
    col1, col2 = st.columns(2)

    with col1:
        drug_a = st.text_input("Medicamento A:", value="lyrica", key="drug_a")
    with col2:
        drug_b = st.text_input("Medicamento B:", value="gabapentin", key="drug_b")

    if st.button("Calcular Similitud", key="btn3"):
        da, db = drug_a.lower().strip(), drug_b.lower().strip()
        if da not in drug_sim_df.index:
            st.error(f"'{da}' no encontrado.")
        elif db not in drug_sim_df.index:
            st.error(f"'{db}' no encontrado.")
        else:
            sim = drug_sim_df.loc[da, db]
            st.metric("Similitud Coseno", f"{sim:.4f}", delta=None)
            st.info("Valores cercanos a 1 indican alta similitud en perfil de ratings y popularidad entre condiciones.")

            # Top 10 similares al medicamento A
            top_sim = drug_sim_df[da].drop(index=da).sort_values(ascending=False).head(10).reset_index()
            top_sim.columns = ['Medicamento', 'Similitud']

            fig3 = px.bar(
                top_sim, x='Similitud', y='Medicamento',
                orientation='h', color='Similitud',
                color_continuous_scale='RdYlGn',
                title=f'Top 10 Medicamentos más Similares a: {da.title()}'
            )
            fig3.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig3, use_container_width=True)

    # Heatmap de lista personalizada
    st.subheader("Heatmap de Similitud entre Múltiples Medicamentos")
    lista_drugs = st.text_input(
        "Lista de medicamentos (separados por coma):",
        value="lyrica, prozac, zoloft, lexapro, cymbalta, gabapentin, xanax"
    )

    if st.button("Generar Heatmap", key="btn_heat"):
        drugs_list = [d.strip().lower() for d in lista_drugs.split(',')]
        valid_drugs = [d for d in drugs_list if d in drug_sim_df.index]

        if len(valid_drugs) < 2:
            st.error("Se necesitan al menos 2 medicamentos válidos.")
        else:
            sub = drug_sim_df.loc[valid_drugs, valid_drugs]
            fig_heat = go.Figure(data=go.Heatmap(
                z=sub.values, x=sub.columns.tolist(), y=sub.index.tolist(),
                colorscale='RdBu', zmid=0,
                text=np.round(sub.values, 2),
                texttemplate='%{text}',
                colorbar=dict(title='Similitud')
            ))
            fig_heat.update_layout(title='Heatmap de Similitud entre Medicamentos',
                                   width=700, height=600)
            st.plotly_chart(fig_heat, use_container_width=True)

# ── TAB 4: Condiciones por medicamento ──────────────────────────────────────
with tab4:
    st.header("Condiciones para las que es Formulado un Medicamento")
    drug_input = st.text_input("Nombre del medicamento:", value="lyrica", key="tab4_drug")
    top_n_cond = st.slider("Número de condiciones a mostrar:", 3, 15, 5)

    if st.button("Buscar Condiciones", key="btn4"):
        result4 = condiciones_med(drug_input, df, top_n_cond)
        if result4 is None:
            st.error(f"No se encontraron datos para: '{drug_input}'")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                fig4a = px.pie(
                    result4, names='condition', values='n',
                    title=f'Distribución de Condiciones — {drug_input.title()}',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig4a, use_container_width=True)
            with col_b:
                fig4b = px.bar(
                    result4, x='n', y='condition', orientation='h',
                    color='avg_rating', color_continuous_scale='Blues',
                    title='Condiciones por Frecuencia y Rating',
                    labels={'n':'N° Prescripciones','condition':'Condición','avg_rating':'Rating Prom.'}
                )
                fig4b.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig4b, use_container_width=True)

            st.dataframe(result4.rename(columns={
                'condition':'Condición','n':'N° Prescripciones','avg_rating':'Rating Prom.'
            }), use_container_width=True)

# ── TAB 5: Exploración General ───────────────────────────────────────────────
with tab5:
    st.header("Exploración General del Dataset")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reseñas", f"{len(df):,}")
    col2.metric("Medicamentos Únicos", f"{df['drugName'].nunique():,}")
    col3.metric("Condiciones Únicas", f"{df['condition'].nunique():,}")

    # Distribución de ratings
    fig_dist = px.histogram(df, x='rating', nbins=10, color_discrete_sequence=['steelblue'],
                             title='Distribución de Ratings')
    st.plotly_chart(fig_dist, use_container_width=True)

    col_e, col_f = st.columns(2)
    with col_e:
        top_cond = df['condition'].value_counts().head(15).reset_index()
        top_cond.columns = ['condition', 'count']
        fig_cond = px.bar(top_cond, x='count', y='condition', orientation='h',
                          color='count', color_continuous_scale='Teal',
                          title='Top 15 Condiciones Más Frecuentes')
        fig_cond.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_cond, use_container_width=True)

    with col_f:
        top_dr = df['drugName'].value_counts().head(15).reset_index()
        top_dr.columns = ['drugName', 'count']
        fig_dr = px.bar(top_dr, x='count', y='drugName', orientation='h',
                        color='count', color_continuous_scale='Sunset',
                        title='Top 15 Medicamentos Más Reseñados')
        fig_dr.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_dr, use_container_width=True)

    # Word Cloud interactiva
    st.subheader("Nube de Palabras por Condición")
    # Identificar columna de reseña disponible
    review_cols = [c for c in df.columns if 'review' in c.lower() or 'benefit' in c.lower() or 'comment' in c.lower()]
    cond_wc = st.text_input("Condición para Word Cloud:", value="depression", key="wc_cond")
    rev_col = st.selectbox("Columna de reseña:", review_cols if review_cols else ['review'])

    if st.button("Generar Word Cloud", key="btn_wc"):
        import re
        subset_wc = df[df['condition'] == cond_wc.lower().strip()]
        if subset_wc.empty or rev_col not in df.columns:
            st.error("No hay datos disponibles.")
        else:
            texto = ' '.join(subset_wc[rev_col].dropna().astype(str))
            texto = re.sub(r'&#\d+;', ' ', texto)
            texto = re.sub(r'[^a-zA-Z\s]', ' ', texto)
            wc = WordCloud(width=800, height=400, background_color='white',
                           max_words=80, colormap='viridis').generate(texto)
            fig_wc, ax = plt.subplots(figsize=(12, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig_wc)
```

---

## 8. Generación del Notebook y PDF

### 8.1 Ejecución del Notebook

```bash
# Desde la carpeta streamlit-project
jupyter nbconvert --to notebook --execute notebooks/sistema_recomendacion.ipynb \
    --output notebooks/sistema_recomendacion_ejecutado.ipynb
```

### 8.2 Exportar Notebook a PDF

```bash
jupyter nbconvert --to pdf notebooks/sistema_recomendacion_ejecutado.ipynb \
    --output outputs/reporte_final.pdf
```

> **Alternativa si LaTeX no está disponible:**

```python
from fpdf2 import FPDF

def exportar_pdf(imagenes, titulo, salida):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, titulo, ln=True, align='C')
    pdf.ln(5)

    for img_path in imagenes:
        try:
            pdf.add_page()
            pdf.image(img_path, x=10, y=20, w=190)
        except Exception as e:
            print(f"Error con {img_path}: {e}")

    pdf.output(salida)
    print(f"PDF generado: {salida}")

imagenes = [
    'outputs/eda_rating_dist.png',
    'outputs/eda_top_conditions.png',
    'outputs/eda_top_drugs.png',
    'outputs/top_drugs_depression.png',
    'outputs/top_drugs_breast_cancer.png',
    'outputs/top5_allergies.png',
    'outputs/top5_anxiety.png',
    'outputs/heatmap_similitud.png',
    'outputs/condiciones_lyrica.png',
    'outputs/condiciones_prozac.png',
]
exportar_pdf(imagenes, "Sistema de Recomendación de Medicamentos — Resultados", "outputs/reporte_final.pdf")
```

---

## 9. Ejecución de la Aplicación Streamlit

```bash
cd streamlit-project
streamlit run app/app.py
```

---

## 10. Orden de Ejecución para el Agente de Código

| Paso | Acción | Archivo |
|------|--------|---------|
| 1 | Instalar dependencias | `requirements.txt` |
| 2 | Verificar columnas reales del TSV | notebook — Sección 3.1 |
| 3 | EDA completo + gráficas | notebook — Sección 3 |
| 4 | Construir matriz de similitud y serializar | notebook — Sección 4 |
| 5 | Responder preguntas a–d con pruebas | notebook — Sección 5 |
| 6 | Generar Word Clouds | notebook — Sección 6 |
| 7 | Ejecutar y exportar notebook a PDF | Sección 8 |
| 8 | Lanzar app Streamlit | `app/app.py` |

---

## 11. Notas Finales para el Agente

- **Nombres de columnas:** Verificar siempre con `df.columns.tolist()` antes de usarlas. Los TSV pueden tener variaciones como `&#34;` en los textos de reseñas (entidades HTML); limpiarlas antes de NLP.
- **Índice en TSV:** El archivo puede tener una columna `Unnamed: 0` que es el índice; descartarla si existe.
- **Escala cualitativa:** Las columnas `sideEffects` y `effectiveness` pueden ser texto (e.g., "Mild Side Effects", "Highly Effective"); no convertir a numérico sin codificación ordinal explícita.
- **Plotly en Streamlit:** Siempre usar `st.plotly_chart(fig, use_container_width=True)`.
- **Tamaño de la matriz:** Si la memoria es limitada, usar solo los top-500 medicamentos más frecuentes para construir la matriz de similitud.
- **Similitud coseno:** Valores negativos son posibles y válidos; indican perfiles de uso opuestos entre condiciones.
