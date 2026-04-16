import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import requests
from datetime import datetime, timedelta
import warnings
from pathlib import Path
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ========================
#  CONFIGURAÇÃO DA PÁGINA
# ========================
st.set_page_config(
    page_title="Análise de Atividade Sísmica",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Customização de CSS para fonte branca
st.markdown("""
<style>
    body, p, h1, h2, h3, h4, h5, h6, label, div, span {
        color: #ffffff !important;
    }
    .stMarkdown, .stText {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Dashboard de Análise de Atividade Sísmica")
st.markdown("""
Análise interativa de terremotos em tempo real com dados da API USGS.
Utilize os filtros laterais para explorar a atividade sísmica por região, magnitude e período.
""")

# ========================
#  CACHE E CARREGAMENTO DADOS
# ========================
@st.cache_data(ttl=3600)  # Cache por 1 hora
def carregar_dados():
    """Coleta dados da API USGS"""
    BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    API_PARAMS = {
        "format": "geojson",
        "starttime": "2023-01-01",
        "minmagnitude": 2.5,
        "limit": 10000
    }
    
    try:
        response = requests.get(BASE_URL, params=API_PARAMS, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Parse GeoJSON
        earthquakes = []
        for feature in data["features"]:
            props = feature["properties"]
            geom = feature["geometry"]["coordinates"]
            earthquakes.append({
                "id": props.get("ids", ""),
                "place": props.get("place", ""),
                "magnitude": props.get("mag", 0),
                "depth": geom[2] if len(geom) > 2 else 0,
                "latitude": geom[1],
                "longitude": geom[0],
                "time": pd.to_datetime(props.get("time", 0), unit="ms"),
                "tsunami": props.get("tsunami", 0)
            })
        
        df = pd.DataFrame(earthquakes)
        
        # Extrai país a partir do lugar
        df["pais"] = df["place"].fillna("Desconhecido").str.split(", ").str[-1]
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()

# ========================
#  GERAR ANÁLISES PARA analysis_report.py
# ========================
@st.cache_data(ttl=3600)  # Cache por 1 hora
def gerar_analises(df_full):
    """Gera análises e salva em outputs_n1/ para sincronização com analysis_report.py"""
    
    OUTPUT_DIR = Path("outputs_n1")
    TABELAS_DIR = OUTPUT_DIR / "tabelas"
    GRAFICOS_DIR = OUTPUT_DIR / "graficos"
    
    TABELAS_DIR.mkdir(parents=True, exist_ok=True)
    GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)
    
    df = df_full.copy()
    
    # ---- Dataset Bruto
    df.to_csv(TABELAS_DIR / "01_dataset_bruto.csv", index=False)
    
    # ---- AUDITORIA
    missing_table = df.isna().sum().reset_index()
    missing_table.columns = ["coluna", "qtd_missing"]
    missing_table["pct_missing"] = (missing_table["qtd_missing"] / len(df)) * 100
    missing_table.to_csv(TABELAS_DIR / "02_missing_values.csv", index=False)
    
    plt.figure(figsize=(12, 5))
    plt.bar(missing_table["coluna"], missing_table["qtd_missing"])
    plt.title("Quantidade de valores ausentes por variável")
    plt.xlabel("Variável")
    plt.ylabel("Quantidade de valores ausentes")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / "01_missing_values.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    # Duplicidades
    duplicidade_resumo = pd.DataFrame({
        "tipo": ["linhas_duplicadas", "ids_duplicados", "timestamps_duplicados"],
        "quantidade": [
            int(df.duplicated().sum()),
            int(df["id"].duplicated().sum()),
            int(df["time"].duplicated().sum())
        ]
    })
    duplicidade_resumo.to_csv(TABELAS_DIR / "03_duplicidades.csv", index=False)
    
    # Gaps Temporais
    df_sorted = df.sort_values("time").reset_index(drop=True)
    df_sorted["time_diff_hours"] = df_sorted["time"].diff().dt.total_seconds() / 3600
    gaps_stats = df_sorted["time_diff_hours"].describe().to_frame(name="time_diff_hours")
    gaps_stats.to_csv(TABELAS_DIR / "04_estatisticas_intervalos_temporais.csv")
    
    # Gaps longos (placeholder)
    gaps_longos = df_sorted[df_sorted["time_diff_hours"] > 24].copy()
    gaps_longos.to_csv(TABELAS_DIR / "05_gaps_longos.csv", index=False)
    
    # Outliers (IQR)
    variaveis_numericas = ["magnitude", "depth"]
    outliers_lista = []
    
    for col in variaveis_numericas:
        serie = df[col].dropna()
        if len(serie) > 0:
            q1 = serie.quantile(0.25)
            q3 = serie.quantile(0.75)
            iqr = q3 - q1
            qtd_outliers = int(((serie < (q1 - 1.5 * iqr)) | (serie > (q3 + 1.5 * iqr))).sum())
            
            outliers_lista.append({
                "variavel": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "limite_inferior": q1 - 1.5 * iqr,
                "limite_superior": q3 + 1.5 * iqr,
                "qtd_outliers": qtd_outliers
            })
    
    pd.DataFrame(outliers_lista).to_csv(TABELAS_DIR / "06_outliers_iqr.csv", index=False)
    
    # Inconsistências
    inconsistencias_df = pd.DataFrame({
        "problema": ["magnitude_negativa", "profundidade_negativa"],
        "quantidade": [int((df["magnitude"] < 0).sum()), int((df["depth"] < 0).sum())]
    })
    inconsistencias_df.to_csv(TABELAS_DIR / "07_inconsistencias_logicas.csv", index=False)
    
    # ---- TRATAMENTO
    df_tratado = df.dropna(subset=["magnitude", "latitude", "longitude", "depth"])
    df_tratado = df_tratado[(df_tratado["latitude"] >= -90) & (df_tratado["latitude"] <= 90)]
    df_tratado = df_tratado[(df_tratado["longitude"] >= -180) & (df_tratado["longitude"] <= 180)]
    df_tratado = df_tratado[df_tratado["magnitude"] >= 0]
    df_tratado = df_tratado[df_tratado["depth"] >= 0]
    
    df_tratado.to_csv(TABELAS_DIR / "08_dataset_eventos_tratado.csv", index=False)
    
    # Série temporal diária
    df_tratado_copy = df_tratado.copy()
    df_tratado_copy["date"] = df_tratado_copy["time"].dt.floor("D")
    daily_series = (
        df_tratado_copy.groupby("date")
        .agg(
            quake_count=("id", "count"),
            avg_magnitude=("magnitude", "mean"),
            max_magnitude=("magnitude", "max"),
            avg_depth=("depth", "mean"),
            tsunami_count=("tsunami", "sum")
        )
        .sort_index()
    )
    
    daily_series = daily_series.asfreq("D")
    
    for col in ["quake_count", "tsunami_count"]:
        daily_series[col] = daily_series[col].fillna(0)
    
    for col in ["avg_magnitude", "max_magnitude", "avg_depth"]:
        daily_series[col] = daily_series[col].interpolate(method="time")
    
    # Variáveis derivadas
    daily_series["quake_count_ma7"] = daily_series["quake_count"].rolling(7, min_periods=1).mean()
    daily_series["quake_count_ma30"] = daily_series["quake_count"].rolling(30, min_periods=1).mean()
    daily_series["month"] = daily_series.index.month
    daily_series["day_of_week"] = daily_series.index.dayofweek
    
    # Normalização
    cols_scale = ["quake_count", "avg_magnitude", "max_magnitude", "avg_depth"]
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(daily_series[cols_scale].fillna(0))
    
    for i, col in enumerate(cols_scale):
        daily_series[f"{col}_z"] = scaled_values[:, i]
    
    daily_series.to_csv(TABELAS_DIR / "09_serie_temporal_diaria_tratada.csv")
    
    # ---- EDA
    stats_table = pd.DataFrame({
        "media": daily_series[cols_scale].mean(),
        "mediana": daily_series[cols_scale].median(),
        "desvio_padrao": daily_series[cols_scale].std(),
        "minimo": daily_series[cols_scale].min(),
        "maximo": daily_series[cols_scale].max(),
    })
    stats_table.to_csv(TABELAS_DIR / "10_estatisticas_descritivas.csv")
    
    # Série temporal plot
    plt.figure(figsize=(14, 5))
    plt.plot(daily_series.index, daily_series["quake_count"], label="Diária")
    plt.plot(daily_series.index, daily_series["quake_count_ma7"], label="MA7")
    plt.plot(daily_series.index, daily_series["quake_count_ma30"], label="MA30")
    plt.title("Evolução temporal da quantidade de terremotos por dia")
    plt.xlabel("Data")
    plt.ylabel("Quantidade")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / "03_serie_quake_count.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    # Magnitudes
    plt.figure(figsize=(14, 5))
    plt.plot(daily_series.index, daily_series["avg_magnitude"], label="Média")
    plt.plot(daily_series.index, daily_series["max_magnitude"], label="Máxima")
    plt.title("Magnitude média e máxima ao longo do tempo")
    plt.xlabel("Data")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / "04_serie_magnitudes.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    # Profundidade
    plt.figure(figsize=(14, 5))
    plt.plot(daily_series.index, daily_series["avg_depth"], label="Profundidade média")
    plt.title("Profundidade média ao longo do tempo")
    plt.xlabel("Data")
    plt.ylabel("Profundidade (km)")
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / "05_serie_profundidade.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    # Sazonalidade por mês
    monthly_seasonality = daily_series.groupby("month")["quake_count"].agg(["mean", "std", "count"])
    plt.figure(figsize=(12, 5))
    plt.bar(monthly_seasonality.index, monthly_seasonality["mean"], yerr=monthly_seasonality["std"])
    plt.title("Sazonalidade: atividade sísmica por mês")
    plt.xlabel("Mês")
    plt.ylabel("Média de eventos por dia")
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / "06_sazonalidade_mes.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    # Sazonalidade por dia da semana
    dow_seasonality = daily_series.groupby("day_of_week")["quake_count"].agg(["mean", "std", "count"])
    plt.figure(figsize=(12, 5))
    plt.bar(dow_seasonality.index, dow_seasonality["mean"], yerr=dow_seasonality["std"])
    plt.title("Sazonalidade: atividade sísmica por dia da semana")
    plt.xlabel("Dia da semana (0=segunda, 6=domingo)")
    plt.ylabel("Média de eventos por dia")
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / "07_sazonalidade_dia_semana.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    # Correlações
    corr = daily_series[["quake_count", "avg_magnitude", "max_magnitude", "avg_depth"]].corr()
    corr.to_csv(TABELAS_DIR / "11_correlacoes.csv")
    
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, aspect="auto", cmap="coolwarm")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Mapa de calor de correlação entre variáveis")
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / "08_correlacao.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    # Anomalias
    daily_series["quake_count_zscore"] = zscore(daily_series["quake_count"].fillna(0))
    anomalous_days = daily_series[np.abs(daily_series["quake_count_zscore"]) > 3]
    anomalous_days.to_csv(TABELAS_DIR / "12_dias_anomalos.csv")
    
    # Plot anomalias
    plt.figure(figsize=(14, 5))
    plt.plot(daily_series.index, daily_series["quake_count"], label="Quantidade de eventos")
    plt.scatter(anomalous_days.index, anomalous_days["quake_count"], color="red", s=100, label="Dias anômalos")
    plt.title("Detecção de anomalias na série temporal")
    plt.xlabel("Data")
    plt.ylabel("Quantidade de eventos")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / "09_anomalias_quake_count.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    # Decomposição
    try:
        serie_decomp = daily_series["quake_count"].copy().ffill()
        decomp = seasonal_decompose(serie_decomp, model="additive", period=30)
        
        fig = decomp.plot()
        fig.set_size_inches(14, 10)
        plt.suptitle("Decomposição da série temporal", y=1.02)
        plt.tight_layout()
        plt.savefig(GRAFICOS_DIR / "10_decomposicao_serie.png", dpi=200, bbox_inches="tight")
        plt.close()
        
        decomp_df = pd.DataFrame({
            "observado": decomp.observed,
            "tendencia": decomp.trend,
            "sazonalidade": decomp.seasonal,
            "residuo": decomp.resid
        })
        decomp_df.to_csv(TABELAS_DIR / "13_decomposicao_serie.csv")
    except Exception as e:
        pass
    
    # Top 15 dias com maior atividade
    top_dias = daily_series.nlargest(15, "quake_count")
    top_dias.to_csv(TABELAS_DIR / "14_top_dias_atividade.csv")
    
    # Resumo final
    resumo = pd.DataFrame({
        "registros_brutos": [len(df)],
        "registros_tratados_eventos": [len(df_tratado)],
        "inicio_serie_diaria": [str(daily_series.index.min().date())],
        "fim_serie_diaria": [str(daily_series.index.max().date())],
        "dias_na_serie": [len(daily_series)],
        "dias_anomalos": [len(anomalous_days)],
        "media_diaria_eventos": [daily_series["quake_count"].mean()],
        "mediana_diaria_eventos": [daily_series["quake_count"].median()],
        "maximo_diario_eventos": [daily_series["quake_count"].max()]
    })
    resumo.to_csv(TABELAS_DIR / "15_resumo_final.csv", index=False)
    
    # Boxplots para validação de outliers
    for col in ["magnitude", "depth", "latitude", "longitude", "tsunami"]:
        if col in df_tratado.columns:
            plt.figure(figsize=(10, 5))
            plt.boxplot(df_tratado[col].dropna())
            plt.title(f"Boxplot: {col}")
            plt.ylabel(col)
            plt.tight_layout()
            plt.savefig(GRAFICOS_DIR / f"boxplot_{col}.png", dpi=200, bbox_inches="tight")
            plt.close()
    
    # Boxplot para "felt" (reports)
    if "felt" in df.columns or "title" in df.columns:
        plt.figure(figsize=(10, 5))
        plt.title("Boxplot: distribuição genérica")
        plt.tight_layout()
        plt.savefig(GRAFICOS_DIR / "boxplot_felt.png", dpi=200, bbox_inches="tight")
        plt.close()

# ========================
#  CLASSIFICAÇÃO DE MAGNITUDE
# ========================
def classificar_magnitude(magnitude):
    """Classifica magnitude em categorias"""
    if magnitude < 3:
        return "Micro"
    elif magnitude < 4:
        return "Pequeno"
    elif magnitude < 5:
        return "Moderado"
    elif magnitude < 6:
        return "Grande"
    elif magnitude < 7:
        return "Muito grande"
    else:
        return "Terrível"

# ========================
#  CARREGAMENTO DE DADOS
# ========================
df_full = carregar_dados()

if df_full.empty:
    st.error("Nenhum dado disponível. Tente novamente mais tarde.")
    st.stop()

# Gera análises para sincronização
gerar_analises(df_full)

# ========================
#  FILTROS NA BARRA LATERAL
# ========================
st.sidebar.header("Filtros")

# Datas
data_min = df_full["time"].dt.date.min()
data_max = df_full["time"].dt.date.max()

date_range = st.sidebar.date_input(
    "Período",
    value=(data_min, data_max),
    min_value=data_min,
    max_value=data_max
)

# Magnitude
mag_min = float(df_full["magnitude"].min())
mag_max = float(df_full["magnitude"].max())

mag_range = st.sidebar.slider(
    "Range de Magnitude",
    min_value=mag_min,
    max_value=mag_max,
    value=(mag_min, mag_max),
    step=0.1
)

# Profundidade
depth_min = float(df_full["depth"].min())
depth_max = float(df_full["depth"].max())

depth_range = st.sidebar.slider(
    "Profundidade (km)",
    min_value=depth_min,
    max_value=depth_max,
    value=(depth_min, depth_max),
    step=1.0
)

# País
paises = sorted(df_full["pais"].unique())
paises_selecionados = st.sidebar.multiselect(
    "País/Região",
    options=paises,
    default=paises
)

# ========================
#  APLICAÇÃO DE FILTROS
# ========================
df_filtered = df_full[
    (df_full["time"].dt.date >= date_range[0]) &
    (df_full["time"].dt.date <= date_range[1]) &
    (df_full["magnitude"] >= mag_range[0]) &
    (df_full["magnitude"] <= mag_range[1]) &
    (df_full["depth"] >= depth_range[0]) &
    (df_full["depth"] <= depth_range[1]) &
    (df_full["pais"].isin(paises_selecionados))
]

# ========================
#  MÉTRICAS
# ========================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total de Eventos", len(df_filtered))

with col2:
    st.metric("Magnitude Média", f"{df_filtered['magnitude'].mean():.2f}")

with col3:
    st.metric("Magnitude Máxima", f"{df_filtered['magnitude'].max():.2f}")

with col4:
    st.metric("Profundidade Média", f"{df_filtered['depth'].mean():.1f} km")

with col5:
    st.metric("Tsunamis", int(df_filtered['tsunami'].sum()))

# ========================
#  ABAS DE VISUALIZAÇÃO
# ========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Mapa",
    "Série Temporal",
    "Estatísticas",
    "Distribuição",
    "Tabela",
    "Análise Completa"
])

# ---- TAB 1: MAPA
with tab1:
    st.subheader("Mapa de Terremotos")
    
    if len(df_filtered) > 0:
        map_center = [df_filtered["latitude"].mean(), df_filtered["longitude"].mean()]
        mapa = folium.Map(
            location=map_center,
            zoom_start=3,
            tiles="OpenStreetMap"
        )
        
        # Cria escala de cores por magnitude
        def get_color(mag):
            if mag < 3:
                return "blue"
            elif mag < 4:
                return "green"
            elif mag < 5:
                return "orange"
            elif mag < 6:
                return "red"
            else:
                return "darkred"
        
        # Adiciona marcadores com popups detalhados
        for idx, row in df_filtered.iterrows():
            # Formatação do popup com informações detalhadas
            popup_text = f"""
            <b>Informações do Terremoto</b><br>
            <b>Data/Hora:</b> {row['time'].strftime('%d/%m/%Y %H:%M:%S')}<br>
            <b>Local:</b> {row['place']}<br>
            <b>Magnitude:</b> {row['magnitude']:.2f}<br>
            <b>Profundidade:</b> {row['depth']:.1f} km<br>
            <b>Latitude:</b> {row['latitude']:.4f}<br>
            <b>Longitude:</b> {row['longitude']:.4f}<br>
            <b>País:</b> {row['pais']}<br>
            <b>Tsunami:</b> {'Sim' if row['tsunami'] == 1 else 'Não'}
            """
            
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=row["magnitude"] / 2,
                popup=folium.Popup(popup_text, max_width=300),
                color=get_color(row["magnitude"]),
                fill=True,
                fillColor=get_color(row["magnitude"]),
                fillOpacity=0.7,
                weight=2
            ).add_to(mapa)
        
        # Adiciona legenda de cores
        legend_html = '''
        <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 220px; height: 220px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px; border-radius: 5px;">
            <p style="margin: 5px; font-weight: bold; color: #666;">Legenda de Magnitude</p>
            <p style="margin: 5px; color: #666;"><span style="background-color: blue; padding: 5px 10px; border-radius: 3px; color: white;">●</span > Micro (&lt; 3.0)</p>
            <p style="margin: 5px; color: #666;"><span style="background-color: green; padding: 5px 10px; border-radius: 3px; color: white;">●</span > Pequeno (3.0 - 3.9)</p>
            <p style="margin: 5px; color: #666;"><span style="background-color: orange; padding: 5px 10px; border-radius: 3px; color: white;">●</span > Moderado (4.0 - 4.9)</p>
            <p style="margin: 5px; color: #666;"><span style="background-color: red; padding: 5px 10px; border-radius: 3px; color: white;">●</span > Grande (5.0 - 5.9)</p>
            <p style="margin: 5px; color: #666;"><span style="background-color: darkred; padding: 5px 10px; border-radius: 3px; color: white;">●</span > Muito Grande (≥ 6.0)</p>
            <p style="margin: 5px; font-size: 12px; color: #666;">Tamanho = Magnitude</p>
        </div>
        '''
        mapa.get_root().html.add_child(folium.Element(legend_html))
        
        st_folium(mapa, width=1400, height=600)
    else:
        st.warning("Nenhum dado disponível com os filtros selecionados.")

# ---- TAB 2: SÉRIE TEMPORAL
with tab2:
    st.subheader("Evolução Temporal")
    
    if len(df_filtered) > 1:
        df_temp = df_filtered.copy()
        df_temp["date"] = df_temp["time"].dt.floor("D")
        
        series_data = df_temp.groupby("date").agg({
            "magnitude": ["count", "mean", "max"],
            "tsunami": "sum"
        }).reset_index()
        
        series_data.columns = ["date", "quantidade", "mag_media", "mag_max", "tsunamis"]
        
        fig = px.line(series_data, x="date", y="quantidade", title="Quantidade de eventos por dia")
        fig.add_scatter(x=series_data["date"], y=series_data["mag_media"], name="Magnitude Média", yaxis="y2")
        fig.update_layout(
            yaxis2=dict(
                title="Magnitude",
                overlaying="y",
                side="right"
            ),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Dados insuficientes para exibir série temporal.")

# ---- TAB 3: ESTATÍSTICAS
with tab3:
    st.subheader("Análises Estatísticas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_mag = px.histogram(df_filtered, x="magnitude", nbins=30, title="Distribuição de Magnitudes")
        st.plotly_chart(fig_mag, use_container_width=True)
    
    with col2:
        fig_depth = px.histogram(df_filtered, x="depth", nbins=30, title="Distribuição de Profundidades")
        st.plotly_chart(fig_depth, use_container_width=True)
    
    # Boxplots
    col1, col2 = st.columns(2)
    
    with col1:
        fig_box_mag = px.box(df_filtered, y="magnitude", title="Boxplot de Magnitudes")
        st.plotly_chart(fig_box_mag, use_container_width=True)
    
    with col2:
        fig_box_depth = px.box(df_filtered, y="depth", title="Boxplot de Profundidades")
        st.plotly_chart(fig_box_depth, use_container_width=True)

# ---- TAB 4: DISTRIBUIÇÃO GEOGRÁFICA
with tab4:
    st.subheader("Distribuição Geográfica")
    
    # Top países
    top_paises = df_filtered["pais"].value_counts().head(15)
    fig_paises = px.bar(x=top_paises.values, y=top_paises.index, orientation="h", title="Top 15 Países/Regiões")
    st.plotly_chart(fig_paises, use_container_width=True)
    
    # Scatter lat/lon
    fig_scatter = px.scatter(
        df_filtered,
        x="longitude",
        y="latitude",
        size="magnitude",
        color="depth",
        title="Distribuição espacial dos eventos"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---- TAB 5: TABELA DE DADOS
with tab5:
    st.subheader("Dados Detalhados")
    
    # Preparar dados para exibição
    df_display = df_filtered[[
        "time", "magnitude", "depth", "latitude", "longitude", "pais", "place", "tsunami"
    ]].copy()
    
    df_display = df_display.sort_values("time", ascending=False)
    df_display["time"] = df_display["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    st.dataframe(df_display, use_container_width=True)
    
    # Download CSV
    csv = df_display.to_csv(index=False)
    st.download_button(
        label="Baixar dados como CSV",
        data=csv,
        file_name=f"terremotos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# ---- TAB 6: ANÁLISE COMPLETA
with tab6:
    st.subheader("Análise Completa - 3 Etapas")
    
    st.write("""
    Clique no botão abaixo para acessar o dashboard completo com as 3 etapas de análise:
    - Etapa 1: Auditoria (Dados Brutos)
    - Etapa 2: Tratamento (Dados Limpos)
    - Etapa 3: EDA (Análise Exploratória)
    """)
    
    if st.button("Abrir Análise Completa", key="analysis_link"):
        st.info("Você está sendo redirecionado para a página de análise completa...")
        st.markdown("""
            <meta http-equiv="refresh" content="0; url='http://localhost:8503'" />
        """, unsafe_allow_html=True)
