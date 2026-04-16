import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# ========================
#  CONFIGURAÇÃO DA PÁGINA
# ========================
st.set_page_config(
    page_title="Análise Completa - Atividade Sísmica",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    body, p, h1, h2, h3, h4, h5, h6, label, div, span {
        color: #ffffff !important;
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

st.title("Análise Completa de Atividade Sísmica (2018-2025)")
st.markdown("**Etapas 1, 2 e 3**: Auditoria → Tratamento → Análise Exploratória")

# ========================
#  CARREGAMENTO DE DADOS
# ========================
OUTPUT_DIR = Path("outputs_n1_old")
TABELAS_DIR = OUTPUT_DIR / "tabelas"
GRAFICOS_DIR = OUTPUT_DIR / "graficos"

@st.cache_data
def carregar_dados():
    """Carrega todos os CSVs gerados"""
    dados = {}
    
    # Auditoria
    dados['missing'] = pd.read_csv(TABELAS_DIR / "02_missing_values.csv") if (TABELAS_DIR / "02_missing_values.csv").exists() else None
    dados['duplicidades'] = pd.read_csv(TABELAS_DIR / "03_duplicidades.csv") if (TABELAS_DIR / "03_duplicidades.csv").exists() else None
    dados['gaps'] = pd.read_csv(TABELAS_DIR / "04_estatisticas_intervalos_temporais.csv") if (TABELAS_DIR / "04_estatisticas_intervalos_temporais.csv").exists() else None
    dados['outliers'] = pd.read_csv(TABELAS_DIR / "06_outliers_iqr.csv") if (TABELAS_DIR / "06_outliers_iqr.csv").exists() else None
    dados['inconsistencias'] = pd.read_csv(TABELAS_DIR / "07_inconsistencias_logicas.csv") if (TABELAS_DIR / "07_inconsistencias_logicas.csv").exists() else None
    
    # Tratamento e EDA
    dados['bruto'] = pd.read_csv(TABELAS_DIR / "01_dataset_bruto.csv") if (TABELAS_DIR / "01_dataset_bruto.csv").exists() else None
    dados['tratado'] = pd.read_csv(TABELAS_DIR / "08_dataset_eventos_tratado.csv") if (TABELAS_DIR / "08_dataset_eventos_tratado.csv").exists() else None
    dados['serie_diaria'] = pd.read_csv(TABELAS_DIR / "09_serie_temporal_diaria_tratada.csv") if (TABELAS_DIR / "09_serie_temporal_diaria_tratada.csv").exists() else None
    dados['stats'] = pd.read_csv(TABELAS_DIR / "10_estatisticas_descritivas.csv") if (TABELAS_DIR / "10_estatisticas_descritivas.csv").exists() else None
    dados['correlacao'] = pd.read_csv(TABELAS_DIR / "11_correlacoes.csv", index_col=0) if (TABELAS_DIR / "11_correlacoes.csv").exists() else None
    dados['anomalias'] = pd.read_csv(TABELAS_DIR / "12_dias_anomalos.csv") if (TABELAS_DIR / "12_dias_anomalos.csv").exists() else None
    dados['decomposicao'] = pd.read_csv(TABELAS_DIR / "13_decomposicao_serie.csv") if (TABELAS_DIR / "13_decomposicao_serie.csv").exists() else None
    dados['resumo'] = pd.read_csv(TABELAS_DIR / "15_resumo_final.csv") if (TABELAS_DIR / "15_resumo_final.csv").exists() else None
    
    return dados

dados = carregar_dados()

# ========================
#  NAVEGAÇÃO COM ABAS
# ========================
tab_home, tab_etapa1, tab_etapa2, tab_etapa3, tab_comparativo = st.tabs([
    "Resumo",
    "Etapa 1: Auditoria",
    "Etapa 2: Tratamento",
    "Etapa 3: EDA",
    "Comparativo"
])

# ========================
#  TAB: HOME/RESUMO
# ========================
with tab_home:
    st.header("Resumo Executivo")
    
    if dados['resumo'] is not None:
        resumo = dados['resumo'].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Registros Brutos", f"{int(resumo['registros_brutos']):,}")
        with col2:
            st.metric("Registros Tratados", f"{int(resumo['registros_tratados_eventos']):,}")
        with col3:
            st.metric("Período", f"{resumo['inicio_serie_diaria']} a {resumo['fim_serie_diaria']}")
        with col4:
            st.metric("Dias Analisados", f"{int(resumo['dias_na_serie'])}")
        
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Média Diária de Eventos", f"{resumo['media_diaria_eventos']:.1f}")
        with col2:
            st.metric("Mediana Diária", f"{resumo['mediana_diaria_eventos']:.1f}")
        with col3:
            st.metric("Máximo Diário", f"{resumo['maximo_diario_eventos']:.0f}")
        
        st.divider()
        
        with st.expander("📋 Fluxo de Processamento"):
            st.markdown("""
            ### Fluxo de Dados:
            1. **Extração**: Dados da API USGS (2018-2025) → 20,000 eventos
            2. **Auditoria**: Diagnóstico de qualidade (missing, outliers, gaps)
            3. **Tratamento**: Limpeza, normalização, derivadas
            4. **EDA**: Análise exploratória com visualizações
            5. **Apresentação**: Visualizações interativas no dashboard
            """)

# ========================
#  TAB: ETAPA 1 - AUDITORIA
# ========================
with tab_etapa1:
    st.header("Etapa 1: Auditoria de Dados")
    st.markdown("Diagnóstico completo da qualidade da série temporal")
    
    # Sub-abas para Auditoria
    sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
        "Valores Ausentes",
        "Duplicidades",
        "Gaps Temporais",
        "Outliers",
        "Inconsistências"
    ])
    
    # ---- Valores Ausentes
    with sub_tab1:
        st.subheader("Análise de Valores Ausentes (Missing Values)")
        
        if dados['missing'] is not None:
            # Tabela
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Tabela Resumida")
                missing_display = dados['missing'].head(10)[['coluna', 'qtd_missing', 'pct_missing']].copy()
                missing_display['pct_missing'] = missing_display['pct_missing'].round(2).astype(str) + "%"
                st.dataframe(missing_display, use_container_width=True)
            
            with col2:
                st.markdown("### Visualização")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=dados['missing']['coluna'],
                    y=dados['missing']['qtd_missing'],
                    marker=dict(color='#ff7f0e')
                ))
                fig.update_layout(
                    title="Quantidade de Valores Ausentes por Coluna",
                    xaxis_title="Coluna",
                    yaxis_title="Quantidade",
                    height=400,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Gráfico PNG se existir
            if (GRAFICOS_DIR / "01_missing_values.png").exists():
                st.divider()
                st.markdown("### Gráfico Detalhado")
                img = Image.open(GRAFICOS_DIR / "01_missing_values.png")
                st.image(img, use_column_width=True)
    
    # ---- Duplicidades
    with sub_tab2:
        st.subheader("Análise de Duplicidades")
        
        if dados['duplicidades'] is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=dados['duplicidades']['tipo'],
                    y=dados['duplicidades']['quantidade'],
                    marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                ))
                fig.update_layout(
                    title="Duplicidades Encontradas",
                    xaxis_title="Tipo",
                    yaxis_title="Quantidade",
                    height=400,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Resumo")
                for _, row in dados['duplicidades'].iterrows():
                    tipo_clean = row['tipo'].replace('_', ' ').title()
                    st.info(f"**{tipo_clean}**: {int(row['quantidade'])} ocorrências")
    
    # ---- Gaps Temporais
    with sub_tab3:
        st.subheader("Análise de Intervalos Temporais (Gaps)")
        
        if (GRAFICOS_DIR / "02_intervalos_temporais.png").exists():
            img = Image.open(GRAFICOS_DIR / "02_intervalos_temporais.png")
            st.image(img, use_column_width=True)
        
        if dados['gaps'] is not None:
            st.markdown("### Estatísticas dos Intervalos entre Eventos (em horas)")
            stats_df = dados['gaps'].T
            st.dataframe(stats_df, use_container_width=True)
            
            st.info("""
            **Interpretação**: A série sísmica é naturalmente irregular. 
            Eventos não seguem frequência regular - há períodos com muita atividade e períodos de calma.
            """)
    
    # ---- Outliers
    with sub_tab4:
        st.subheader("Análise de Outliers (Método IQR)")
        
        if dados['outliers'] is not None:
            coluna_sel = st.selectbox(
                "Selecione uma variável para visualizar",
                options=dados['outliers']['variavel'].unique()
            )
            
            outlier_data = dados['outliers'][dados['outliers']['variavel'] == coluna_sel].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Quantidade de Outliers", int(outlier_data['qtd_outliers']))
                st.metric("Limite Inferior (Q1 - 1.5×IQR)", f"{outlier_data['limite_inferior']:.2f}")
                st.metric("Limite Superior (Q3 + 1.5×IQR)", f"{outlier_data['limite_superior']:.2f}")
            
            with col2:
                st.metric("Q1 (25º percentil)", f"{outlier_data['q1']:.2f}")
                st.metric("Q3 (75º percentil)", f"{outlier_data['q3']:.2f}")
                st.metric("IQR (Intervalo Interquartil)", f"{outlier_data['iqr']:.2f}")
            
            # Gráfico boxplot
            if (GRAFICOS_DIR / f"boxplot_{coluna_sel}.png").exists():
                st.divider()
                img = Image.open(GRAFICOS_DIR / f"boxplot_{coluna_sel}.png")
                st.image(img, use_column_width=True)
    
    # ---- Inconsistências
    with sub_tab5:
        st.subheader("Inconsistências Lógicas Identificadas")
        
        if dados['inconsistencias'] is not None:
            for _, row in dados['inconsistencias'].iterrows():
                problema = row['problema'].replace('_', ' ').upper()
                quantidade = int(row['quantidade'])
                
                if quantidade > 0:
                    st.error(f"🔴 **{problema}**: {quantidade} ocorrências")
                else:
                    st.success(f"✅ **{problema}**: OK (0 ocorrências)")

# ========================
#  TAB: ETAPA 2 - TRATAMENTO
# ========================
with tab_etapa2:
    st.header("Etapa 2: Tratamento e Preparação (ETL)")
    st.markdown("Transformação e preparação dos dados para análise")
    
    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "Resumo ETL",
        "Variáveis Derivadas",
        "Normalização"
    ])
    
    # ---- Resumo ETL
    with sub_tab1:
        col1, col2, col3 = st.columns(3)
        
        if dados['bruto'] is not None and dados['tratado'] is not None:
            registros_removidos = len(dados['bruto']) - len(dados['tratado'])
            pct_reducao = (registros_removidos / len(dados['bruto']) * 100)
            
            with col1:
                st.metric("Registros Originais", f"{len(dados['bruto']):,}")
            with col2:
                st.metric("Registros Após Limpeza", f"{len(dados['tratado']):,}")
            with col3:
                st.metric("Registros Removidos", f"{registros_removidos:,} ({pct_reducao:.1f}%)")
        
        st.divider()
        
        st.markdown("### Transformações Aplicadas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Retirada de Dados
            - ✅ Timestamps inválidos removidos
            - ✅ Registros duplicados eliminados
            - ✅ IDs duplicados filtrados
            - ✅ Magnitude, latitude, longitude, profundidade: obrigatórios
            """)
        
        with col2:
            st.markdown("""
            #### Preenchimento de Dados
            - ✅ `felt` → preenchida com 0 (sem relato)
            - ✅ `tsunami` → preenchida com 0 (sem indicação)
            - ✅ `sig` → imputada pela mediana
            - ✅ `place`, `mag_type` → preenchidas com "não informado"
            """)
        
        st.divider()
        
        st.markdown("### Filtragem de Inconsistências Lógicas")
        st.markdown("""
        - Latitude: -90° a 90° ✅
        - Longitude: -180° a 180° ✅
        - Magnitude: ≥ 0 ✅
        - Profundidade: ≥ 0 ✅
        """)
    
    # ---- Variáveis Derivadas
    with sub_tab2:
        st.subheader("Variáveis Derivadas Criadas")
        
        derives = [
            ("quake_count_ma7", "Média Móvel 7 Dias", "Suavização de curto prazo da quantidade de eventos"),
            ("quake_count_ma30", "Média Móvel 30 Dias", "Suavização de longo prazo da quantidade de eventos"),
            ("avg_magnitude_pct_change", "Variação % Magnitude Diária", "Mudança percentual dia-a-dia da magnitude média"),
            ("month", "Mês do Ano", "Identificação de sazonalidade mensal (1-12)"),
            ("day_of_week", "Dia da Semana", "Padrão de dias: Segunda(0) a Domingo(6)"),
            ("is_weekend", "Flag Fim de Semana", "Binária: 1=Sábado/Domingo, 0=Semana")
        ]
        
        for i, (var, descricao, uso) in enumerate(derives, 1):
            with st.expander(f"**{i}. {descricao}** ({var})"):
                st.markdown(f"**Descrição**: {uso}")
        
        st.divider()
        
        st.info("✅ Total de 6 variáveis derivadas + z-score normalizados de 6 variáveis = 12 novas features")
    
    # ---- Normalização
    with sub_tab3:
        st.subheader("Normalização (StandardScaler - Z-score)")
        
        st.markdown("""
        Foram normalizadas as seguintes variáveis usando StandardScaler (média=0, desvio padrão=1):
        """)
        
        cols_normalized = [
            "quake_count_z",
            "avg_magnitude_z",
            "max_magnitude_z",
            "avg_depth_z",
            "total_felt_z",
            "avg_sig_z"
        ]
        
        col1, col2, col3 = st.columns(3)
        
        for i, col in enumerate(cols_normalized):
            with [col1, col2, col3][i % 3]:
                st.markdown(f"✅ `{col}`")
        
        st.info("**Propósito**: Colocar variáveis em escala comparável para análises posteriores e ML")

# ========================
#  TAB: ETAPA 3 - EDA
# ========================
with tab_etapa3:
    st.header("Etapa 3: Análise Exploratória (EDA)")
    st.markdown("Extração de valor e padrões dos dados")
    
    sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5, sub_tab6 = st.tabs([
        "Estatísticas",
        "Série Temporal",
        "Sazonalidade",
        "Correlações",
        "Anomalias",
        "Decomposição"
    ])
    
    # ---- Estatísticas
    with sub_tab1:
        st.subheader("Estatísticas Descritivas")
        
        if dados['stats'] is not None:
            st.dataframe(dados['stats'].round(3), use_container_width=True)
            
            st.markdown("### Interpretação")
            st.markdown("""
            - **Média**: Valor central típico da variável
            - **Mediana**: Valor que divide a distribuição ao meio
            - **Desvio Padrão**: Variabilidade ao redor da média
            - **Amplitude**: Diferença entre máximo e mínimo
            """)
    
    # ---- Série Temporal
    with sub_tab2:
        st.subheader("Evolução Temporal")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if (GRAFICOS_DIR / "03_serie_quake_count.png").exists():
                st.markdown("#### Quantidade de Eventos + Médias Móveis")
                img = Image.open(GRAFICOS_DIR / "03_serie_quake_count.png")
                st.image(img, use_column_width=True)
        
        with col2:
            if (GRAFICOS_DIR / "04_serie_magnitudes.png").exists():
                st.markdown("#### Magnitudes Média e Máxima")
                img = Image.open(GRAFICOS_DIR / "04_serie_magnitudes.png")
                st.image(img, use_column_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if (GRAFICOS_DIR / "05_serie_profundidade.png").exists():
                st.markdown("#### Profundidade Média Diária")
                img = Image.open(GRAFICOS_DIR / "05_serie_profundidade.png")
                st.image(img, use_column_width=True)
        
        with col2:
            st.markdown("#### Insights Temporais")
            st.markdown("""
            **Observações da série**:
            - Tendência: Há variações significativas ao longo do período
            - Sazonalidade: Padrões cíclicos em diferentes escalas
            - Ruído: Flutuações aleatórias em torno da tendência
            - Anomalias: Picos de atividade sísmica em período específicos
            """)
    
    # ---- Sazonalidade
    with sub_tab3:
        st.subheader("Padrões Sazonais")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if (GRAFICOS_DIR / "06_sazonalidade_mes.png").exists():
                st.markdown("#### Por Mês")
                img = Image.open(GRAFICOS_DIR / "06_sazonalidade_mes.png")
                st.image(img, use_column_width=True)
        
        with col2:
            if (GRAFICOS_DIR / "07_sazonalidade_dia_semana.png").exists():
                st.markdown("#### Por Dia da Semana")
                img = Image.open(GRAFICOS_DIR / "07_sazonalidade_dia_semana.png")
                st.image(img, use_column_width=True)
        
        st.info("""
        **Interpretação**: A atividade sísmica não segue padrões sazonais significativos previstos,
        indicando que eventos sísmicos são fenômenos naturais com baixa previsibilidade temporal.
        """)
    
    # ---- Correlações
    with sub_tab4:
        st.subheader("Correlações Entre Variáveis")
        
        if (GRAFICOS_DIR / "08_correlacao.png").exists():
            img = Image.open(GRAFICOS_DIR / "08_correlacao.png")
            st.image(img, use_column_width=True)
        
        if dados['correlacao'] is not None:
            st.markdown("### Matriz de Correlação (Valores Numéricos)")
            st.dataframe(dados['correlacao'].round(3), use_container_width=True)
            
            st.markdown("""
            **Principais correlações encontradas**:
            - Correlação forte entre `max_magnitude` e `avg_magnitude` (esperado)
            - Correlação positiva entre `quake_count` e outras variáveis (eventos maiores tendem a vir em clusters)
            - Correlação fraca com `total_felt` (poucos eventos são sentidos/reportados)
            """)
    
    # ---- Anomalias
    with sub_tab5:
        st.subheader("Identificação de Anomalias")
        
        if (GRAFICOS_DIR / "09_anomalias_quake_count.png").exists():
            img = Image.open(GRAFICOS_DIR / "09_anomalias_quake_count.png")
            st.image(img, use_column_width=True)
        
        if dados['anomalias'] is not None:
            st.markdown(f"### Dias Anômalos Detectados (|Z-Score| > 3): {len(dados['anomalias'])} eventos")
            
            if len(dados['anomalias']) > 0:
                anomalias_display = dados['anomalias'][['quake_count', 'quake_count_zscore']].head(20).copy()
                anomalias_display = anomalias_display.round(3)
                st.dataframe(anomalias_display, use_container_width=True)
            else:
                st.info("Nenhuma anomalia detectada com critério Z > 3")
    
    # ---- Decomposição
    with sub_tab6:
        st.subheader("Decomposição Sazonal da Série Temporal")
        
        if (GRAFICOS_DIR / "10_decomposicao_serie.png").exists():
            img = Image.open(GRAFICOS_DIR / "10_decomposicao_serie.png")
            st.image(img, use_column_width=True)
        
        st.markdown("""
        ### Componentes da Decomposição:
        - **Observado**: Série original de quantidade diária de queremos
        - **Tendência**: Padrão de longo prazo (suavizado com período=30)
        - **Sazonalidade**: Ciclos regulares intra-mensais
        - **Resíduo**: Variações aleatórias e ruído
        """)

# ========================
#  TAB: COMPARATIVO ANTES/DEPOIS
# ========================
with tab_comparativo:
    st.header("Comparativo: Antes vs. Depois do Tratamento")
    
    if dados['bruto'] is not None and dados['tratado'] is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Antes do Tratamento (Bruto)")
            st.metric("Total de Registros", f"{len(dados['bruto']):,}")
            st.metric("Colunas", len(dados['bruto'].columns))
            st.metric("Missing Values", f"{dados['bruto'].isna().sum().sum():,} células")
            
            if 'time' in dados['bruto'].columns:
                st.metric("Período", f"{len(dados['bruto'].iloc[:, 0].unique())} únicos")
        
        with col2:
            st.markdown("### Depois do Tratamento (Limpo)")
            st.metric("Total de Registros", f"{len(dados['tratado']):,}")
            st.metric("Colunas", len(dados['tratado'].columns))
            st.metric("Missing Values", f"{dados['tratado'].isna().sum().sum():,} células")
            
            if 'time' in dados['tratado'].columns:
                st.metric("Período", f"{len(dados['tratado'].iloc[:, 0].unique())} únicos")
        
        st.divider()
        
        # Visualização de redução
        dados_reducao = pd.DataFrame({
            'Métrica': ['Registros', 'Missing Values'],
            'Antes': [
                len(dados['bruto']),
                dados['bruto'].isna().sum().sum()
            ],
            'Depois': [
                len(dados['tratado']),
                dados['tratado'].isna().sum().sum()
            ]
        })
        
        fig = go.Figure(data=[
            go.Bar(name='Antes', x=dados_reducao['Métrica'], y=dados_reducao['Antes'], marker_color='#d62728'),
            go.Bar(name='Depois', x=dados_reducao['Métrica'], y=dados_reducao['Depois'], marker_color='#2ca02c')
        ])
        
        fig.update_layout(
            title="Impacto do Tratamento ETL",
            barmode='group',
            height=400,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Impacto das Transformações:
        - ✅ Redução de ruído e inconsistências
        - ✅ Série temporal regularizada
        - ✅ 12 novas features derivadas/normalizadas
        - ✅ Dados prontos para análise estatística avançada
        """)

# ========================
#  FOOTER
# ========================
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("📁 **Dados**: `/outputs_n1_old/tabelas/` (15 CSVs)")

with col2:
    st.markdown("🖼️ **Gráficos**: `/outputs_n1_old/graficos/` (16 PNGs)")

with col3:
    st.markdown("📅 **Período**: 2018-01-01 a 2025-12-31")

st.markdown("---")
st.caption("Dashboard interativo de análise sísmica completa - Etapas 1, 2 e 3 de Auditoria, Tratamento e EDA")
