# Análise de Atividade Sísmica Global

Um dashboard interativo e abrangente para análise em tempo real de dados de terremotos com visualizações geográficas, estatísticas e análises exploratórias.

---

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Características](#características)
- [Tecnologias](#tecnologias)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Dashboards](#dashboards)
- [Fluxo de Dados](#fluxo-de-dados)
- [Etapas de Análise](#etapas-de-análise)
- [Requisitos](#requisitos)

---

## 🌍 Visão Geral

Este projeto implementa uma solução completa de análise de atividade sísmica global, integrando:

- **Dados em tempo real** da API USGS Earthquake Hazards
- **Dashboard interativo** com filtros e múltiplas visualizações
- **Análise completa** em 3 etapas: Auditoria, Tratamento e EDA
- **Sincronização automática** de dados entre dashboards
- **Visualizações geográficas** com mapas interativos (Folium)
- **Análises estatísticas** com Plotly e Matplotlib

---

## ✨ Características

### 🗺️ Dashboard Interativo (app.py)

- **Mapa Interativo**: Visualização geográfica com código de cores por magnitude
  - Legenda de cores: Micro (azul) → Muito grande (vermelho escuro)
  - Popups com informações completas ao clicar em cada terremoto
- **Série Temporal**: Evolução da quantidade e magnitude de eventos ao longo do tempo
- **Estatísticas**: Histogramas e boxplots de magnitudes e profundidades
- **Distribuição Geográfica**: Top 15 países/regiões com maior atividade sísmica
- **Tabela de Dados**: Visualização completa com opção de download em CSV
- **Filtros Avançados**:
  - Período de datas
  - Range de magnitude
  - Range de profundidade
  - Seleção por país/região

### 📊 Dashboard de Análise (analysis_report.py)

- **Resumo Executivo**: Métricas principais e fluxo de dados
- **Etapa 1: Auditoria**
  - Valores ausentes (missing values)
  - Duplicidades
  - Gaps temporais
  - Outliers (método IQR)
  - Inconsistências lógicas
- **Etapa 2: Tratamento**
  - Resumo do ETL
  - Variáveis derivadas
  - Normalização de dados
- **Etapa 3: EDA (Análise Exploratória)**
  - Estatísticas descritivas
  - Série temporal com médias móveis
  - Sazonalidade (por mês e dia da semana)
  - Correlações
  - Detecção de anomalias
  - Decomposição da série
- **Comparativo**: Impacto antes/depois do tratamento

---

## 🛠️ Tecnologias

```
Python 3.13
├── Streamlit          # Framework web interativo
├── Pandas             # Manipulação de dados
├── NumPy              # Operações numéricas
├── Plotly             # Gráficos interativos
├── Folium             # Mapas interativos
├── Matplotlib         # Visualizações estáticas
├── Scikit-learn       # Normalização, estatísticas
├── SciPy              # Análises estatísticas avançadas
└── Statsmodels        # Decomposição de séries temporais

API USGS Earthquake Hazards
```

---

## 📦 Instalação

### Pré-requisitos

- Python 3.13+
- pip (gerenciador de pacotes)

### Passos

1. **Clone ou baixe o projeto**

```bash
cd "c:\Users\joao.vitor\Desktop\teste py\earthquake"
```

2. **Instale as dependências**

```bash
pip install streamlit pandas numpy plotly folium streamlit-folium requests scipy scikit-learn statsmodels matplotlib
```

Ou, se usar um arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🚀 Como Usar

### Executar o Dashboard Principal

**Opção 1: Via Linha de Comando**

```powershell
C:/Users/joao.vitor/AppData/Local/Microsoft/WindowsApps/python3.13.exe -m streamlit run app.py
```

**Opção 2: Duplo-clique no arquivo batch**

```
rodar_app.bat
```

O app abrirá em: `http://localhost:8504`

### Executar o Dashboard de Análise

```powershell
C:/Users/joao.vitor/AppData/Local/Microsoft/WindowsApps/python3.13.exe -m streamlit run analysis_report.py --server.port 8505
```

O app abrirá em: `http://localhost:8505`

---

## 📁 Estrutura do Projeto

```
earthquake/
│
├── app.py                          # Dashboard principal com filtros
├── analysis_report.py              # Dashboard de análise (3 etapas)
├── rodar_app.bat                   # Script para executar app.py
│
├── outputs_n1/                     # Gerado automaticamente
│   ├── tabelas/                    # 15 arquivos CSV
│   │   ├── 01_dataset_bruto.csv
│   │   ├── 02_missing_values.csv
│   │   ├── 03_duplicidades.csv
│   │   ├── 04_estatisticas_intervalos_temporais.csv
│   │   ├── 05_gaps_longos.csv
│   │   ├── 06_outliers_iqr.csv
│   │   ├── 07_inconsistencias_logicas.csv
│   │   ├── 08_dataset_eventos_tratado.csv
│   │   ├── 09_serie_temporal_diaria_tratada.csv
│   │   ├── 10_estatisticas_descritivas.csv
│   │   ├── 11_correlacoes.csv
│   │   ├── 12_dias_anomalos.csv
│   │   ├── 13_decomposicao_serie.csv
│   │   ├── 14_top_dias_atividade.csv
│   │   └── 15_resumo_final.csv
│   │
│   └── graficos/                   # 14 arquivos PNG
│       ├── 01_missing_values.png
│       ├── 03_serie_quake_count.png
│       ├── 04_serie_magnitudes.png
│       ├── 05_serie_profundidade.png
│       ├── 06_sazonalidade_mes.png
│       ├── 07_sazonalidade_dia_semana.png
│       ├── 08_correlacao.png
│       ├── 09_anomalias_quake_count.png
│       ├── 10_decomposicao_serie.png
│       └── boxplot_*.png (5 variáveis)
│
├── .gitignore                      # Arquivos ignorados pelo Git
└── README.md                       # Este arquivo
```

---

## 📊 Dashboards

### Dashboard Principal (app.py) - 6 Abas

| Aba                 | Funcionalidade                              | Interatividade                 |
| ------------------- | ------------------------------------------- | ------------------------------ |
| 🗺️ Mapa             | Visualização geográfica com código de cores | Clique para popup com detalhes |
| 📈 Série Temporal   | Evolução temporal de quantidade e magnitude | Zoom, pan, hover               |
| 📊 Estatísticas     | Histogramas e boxplots                      | Sortear, filtrar               |
| 🌍 Distribuição     | Top 15 países + scatter plot espacial       | Hover, zoom                    |
| 📋 Tabela           | Dados completos com sorting                 | Download CSV                   |
| 📁 Análise Completa | Link para dashboard de análise              | Redirecionamento               |

### Dashboard de Análise (analysis_report.py) - 5 Abas

| Aba            | Sub-análises                                                    | Quantidade     |
| -------------- | --------------------------------------------------------------- | -------------- |
| 📌 Resumo      | Métricas principais + fluxo de dados                            | 5 seções       |
| 🔍 Auditoria   | Missing, duplicidades, gaps, outliers, inconsistências          | 5 sub-abas     |
| 🛠️ Tratamento  | ETL summary, variáveis derivadas, normalização                  | 3 sub-abas     |
| 📉 EDA         | Stats, série, sazonalidade, correlação, anomalias, decomposição | 6 sub-abas     |
| ⚖️ Comparativo | Antes/depois do tratamento                                      | Impacto visual |

---

## 🔄 Fluxo de Dados

```
┌─────────────────────────────────────────┐
│   USGS Earthquake Hazards API           │
│   (2023-present, minmagnitude: 2.5)     │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│   app.py - carregar_dados()             │
│   • Parse GeoJSON                       │
│   • Extrai país do lugar                │
│   • ~10.000 registros                   │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│   gerar_analises() [CACHE 1 HORA]       │
│   • Auditoria (5 análises)              │
│   • Tratamento (limpeza + features)     │
│   • EDA (estatísticas + visualizações)  │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│   outputs_n1/ (FONTE ÚNICA)             │
│   • 15 CSVs (tabelas análises)          │
│   • 14 PNGs (visualizações)             │
└────────────────────┬────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
    ┌──────────┐          ┌──────────┐
    │ app.py   │          │analysis_ │
    │Dashboard │          │report.py │
    │(interativa)         │(análise) │
    │          │          │          │
    │6 abas    │          │5 abas    │
    │Filtros   │          │3 etapas  │
    │Mapas     │          │Detalhada │
    └──────────┘          └──────────┘
```

---

## 🔬 Etapas de Análise

### Etapa 1: Auditoria

Análise dos dados brutos para identificar problemas:

- **Valores Ausentes**: Quantidade e percentual por coluna
- **Duplicidades**: Linhas duplicadas, IDs duplicados, timestamps duplicados
- **Gaps Temporais**: Intervalos entre eventos, estatísticas
- **Outliers**: Detecção via método IQR (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
- **Inconsistências**: Valores negativos, coordenadas inválidas

**Saída**: 5 arquivos CSV + 1 PNG

### Etapa 2: Tratamento

Limpeza e transformação dos dados:

- **Remoção de Nulos**: Apenas em magnitude, latitude, longitude, depth
- **Validação de Coordenadas**: Latitude [-90, 90], Longitude [-180, 180]
- **Validação de Magnitude**: Não negativa
- **Validação de Profundidade**: Não negativa
- **Variáveis Derivadas**:
  - Médias móveis 7 dias (MA7)
  - Médias móveis 30 dias (MA30)
  - Mês e dia da semana
- **Normalização**: Z-score para variáveis numéricas

**Saída**: Dataset tratado + série temporal com features

### Etapa 3: EDA (Análise Exploratória)

Descoberta de padrões e insights:

- **Estatísticas Descritivas**: Média, mediana, desvio padrão, min, max
- **Série Temporal**: Evolução demanda com MA
- **Sazonalidade**: Variação por mês e dia da semana
- **Correlações**: Matriz de correlação entre variáveis
- **Anomalias**: Dias com z-score > 3
- **Decomposição**: Tendência, sazonalidade, resíduo

**Saída**: 6 sub-análises + múltiplas visualizações

---

## 📈 Dados Gerados

### Exemplo de Resumo Final

```
Registros Brutos:               10.000
Registros Tratados:             9.981 (~99.8%)
Período:                        2025-12-06 até 2026-04-16
Dias na Série:                  132
Dias Anômalos:                  2
Média Diária de Eventos:        75.6
Mediana Diária:                 71.0
Máximo Diário:                  366
```

---

## 📥 Requisitos do Sistema

- **Python**: 3.13+
- **RAM**: Mínimo 2GB (recomendado 4GB+)
- **Disco**: ~500MB (com outputs_n1/)
- **Internet**: Necessária para API USGS
- **Navegador**: Chrome, Firefox, Edge, Safari (moderno)

---

## 🔧 Configurações

### Parâmetros de API USGS (em app.py)

```python
API_PARAMS = {
    "format": "geojson",
    "starttime": "2023-01-01",      # Altere aqui para outro período
    "minmagnitude": 2.5,             # Altere aqui para filtrar magnitude
    "limit": 10000                   # Máximo de registros
}
```

### Cache de Dados

```python
@st.cache_data(ttl=3600)  # TTL em segundos (1 hora)
def carregar_dados():
    ...
```

---

## 📝 Exemplos de Uso

### Filtrar por Magnitude

1. Abrir app.py em `http://localhost:8504`
2. Na barra lateral, ajustar "Range de Magnitude"
3. Dashboard atualiza automaticamente

### Filtrar por Período

1. Na barra lateral, clicar em "Período"
2. Selecionar datas de início e fim
3. Mapa e gráficos se atualizam

### Explorar Detalhes de um Terremoto

1. Na aba "Mapa", clicar em uma bolinha colorida
2. Popup aparece com:
   - Data e hora exato
   - Local completo
   - Magnitude
   - Profundidade
   - Coordenadas
   - País
   - Indicação de tsunami

### Baixar Dados Filtrados

1. Na aba "Tabela", aplicar filtros desejados
2. Clicar em "Baixar dados como CSV"
3. Arquivo é baixado com timestamp

---

## 🐛 Troubleshooting

### Problema: "streamlit não é reconhecido"

**Solução**: Use o comando completo com Python diretamente

```powershell
C:/Users/joao.vitor/AppData/Local/Microsoft/WindowsApps/python3.13.exe -m streamlit run app.py
```

### Problema: "API não responde"

**Solução**:

- Verificar conexão à internet
- Aguardar 1 hora para liberar cache
- Tentar manualmente em: https://earthquake.usgs.gov/fdsnws/event/1/query

### Problema: "Erro de matplotlib"

**Solução**: Reiniciar o app. O cache de 1 hora resolve

### Problema: "Porta 8504 já em uso"

**Solução**: Especificar outra porta

```powershell
python3.13.exe -m streamlit run app.py --server.port 8506
```

---

## 📚 Referências

- [USGS Earthquake Data API](https://earthquake.usgs.gov/fdsnws/event/1/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Folium Maps](https://python-visualization.github.io/folium/)
- [Plotly Python](https://plotly.com/python/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

## 📄 Licença

Este projeto está disponível para uso educacional e comercial.

---

## 👨‍💻 Autor

Desenvolvido como projeto de análise de dados com Streamlit e Python.

---

## 🤝 Contribuições

Para contribuições, sugestões ou reportar bugs:

1. Verifique a estrutura do código
2. Adicione comentários explicativos
3. Teste em ambos os dashboards
4. Documente alterações

---

## 📅 Histórico de Atualizações

| Data       | Versão | Alteração                                 |
| ---------- | ------ | ----------------------------------------- |
| 16/04/2026 | 1.0    | Versão inicial com sincronização de dados |
| -          | -      | -                                         |

---

**Última atualização**: 16 de abril de 2026
