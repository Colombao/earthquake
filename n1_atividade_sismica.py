import os
import warnings
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

# =========================
#  CONFIGURAÇÕES GERAIS
# =========================

# Pasta principal de saída
BASE_OUTPUT = Path("outputs_n1_old")
GRAFICOS_DIR = BASE_OUTPUT / "graficos"
TABELAS_DIR = BASE_OUTPUT / "tabelas"

GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)
TABELAS_DIR.mkdir(parents=True, exist_ok=True)

# Endpoint da API do USGS
BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Janela histórica escolhida para análise
# Você pode ajustar esse período se desejar trabalhar com outra janela.
API_PARAMS = {
    "format": "geojson",
    "starttime": "2018-01-01",
    "endtime": "2025-12-31",
    "minmagnitude": 2.5,
    "limit": 20000,
    "orderby": "time-asc"
}

# =========================
#  FUNÇÕES AUXILIARES
# =========================

def salvar_figura(nome_arquivo: str) -> None:
    """
    Salva a figura atual em PNG com layout ajustado.
    """
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / nome_arquivo, dpi=200, bbox_inches="tight")
    plt.show()


def imprimir_titulo(texto: str) -> None:
    """
    Apenas para deixar a execução mais organizada no terminal.
    """
    print("\n" + "=" * 100)
    print(texto)
    print("=" * 100)


def amplitude_serie(serie: pd.Series) -> float:
    """
    Calcula a amplitude (máximo - mínimo) de uma série numérica.
    """
    return float(serie.max() - serie.min())


# =========================
#  EXTRAÇÃO - COLETA DOS DADOS BRUTOS
# =========================
imprimir_titulo("ETAPA DE EXTRAÇÃO - COLETANDO DADOS DA API USGS")

response = requests.get(BASE_URL, params=API_PARAMS, timeout=60)
response.raise_for_status()
data = response.json()

print(f"Total de eventos retornados pela API: {len(data.get('features', []))}")

# =========================
#  TRANSFORMAÇÃO INICIAL - CONVERSÃO DO JSON PARA DATAFRAME
# =========================
imprimir_titulo("CONVERTENDO JSON PARA DATAFRAME")

registros = []

for quake in data["features"]:
    props = quake.get("properties", {})
    geometry = quake.get("geometry", {})
    coords = geometry.get("coordinates", [None, None, None])

    registros.append({
        "id": quake.get("id"),
        "time": props.get("time"),
        "place": props.get("place"),
        "magnitude": props.get("mag"),
        "mag_type": props.get("magType"),
        "felt": props.get("felt"),
        "tsunami": props.get("tsunami"),
        "sig": props.get("sig"),
        "status": props.get("status"),
        "latitude": coords[1] if len(coords) > 1 else None,
        "longitude": coords[0] if len(coords) > 0 else None,
        "depth": coords[2] if len(coords) > 2 else None
    })

df_raw = pd.DataFrame(registros)

print("Dimensão inicial do dataset bruto:", df_raw.shape)
print("\nAmostra inicial:")
print(df_raw.head())

# Armazena cópia do bruto
df_raw.to_csv(TABELAS_DIR / "01_dataset_bruto.csv", index=False)

# =========================
#  ETAPA 1 - AUDITORIA DOS DADOS
# =========================
imprimir_titulo("ETAPA 1 - AUDITORIA DOS DADOS")

# Fazemos uma cópia para auditoria, sem alterar ainda o dado bruto original
df_audit = df_raw.copy()

# ---------------------------------
#  Conversão do timestamp
# ---------------------------------
# O campo "time" vem em milissegundos desde a época Unix.
# Convertê-lo é obrigatório para qualquer análise temporal consistente.
df_audit["time"] = pd.to_datetime(df_audit["time"], unit="ms", errors="coerce")

print("Período temporal encontrado:")
print("Data inicial:", df_audit["time"].min())
print("Data final:", df_audit["time"].max())

# ---------------------------------
#  Tipos de dados
# ---------------------------------
print("\nTipos de dados:")
print(df_audit.dtypes)

# ---------------------------------
#  Valores ausentes
# ---------------------------------
missing_table = df_audit.isna().sum().reset_index()
missing_table.columns = ["coluna", "qtd_missing"]
missing_table["pct_missing"] = (missing_table["qtd_missing"] / len(df_audit)) * 100
missing_table = missing_table.sort_values("qtd_missing", ascending=False)

print("\nValores ausentes por coluna:")
print(missing_table)

missing_table.to_csv(TABELAS_DIR / "02_missing_values.csv", index=False)

plt.figure(figsize=(12, 5))
plt.bar(missing_table["coluna"], missing_table["qtd_missing"])
plt.title("Quantidade de valores ausentes por variável")
plt.xlabel("Variável")
plt.ylabel("Quantidade de valores ausentes")
plt.xticks(rotation=45, ha="right")
salvar_figura("01_missing_values.png")

# ---------------------------------
#  Duplicidades
# ---------------------------------
duplicated_rows = int(df_audit.duplicated().sum())
duplicated_ids = int(df_audit["id"].duplicated().sum())
duplicated_times = int(df_audit["time"].duplicated().sum())

duplicidade_resumo = pd.DataFrame({
    "tipo": ["linhas_duplicadas", "ids_duplicados", "timestamps_duplicados"],
    "quantidade": [duplicated_rows, duplicated_ids, duplicated_times]
})

print("\nResumo de duplicidades:")
print(duplicidade_resumo)

duplicidade_resumo.to_csv(TABELAS_DIR / "03_duplicidades.csv", index=False)

# ---------------------------------
#  Frequência irregular e gaps temporais
# ---------------------------------
# Como eventos sísmicos são naturalmente irregulares, vamos medir os intervalos
# entre eventos consecutivos para documentar esse comportamento.
df_audit = df_audit.sort_values("time").reset_index(drop=True)
df_audit["time_diff_hours"] = df_audit["time"].diff().dt.total_seconds() / 3600

gaps_stats = df_audit["time_diff_hours"].describe().to_frame(name="time_diff_hours")
gaps_stats.to_csv(TABELAS_DIR / "04_estatisticas_intervalos_temporais.csv")

print("\nEstatísticas dos intervalos temporais entre eventos consecutivos (horas):")
print(gaps_stats)

plt.figure(figsize=(12, 5))
plt.hist(df_audit["time_diff_hours"].dropna(), bins=50)
plt.title("Distribuição dos intervalos de tempo entre eventos consecutivos")
plt.xlabel("Horas entre eventos")
plt.ylabel("Frequência")
salvar_figura("02_intervalos_temporais.png")

# Vamos considerar como "gap longo" os intervalos acima do percentil 99
limiar_gap_longo = df_audit["time_diff_hours"].quantile(0.99)
gaps_longos = df_audit[df_audit["time_diff_hours"] > limiar_gap_longo][
    ["time", "time_diff_hours", "magnitude", "place"]
].copy()

gaps_longos.to_csv(TABELAS_DIR / "05_gaps_longos.csv", index=False)

# ---------------------------------
#  Outliers numéricos
# ---------------------------------
# Usaremos IQR e também z-score em etapas posteriores para evidenciar anomalias.
variaveis_numericas = ["magnitude", "depth", "felt", "sig", "latitude", "longitude"]

outliers_lista = []

for col in variaveis_numericas:
    serie = df_audit[col].dropna()

    if len(serie) == 0:
        continue

    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    qtd_outliers = int(((serie < limite_inferior) | (serie > limite_superior)).sum())

    outliers_lista.append({
        "variavel": col,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "limite_inferior": limite_inferior,
        "limite_superior": limite_superior,
        "qtd_outliers": qtd_outliers
    })

    plt.figure(figsize=(10, 3.5))
    plt.boxplot(serie, vert=False)
    plt.title(f"Boxplot da variável {col}")
    plt.xlabel(col)
    salvar_figura(f"boxplot_{col}.png")

outliers_table = pd.DataFrame(outliers_lista).sort_values("qtd_outliers", ascending=False)
outliers_table.to_csv(TABELAS_DIR / "06_outliers_iqr.csv", index=False)

print("\nResumo de outliers pelo método IQR:")
print(outliers_table)

# ---------------------------------
# Inconsistências lógicas
# ---------------------------------
# Aqui documentamos problemas que, mesmo não sendo missing,
# podem comprometer a análise.
inconsistencias = {
    "magnitude_negativa": int((df_audit["magnitude"] < 0).sum()),
    "profundidade_negativa": int((df_audit["depth"] < 0).sum()),
    "latitude_fora_intervalo": int(((df_audit["latitude"] < -90) | (df_audit["latitude"] > 90)).sum()),
    "longitude_fora_intervalo": int(((df_audit["longitude"] < -180) | (df_audit["longitude"] > 180)).sum()),
    "timestamp_invalido": int(df_audit["time"].isna().sum())
}

inconsistencias_df = pd.DataFrame(
    [{"problema": k, "quantidade": v} for k, v in inconsistencias.items()]
)

inconsistencias_df.to_csv(TABELAS_DIR / "07_inconsistencias_logicas.csv", index=False)

print("\nInconsistências lógicas identificadas:")
print(inconsistencias_df)

# =========================
#  ETAPA 2 - TRATAMENTO E PREPARAÇÃO (ETL)
# =========================
imprimir_titulo("ETAPA 2 - TRATAMENTO E PREPARAÇÃO (ETL)")

# Criamos uma cópia para tratamento, preservando o dado bruto para rastreabilidade.
df = df_raw.copy()

# ---------------------------------
#  Conversão temporal
# ---------------------------------
df["time"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")

# Justificativa:
# Registros sem timestamp válido não podem participar de série temporal.
df = df.dropna(subset=["time"])

# ---------------------------------
#  Remoção de duplicidades
# ---------------------------------
# Primeiro removemos duplicatas completas.
df = df.drop_duplicates()

# Depois removemos ids duplicados, preservando o primeiro registro.
# Justificativa:
# O id do evento deve representar unicidade.
df = df.drop_duplicates(subset=["id"], keep="first")

# ---------------------------------
#  Tratamento de valores ausentes essenciais
# ---------------------------------
# Variáveis essenciais ao problema: magnitude, latitude, longitude e depth.
# Sem elas, o evento perde valor analítico central.
df = df.dropna(subset=["magnitude", "latitude", "longitude", "depth"])

# ---------------------------------
#  Tratamento de colunas auxiliares
# ---------------------------------
# felt: representa número de relatos humanos de percepção do evento.
# Ausência será tratada como 0, interpretando que não houve relato registrado.
df["felt"] = df["felt"].fillna(0)

# tsunami: variável binária (0/1).
# Ausência será tratada como 0, interpretando ausência de indicação de tsunami.
df["tsunami"] = df["tsunami"].fillna(0)

# sig: relevância do evento. Podemos imputar pela mediana para reduzir influência de extremos.
if df["sig"].isna().sum() > 0:
    df["sig"] = df["sig"].fillna(df["sig"].median())

# place e mag_type podem ficar textualmente ausentes sem impedir a análise temporal.
df["place"] = df["place"].fillna("Local não informado")
df["mag_type"] = df["mag_type"].fillna("não informado")
df["status"] = df["status"].fillna("não informado")

# ---------------------------------
#  Filtragem de inconsistências lógicas severas
# ---------------------------------
# Removemos apenas inconsistências claramente inválidas.
df = df[(df["latitude"] >= -90) & (df["latitude"] <= 90)]
df = df[(df["longitude"] >= -180) & (df["longitude"] <= 180)]

# Magnitudes negativas e profundidades negativas tendem a ser inconsistentes para esta análise.
df = df[df["magnitude"] >= 0]
df = df[df["depth"] >= 0]

print("Dimensão após limpeza:", df.shape)

# Salva versão tratada em nível de evento
df.to_csv(TABELAS_DIR / "08_dataset_eventos_tratado.csv", index=False)

# ---------------------------------
#  Padronização do índice temporal
# ---------------------------------
# Como os terremotos são eventos irregulares, transformamos a base de eventos
# em uma série temporal diária agregada. Isso é essencial para análises de tendência,
# sazonalidade, decomposição e comparação temporal.
df["date"] = df["time"].dt.floor("D")

daily_series = (
    df.groupby("date")
      .agg(
          quake_count=("id", "count"),
          avg_magnitude=("magnitude", "mean"),
          max_magnitude=("magnitude", "max"),
          avg_depth=("depth", "mean"),
          tsunami_count=("tsunami", "sum"),
          total_felt=("felt", "sum"),
          avg_sig=("sig", "mean")
      )
      .sort_index()
)

# Padroniza explicitamente a frequência diária
daily_series = daily_series.asfreq("D")

# ---------------------------------
#  Tratamento de gaps da série diária
# ---------------------------------
# Dias sem registros de terremoto passam a existir como lacunas após o asfreq("D").
# A decisão de tratamento depende do significado da variável:
#
# - Contagens e somas: preencher com zero faz sentido,
#   pois "não houve evento" ou "não houve registro agregado" é informativo.
#
# - Métricas contínuas (médias e máximos): podem ser interpoladas no tempo
#   para reduzir descontinuidades analíticas, desde que essa escolha seja documentada.
#
# Essa distinção é importante para não misturar ausência de evento com valor real.

for col in ["quake_count", "tsunami_count", "total_felt"]:
    daily_series[col] = daily_series[col].fillna(0)

for col in ["avg_magnitude", "max_magnitude", "avg_depth", "avg_sig"]:
    daily_series[col] = daily_series[col].interpolate(method="time")

# ---------------------------------
#  Criação de variáveis derivadas
# ---------------------------------
# A atividade pede pelo menos duas variáveis derivadas relevantes.
# Vamos criar mais de duas para enriquecer a análise.

# 1) Média móvel de 7 dias da quantidade de terremotos
daily_series["quake_count_ma7"] = daily_series["quake_count"].rolling(7, min_periods=1).mean()

# 2) Média móvel de 30 dias da quantidade de terremotos
daily_series["quake_count_ma30"] = daily_series["quake_count"].rolling(30, min_periods=1).mean()

# 3) Variação percentual diária da magnitude média
daily_series["avg_magnitude_pct_change"] = daily_series["avg_magnitude"].pct_change()

# 4) Sazonalidade temporal extraída do calendário
daily_series["month"] = daily_series.index.month
daily_series["day_of_week"] = daily_series.index.dayofweek
daily_series["is_weekend"] = daily_series["day_of_week"].isin([5, 6]).astype(int)

# ---------------------------------
#  Escalonamento / normalização
# ---------------------------------
# Nem toda EDA precisa de escalonamento, mas ele é pertinente aqui para:
# - comparar métricas em escalas muito diferentes
# - preparar base para técnicas posteriores de ML / clusterização
#
# Aplicaremos StandardScaler em algumas variáveis-chave.
cols_scale = ["quake_count", "avg_magnitude", "max_magnitude", "avg_depth", "total_felt", "avg_sig"]

scaler = StandardScaler()
scaled_values = scaler.fit_transform(daily_series[cols_scale])

scaled_df = pd.DataFrame(
    scaled_values,
    index=daily_series.index,
    columns=[f"{col}_z" for col in cols_scale]
)

daily_series = pd.concat([daily_series, scaled_df], axis=1)

# Salva série temporal final tratada
daily_series.to_csv(TABELAS_DIR / "09_serie_temporal_diaria_tratada.csv")

print("\nSérie temporal diária tratada:")
print(daily_series.head())

# =========================
#  ETAPA 3 - ANÁLISE EXPLORATÓRIA (EDA)
# =========================
imprimir_titulo("ETAPA 3 - ANÁLISE EXPLORATÓRIA (EDA)")

# ---------------------------------
#  Estatísticas descritivas
# ---------------------------------
stats_table = pd.DataFrame({
    "media": daily_series[cols_scale].mean(),
    "mediana": daily_series[cols_scale].median(),
    "desvio_padrao": daily_series[cols_scale].std(),
    "minimo": daily_series[cols_scale].min(),
    "maximo": daily_series[cols_scale].max(),
    "amplitude": [amplitude_serie(daily_series[col]) for col in cols_scale]
})

stats_table.to_csv(TABELAS_DIR / "10_estatisticas_descritivas.csv")
print("Estatísticas descritivas:")
print(stats_table)

# ---------------------------------
#  Visualização da série temporal
# ---------------------------------
plt.figure(figsize=(14, 5))
plt.plot(daily_series.index, daily_series["quake_count"], label="Quantidade diária")
plt.plot(daily_series.index, daily_series["quake_count_ma7"], label="Média móvel 7 dias")
plt.plot(daily_series.index, daily_series["quake_count_ma30"], label="Média móvel 30 dias")
plt.title("Evolução temporal da quantidade de terremotos por dia")
plt.xlabel("Data")
plt.ylabel("Quantidade de terremotos")
plt.legend()
salvar_figura("03_serie_quake_count.png")

plt.figure(figsize=(14, 5))
plt.plot(daily_series.index, daily_series["avg_magnitude"], label="Magnitude média diária")
plt.plot(daily_series.index, daily_series["max_magnitude"], label="Magnitude máxima diária")
plt.title("Magnitude média e máxima ao longo do tempo")
plt.xlabel("Data")
plt.ylabel("Magnitude")
plt.legend()
salvar_figura("04_serie_magnitudes.png")

plt.figure(figsize=(14, 5))
plt.plot(daily_series.index, daily_series["avg_depth"])
plt.title("Profundidade média diária ao longo do tempo")
plt.xlabel("Data")
plt.ylabel("Profundidade média")
salvar_figura("05_serie_profundidade.png")

# ---------------------------------
#  Padrões sazonais por mês e dia da semana
# ---------------------------------
monthly_pattern = daily_series.groupby("month")["quake_count"].mean()

plt.figure(figsize=(10, 4))
plt.bar(monthly_pattern.index.astype(str), monthly_pattern.values)
plt.title("Média diária de terremotos por mês")
plt.xlabel("Mês")
plt.ylabel("Média diária de terremotos")
salvar_figura("06_sazonalidade_mes.png")

weekday_pattern = daily_series.groupby("day_of_week")["quake_count"].mean()
weekday_labels = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]

plt.figure(figsize=(10, 4))
plt.bar(weekday_labels, weekday_pattern.values)
plt.title("Média diária de terremotos por dia da semana")
plt.xlabel("Dia da semana")
plt.ylabel("Média diária de terremotos")
salvar_figura("07_sazonalidade_dia_semana.png")

# ---------------------------------
#  Correlação entre variáveis
# ---------------------------------
corr_cols = [
    "quake_count", "avg_magnitude", "max_magnitude",
    "avg_depth", "tsunami_count", "total_felt", "avg_sig"
]

corr = daily_series[corr_cols].corr()
corr.to_csv(TABELAS_DIR / "11_correlacoes.csv")

print("\nMatriz de correlação:")
print(corr)

plt.figure(figsize=(8, 6))
plt.imshow(corr, aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Mapa de calor de correlação entre variáveis")
salvar_figura("08_correlacao.png")

# ---------------------------------
#  Identificação de anomalias
# ---------------------------------
# Aplicamos z-score na quantidade diária de terremotos para detectar dias muito acima/abaixo do padrão.
daily_series["quake_count_zscore"] = zscore(daily_series["quake_count"].fillna(0))
anomalous_days = daily_series[np.abs(daily_series["quake_count_zscore"]) > 3].copy()

anomalous_days.to_csv(TABELAS_DIR / "12_dias_anomalos.csv")

print("\nDias anômalos identificados (|zscore| > 3):")
print(anomalous_days[["quake_count", "quake_count_zscore"]].head(20))

plt.figure(figsize=(14, 5))
plt.plot(daily_series.index, daily_series["quake_count"], label="Quantidade diária")
plt.scatter(
    anomalous_days.index,
    anomalous_days["quake_count"],
    label="Anomalias",
    marker="x"
)
plt.title("Identificação de dias anômalos na atividade sísmica")
plt.xlabel("Data")
plt.ylabel("Quantidade de terremotos")
plt.legend()
salvar_figura("09_anomalias_quake_count.png")

# ---------------------------------
#  Decomposição da série temporal
# ---------------------------------
# A decomposição separa a série em:
# - tendência
# - sazonalidade
# - resíduo
#
# Como usamos frequência diária, um período de 30 dias oferece uma leitura interessante
# de variações intra-mensais.
serie_decomp = daily_series["quake_count"].copy().fillna(method="ffill")

decomp = seasonal_decompose(serie_decomp, model="additive", period=30)

fig = decomp.plot()
fig.set_size_inches(14, 10)
plt.suptitle("Decomposição da série temporal de quantidade diária de terremotos", y=1.02)
salvar_figura("10_decomposicao_serie.png")

# Salvando componentes da decomposição em tabela
decomp_df = pd.DataFrame({
    "observado": decomp.observed,
    "tendencia": decomp.trend,
    "sazonalidade": decomp.seasonal,
    "residuo": decomp.resid
})
decomp_df.to_csv(TABELAS_DIR / "13_decomposicao_serie.csv")

# ---------------------------------
#  Ranking de dias com maior atividade
# ---------------------------------
top_days = daily_series.sort_values("quake_count", ascending=False).head(15)
top_days.to_csv(TABELAS_DIR / "14_top_dias_atividade.csv")

print("\nTop 15 dias com maior atividade sísmica:")
print(top_days[["quake_count", "avg_magnitude", "max_magnitude", "avg_depth"]])

# =========================
#  RESUMO TÉCNICO FINAL
# =========================
imprimir_titulo("RESUMO FINAL DO PROCESSAMENTO")

resumo_final = {
    "registros_brutos": len(df_raw),
    "registros_tratados_eventos": len(df),
    "inicio_serie_diaria": str(daily_series.index.min().date()),
    "fim_serie_diaria": str(daily_series.index.max().date()),
    "dias_na_serie": int(len(daily_series)),
    "dias_anomalos": int(len(anomalous_days)),
    "media_diaria_eventos": float(daily_series["quake_count"].mean()),
    "mediana_diaria_eventos": float(daily_series["quake_count"].median()),
    "maximo_diario_eventos": float(daily_series["quake_count"].max())
}

resumo_final_df = pd.DataFrame([resumo_final])
resumo_final_df.to_csv(TABELAS_DIR / "15_resumo_final.csv", index=False)

print(pd.DataFrame([resumo_final]))

print("\nArquivos gerados com sucesso em:")
print(f"- Gráficos: {GRAFICOS_DIR.resolve()}")
print(f"- Tabelas : {TABELAS_DIR.resolve()}")
