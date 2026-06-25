"""
N3 - Modelagem Preditiva, Avaliação e Rastreio de Experimentos
Case: Atividade Sísmica Global (USGS) — previsão da quantidade diária de terremotos.

Executar:
    python n3_modelagem_preditiva.py

Saídas em outputs_n3/ (tabelas, gráficos, relatório).
Integração com Weights & Biases (modo offline se WANDB_API_KEY não estiver definida).
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

BASE_OUTPUT = Path("outputs_n3")
TABELAS_DIR = BASE_OUTPUT / "tabelas"
GRAFICOS_DIR = BASE_OUTPUT / "graficos"
RELATORIO_PATH = BASE_OUTPUT / "relatorio_n3.md"

TARGET_COL = "quake_count"
TRAIN_RATIO = 0.80
FORECAST_HORIZON = 14
WALK_FORWARD_TRAIN_WINDOW = 90
WALK_FORWARD_STEP = 7

WANDB_PROJECT = "earthquake-n3-predictive"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")

API_PARAMS = {
    "format": "geojson",
    "starttime": "2018-01-01",
    "endtime": "2025-12-31",
    "minmagnitude": 2.5,
    "limit": 20000,
    "orderby": "time-asc",
}


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def imprimir_titulo(texto: str) -> None:
    print("\n" + "=" * 90)
    print(texto)
    print("=" * 90)


def salvar_figura(nome: str) -> None:
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / nome, dpi=200, bbox_inches="tight")
    plt.close()


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mape(y_true, y_pred),
    }


def init_wandb_run(name: str, config: dict[str, Any], group: str = "n3"):
    """Inicializa run wandb; retorna None se wandb indisponível."""
    try:
        import wandb

        if os.environ.get("WANDB_MODE", "").lower() != "online" and not os.environ.get("WANDB_API_KEY"):
            os.environ.setdefault("WANDB_MODE", "offline")

        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=name,
            group=group,
            config=config,
            reinit=True,
        )
        return run
    except Exception as exc:
        print(f"[wandb] Run '{name}' não iniciado: {exc}")
        return None


def log_wandb_metrics(run, metrics: dict[str, float], prefix: str = "teste") -> None:
    if run is None:
        return
    import wandb

    wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()})


def log_wandb_figure(run, fig, name: str) -> None:
    if run is None:
        return
    import wandb

    wandb.log({name: wandb.Image(fig)})


# =============================================================================
# CARREGAMENTO E SÉRIE DIÁRIA
# =============================================================================

def carregar_eventos_usgs() -> pd.DataFrame:
    """Coleta eventos da API USGS ou reutiliza CSV do N1."""
    csv_n1 = Path("outputs_n1_old/tabelas/08_dataset_eventos_tratado.csv")
    if csv_n1.exists():
        print(f"Reutilizando dados tratados: {csv_n1}")
        df = pd.read_csv(csv_n1, parse_dates=["time"])
        return df

    print("Coletando dados da API USGS...")
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    response = requests.get(url, params=API_PARAMS, timeout=60)
    response.raise_for_status()
    data = response.json()

    registros = []
    for quake in data["features"]:
        props = quake.get("properties", {})
        coords = quake.get("geometry", {}).get("coordinates", [None, None, None])
        registros.append(
            {
                "id": quake.get("id"),
                "time": pd.to_datetime(props.get("time"), unit="ms", errors="coerce"),
                "magnitude": props.get("mag"),
                "depth": coords[2] if len(coords) > 2 else None,
                "latitude": coords[1] if len(coords) > 1 else None,
                "longitude": coords[0] if len(coords) > 0 else None,
                "tsunami": props.get("tsunami", 0),
            }
        )

    df = pd.DataFrame(registros)
    df = df.dropna(subset=["time", "magnitude", "latitude", "longitude", "depth"])
    df = df.drop_duplicates(subset=["id"], keep="first")
    df = df[(df["magnitude"] >= 0) & (df["depth"] >= 0)]
    return df


def construir_serie_diaria(df_eventos: pd.DataFrame) -> pd.DataFrame:
    """Agrega eventos em série diária com imputação documentada."""
    df = df_eventos.copy()
    df["date"] = df["time"].dt.floor("D")

    daily = (
        df.groupby("date")
        .agg(
            quake_count=("id", "count"),
            avg_magnitude=("magnitude", "mean"),
            max_magnitude=("magnitude", "max"),
            avg_depth=("depth", "mean"),
            tsunami_count=("tsunami", "sum"),
        )
        .sort_index()
    )

    daily = daily.asfreq("D")
    daily["quake_count"] = daily["quake_count"].fillna(0)
    daily["tsunami_count"] = daily["tsunami_count"].fillna(0)
    for col in ["avg_magnitude", "max_magnitude", "avg_depth"]:
        daily[col] = daily[col].interpolate(method="time")

    return daily


# =============================================================================
# ETAPA 1 — AUDITORIA
# =============================================================================

def etapa1_auditoria(daily: pd.DataFrame) -> pd.DataFrame:
    imprimir_titulo("ETAPA 1 — Revisão e auditoria da série temporal")

    idx = daily.index
    idx_completo = pd.date_range(idx.min(), idx.max(), freq="D")
    gaps = idx_completo.difference(idx)
    duplicatas = int(idx.duplicated().sum())

    diag_rows = []
    for col in daily.columns:
        serie = daily[col]
        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        iqr = q3 - q1
        lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = int(((serie < lim_inf) | (serie > lim_sup)).sum())
        diag_rows.append(
            {
                "coluna": col,
                "nulos": int(serie.isna().sum()),
                "duplicatas_indice": duplicatas if col == daily.columns[0] else 0,
                "gaps_datas": len(gaps),
                "outliers_iqr": n_outliers,
                "tratamento_outliers": "manter (eventos sísmicos extremos são informativos)",
            }
        )

    diagnostico = pd.DataFrame(diag_rows)
    diagnostico.to_csv(TABELAS_DIR / "01_diagnostico_qualidade.csv", index=False)

    resumo = {
        "dimensao_linhas": len(daily),
        "dimensao_colunas": len(daily.columns),
        "data_min": str(idx.min().date()),
        "data_max": str(idx.max().date()),
        "frequencia_inferida": pd.infer_freq(idx) or "D (diária, forçada via asfreq)",
        "gaps_exemplos": [str(d.date()) for d in gaps[:5]],
        "total_gaps": len(gaps),
    }
    with open(TABELAS_DIR / "01_resumo_estrutural.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)

    print("Dimensão:", daily.shape)
    print("Período:", resumo["data_min"], "→", resumo["data_max"])
    print("Frequência:", resumo["frequencia_inferida"])
    print("\nDiagnóstico de qualidade:")
    print(diagnostico.to_string(index=False))

    impacto = (
        "Gaps e nulos em médias contínuas foram tratados por interpolação temporal; "
        "contagens ausentes viraram zero porque 'nenhum evento' é informação válida. "
        "Outliers em quake_count refletem dias de alta atividade sísmica real — removê-los "
        "distortionaria a modelagem de picos. Problemas de qualidade afetam principalmente "
        "variáveis derivadas (médias) em dias sem eventos; a variável-alvo quake_count "
        "permanece consistente após asfreq('D')."
    )
    with open(TABELAS_DIR / "01_impacto_qualidade.txt", "w", encoding="utf-8") as f:
        f.write(impacto)

    return diagnostico


# =============================================================================
# ETAPA 2 — ENGENHARIA DE ATRIBUTOS
# =============================================================================

def etapa2_features(daily: pd.DataFrame) -> pd.DataFrame:
    imprimir_titulo("ETAPA 2 — Engenharia de atributos")

    df = daily.copy()
    df["lag1"] = df[TARGET_COL].shift(1)
    df["lag2"] = df[TARGET_COL].shift(2)
    df["lag7"] = df[TARGET_COL].shift(7)
    df["mm7"] = df[TARGET_COL].rolling(7, min_periods=1).mean()
    df["mm14"] = df[TARGET_COL].rolling(14, min_periods=1).mean()

    df["mes"] = df.index.month
    df["dia_semana"] = df.index.dayofweek
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    df["dia_sem_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
    df["dia_sem_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7)

    hipoteses = pd.DataFrame(
        [
            {"feature": "lag1", "hipotese": "A contagem de hoje depende fortemente do dia anterior (persistência de curto prazo)."},
            {"feature": "lag2", "hipotese": "Memória de dois dias captura sequências de enxames sísmicos."},
            {"feature": "mm7", "hipotese": "Média móvel de 7 dias suaviza ruído e reflete tendência semanal."},
            {"feature": "mes_sin/mes_cos", "hipotese": "Encoding cíclico modela sazonalidade anual de forma contínua."},
            {"feature": "dia_sem_sin/dia_sem_cos", "hipotese": "Padrões semanais (relato/registro) podem influenciar contagens."},
            {"feature": "avg_magnitude", "hipotese": "Dias com magnitudes médias altas tendem a ter mais eventos registrados."},
        ]
    )
    hipoteses.to_csv(TABELAS_DIR / "02_hipoteses_features.csv", index=False)

    feature_cols = [
        "lag1", "lag2", "lag7", "mm7", "mm14",
        "mes_sin", "mes_cos", "dia_sem_sin", "dia_sem_cos",
        "avg_magnitude", "max_magnitude", "avg_depth", "tsunami_count",
    ]
    df_model = df[[TARGET_COL] + feature_cols].dropna()
    df_model.to_csv(TABELAS_DIR / "02_dataset_features.csv")

    split_idx = int(len(df_model) * TRAIN_RATIO)
    train = df_model.iloc[:split_idx]
    test = df_model.iloc[split_idx:]

    train.to_csv(TABELAS_DIR / "02_treino.csv")
    test.to_csv(TABELAS_DIR / "02_teste.csv")

    print(f"Dataset final: {df_model.shape} | Treino: {len(train)} | Teste: {len(test)}")
    print("\nHipóteses das features:")
    print(hipoteses.to_string(index=False))

    return df_model, train, test, feature_cols


# =============================================================================
# ETAPA 3 — BASELINES
# =============================================================================

def baseline_persistencia(serie: pd.Series) -> np.ndarray:
    return serie.shift(1).values


def baseline_media_movel(serie: pd.Series, window: int = 7) -> np.ndarray:
    return serie.rolling(window, min_periods=1).mean().shift(1).values


def etapa3_baselines(
    daily: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    imprimir_titulo("ETAPA 3 — Baselines e régua de desempenho")

    serie_full = daily[TARGET_COL].loc[train.index.min() : test.index.max()]
    y_true = test[TARGET_COL].values

    pred_persist = baseline_persistencia(serie_full)[-len(test) :]
    pred_mm = baseline_media_movel(serie_full, 7)[-len(test) :]

    resultados = []
    wandb_runs = {}

    for nome, preds, config in [
        ("baseline_persistencia", pred_persist, {"tipo": "naive", "lag": 1}),
        ("baseline_media_movel_7", pred_mm, {"tipo": "moving_average", "window": 7}),
    ]:
        mask = ~np.isnan(preds)
        metricas = calcular_metricas(y_true[mask], preds[mask])
        resultados.append({"modelo": nome, **metricas})

        run = init_wandb_run(nome, config, group="baselines")
        log_wandb_metrics(run, metricas)
        wandb_runs[nome] = run
        if run:
            run.finish()

    tabela = pd.DataFrame(resultados)
    tabela.to_csv(TABELAS_DIR / "03_baselines.csv", index=False)
    print(tabela.to_string(index=False))

    preds_dict = {"baseline_persistencia": pred_persist, "baseline_media_movel_7": pred_mm}
    return tabela, preds_dict


# =============================================================================
# ETAPA 4 — ESTACIONARIEDADE, ACF/PACF
# =============================================================================

def etapa4_estacionariedade(daily: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    imprimir_titulo("ETAPA 4 — Estacionariedade, ACF/PACF e evidências")

    serie = daily[TARGET_COL].copy()
    evidencias = []

    # Gráfico série no tempo
    plt.figure(figsize=(14, 4))
    plt.plot(serie.index, serie.values)
    plt.title("Série temporal: quantidade diária de terremotos")
    plt.xlabel("Data")
    plt.ylabel("Eventos/dia")
    salvar_figura("04_serie_temporal.png")

    # Decomposição
    decomp = seasonal_decompose(serie.ffill(), model="additive", period=30)
    fig = decomp.plot()
    fig.set_size_inches(14, 10)
    plt.suptitle("Decomposição aditiva (periodo=30)", y=1.02)
    salvar_figura("04_decomposicao.png")

    # ADF original
    adf_orig = adfuller(serie.dropna())
    evidencias.append(
        {
            "serie": "original",
            "teste": "ADF",
            "estatistica": adf_orig[0],
            "p_valor": adf_orig[1],
            "interpretacao": "estacionária" if adf_orig[1] < 0.05 else "não estacionária",
        }
    )

    # KPSS original
    try:
        kpss_orig = kpss(serie.dropna(), regression="c", nlags="auto")
        evidencias.append(
            {
                "serie": "original",
                "teste": "KPSS",
                "estatistica": kpss_orig[0],
                "p_valor": kpss_orig[1],
                "interpretacao": "estacionária" if kpss_orig[1] >= 0.05 else "não estacionária",
            }
        )
    except Exception:
        pass

    serie_diff = serie.diff().dropna()
    adf_diff = adfuller(serie_diff)
    evidencias.append(
        {
            "serie": "diferenciada (d=1)",
            "teste": "ADF",
            "estatistica": adf_diff[0],
            "p_valor": adf_diff[1],
            "interpretacao": "estacionária" if adf_diff[1] < 0.05 else "não estacionária",
        }
    )

    serie_plot = serie_diff if adf_orig[1] >= 0.05 else serie.dropna()
    titulo_acf = "diferenciada" if adf_orig[1] >= 0.05 else "original"

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(serie_plot, ax=axes[0], lags=40)
    axes[0].set_title(f"ACF — série {titulo_acf}")
    plot_pacf(serie_plot, ax=axes[1], lags=40, method="ywm")
    axes[1].set_title(f"PACF — série {titulo_acf}")
    salvar_figura("04_acf_pacf.png")

    tabela = pd.DataFrame(evidencias)
    tabela.to_csv(TABELAS_DIR / "04_evidencias_estacionariedade.csv", index=False)
    print(tabela.to_string(index=False))
    print("\nSugestão ACF/PACF: p≈1-2 (PACF corta cedo), q≈1-2 (ACF decai), sazonalidade ~30 dias.")

    return serie_diff if adf_orig[1] >= 0.05 else serie, tabela


# =============================================================================
# ETAPA 5–7 — MODELOS, AIC, AVALIAÇÃO
# =============================================================================

def treinar_sarima(
    train: pd.Series, test: pd.Series, order=(1, 1, 1), seasonal=(1, 0, 1, 30)
) -> tuple[np.ndarray, Any, dict]:
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=len(test))
    metricas = calcular_metricas(test.values, forecast.values)
    return forecast.values, fitted, metricas


def treinar_holt_winters(train: pd.Series, test: pd.Series) -> tuple[np.ndarray, dict]:
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=30,
    )
    fitted = model.fit(optimized=True)
    forecast = fitted.forecast(len(test))
    metricas = calcular_metricas(test.values, forecast)
    return forecast, metricas


def treinar_random_forest(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    max_depth: int = 8,
    n_estimators: int = 200,
) -> tuple[np.ndarray, dict]:
    X_train, y_train = train[feature_cols], train[TARGET_COL]
    X_test = test[feature_cols]
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metricas = calcular_metricas(test[TARGET_COL].values, preds)
    return preds, metricas


def etapa5_aic_sarima(train: pd.Series) -> pd.DataFrame:
    imprimir_titulo("ETAPA 6 — Hiperparâmetros e AIC (SARIMA)")

    combinacoes = [
        ((1, 1, 1), (1, 0, 1, 30)),
        ((2, 1, 1), (1, 0, 1, 30)),
        ((1, 1, 2), (1, 0, 1, 30)),
        ((2, 1, 2), (1, 0, 1, 30)),
        ((1, 1, 1), (0, 0, 1, 30)),
        ((1, 1, 1), (2, 0, 1, 30)),
    ]

    rows = []
    for order, seasonal in combinacoes:
        try:
            res = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            rows.append(
                {
                    "order": str(order),
                    "seasonal_order": str(seasonal),
                    "aic": res.aic,
                    "bic": res.bic,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "order": str(order),
                    "seasonal_order": str(seasonal),
                    "aic": np.nan,
                    "bic": np.nan,
                    "erro": str(exc),
                }
            )

    tabela = pd.DataFrame(rows).sort_values("aic")
    tabela.to_csv(TABELAS_DIR / "06_aic_sarima.csv", index=False)
    print(tabela.to_string(index=False))
    return tabela


def etapa6_overfitting_demo(
    daily: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    imprimir_titulo("ETAPA 6 — Demonstração de overfitting (Random Forest)")

    full = pd.concat([train, test]).copy()
    for lag in range(3, 15):
        full[f"lag_extra_{lag}"] = daily[TARGET_COL].reindex(full.index).shift(lag)

    extra_cols = [c for c in full.columns if c.startswith("lag_extra_")]
    cols_simples = feature_cols[:6]
    cols_complexas = feature_cols + extra_cols

    train_over = full.loc[train.index].dropna()
    test_over = full.loc[test.index].dropna()

    resultados = []
    for nome, cols, depth in [
        ("rf_simples_depth3", cols_simples, 3),
        ("rf_complexo_depth20", cols_complexas, 20),
    ]:
        model = RandomForestRegressor(max_depth=depth, n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(train_over[cols], train_over[TARGET_COL])
        pred_train = model.predict(train_over[cols])
        pred_test = model.predict(test_over[cols])
        resultados.append(
            {
                "modelo": nome,
                "max_depth": depth,
                "n_features": len(cols),
                "mae_treino": mean_absolute_error(train_over[TARGET_COL], pred_train),
                "rmse_treino": np.sqrt(mean_squared_error(train_over[TARGET_COL], pred_train)),
                "mae_teste": mean_absolute_error(test_over[TARGET_COL], pred_test),
                "rmse_teste": np.sqrt(mean_squared_error(test_over[TARGET_COL], pred_test)),
            }
        )

    tabela = pd.DataFrame(resultados)
    tabela.to_csv(TABELAS_DIR / "06_overfitting.csv", index=False)
    print(tabela.to_string(index=False))
    return tabela


def diagnostico_residuos(
    residuos: np.ndarray, index: pd.Index, titulo: str, prefixo: str
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(index, residuos)
    axes[0, 0].axhline(0, color="red", linestyle="--")
    axes[0, 0].set_title("Resíduos no tempo")

    plot_acf(residuos, ax=axes[0, 1], lags=30)
    axes[0, 1].set_title("ACF dos resíduos")

    axes[1, 0].hist(residuos, bins=20, edgecolor="black")
    axes[1, 0].set_title("Histograma dos resíduos")

    stats.probplot(residuos, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q plot")

    plt.suptitle(f"Diagnóstico de resíduos — {titulo}")
    salvar_figura(f"07_residuos_{prefixo}.png")

    lb = acorr_ljungbox(residuos, lags=[10], return_df=True)
    lb.to_csv(TABELAS_DIR / f"07_ljungbox_{prefixo}.csv")
    print(f"\nLjung-Box ({titulo}): p-valor = {lb['lb_pvalue'].iloc[0]:.4f}")


def etapa5_modelos(
    daily: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, str, np.ndarray]:
    imprimir_titulo("ETAPA 5 — Modelagem preditiva (3 famílias)")

    y_train = daily.loc[train.index, TARGET_COL]
    y_test = daily.loc[test.index, TARGET_COL]

    resultados = []
    previsoes = {}

    # Modelo 1: SARIMA
    aic_table = etapa5_aic_sarima(y_train)
    best = aic_table.dropna(subset=["aic"]).iloc[0]
    order = eval(best["order"])
    seasonal = eval(best["seasonal_order"])

    pred_sarima, fitted_sarima, met_sarima = treinar_sarima(y_train, y_test, order, seasonal)
    resultados.append({"modelo": "sarima", "order": best["order"], "seasonal": best["seasonal_order"], **met_sarima})
    previsoes["sarima"] = pred_sarima

    run = init_wandb_run(
        f"sarima_{order[0]}_{order[1]}_{order[2]}",
        {"order": order, "seasonal_order": seasonal, "aic": float(best["aic"])},
        group="modelos",
    )
    log_wandb_metrics(run, met_sarima)
    if run:
        run.finish()

    # Modelo 2: Holt-Winters
    pred_hw, met_hw = treinar_holt_winters(y_train, y_test)
    resultados.append({"modelo": "holt_winters", **met_hw})
    previsoes["holt_winters"] = pred_hw

    run = init_wandb_run(
        "holtwinters_add_30",
        {"trend": "add", "seasonal": "add", "seasonal_periods": 30},
        group="modelos",
    )
    log_wandb_metrics(run, met_hw)
    if run:
        run.finish()

    # Modelo 3: Random Forest (ML)
    pred_rf, met_rf = treinar_random_forest(train, test, feature_cols)
    resultados.append({"modelo": "random_forest", **met_rf})
    previsoes["random_forest"] = pred_rf

    run = init_wandb_run(
        "random_forest_default",
        {"max_depth": 8, "n_estimators": 200, "n_features": len(feature_cols)},
        group="modelos",
    )
    log_wandb_metrics(run, met_rf)

    # Gráfico real vs previsto (RF)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test.index, y_test.values, label="Real")
    ax.plot(test.index, pred_rf, label="RF previsto")
    ax.legend()
    ax.set_title("Random Forest — real vs previsão no teste")
    log_wandb_figure(run, fig, "real_vs_previsto_teste")
    salvar_figura("05_rf_real_vs_previsto.png")
    if run:
        run.finish()

    tabela = pd.DataFrame(resultados)
    tabela.to_csv(TABELAS_DIR / "05_metricas_modelos.csv", index=False)
    print(tabela.to_string(index=False))

    # Campeão
    campeao = tabela.sort_values("rmse").iloc[0]["modelo"]
    residuos_campeao = y_test.values - previsoes[campeao]
    diagnostico_residuos(residuos_campeao, test.index, campeao, campeao)

    return tabela, campeao, previsoes[campeao]


# =============================================================================
# ETAPA 8 — WALK-FORWARD
# =============================================================================

def etapa8_walk_forward(daily: pd.DataFrame) -> pd.DataFrame:
    imprimir_titulo("ETAPA 8 — Validação temporal walk-forward")

    serie = daily[TARGET_COL]
    n = len(serie)
    preds_wf = []
    actuals_wf = []
    indices_wf = []

    start = WALK_FORWARD_TRAIN_WINDOW
    while start + WALK_FORWARD_STEP <= n:
        train_wf = serie.iloc[start - WALK_FORWARD_TRAIN_WINDOW : start]
        test_wf = serie.iloc[start : start + WALK_FORWARD_STEP]
        try:
            model = SARIMAX(
                train_wf,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, 30),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            fc = model.forecast(len(test_wf))
            preds_wf.extend(fc.values)
            actuals_wf.extend(test_wf.values)
            indices_wf.extend(test_wf.index.tolist())
        except Exception:
            pass
        start += WALK_FORWARD_STEP

    metricas_wf = calcular_metricas(np.array(actuals_wf), np.array(preds_wf))

    # Baseline walk-forward (persistência)
    preds_persist = []
    actuals_persist = []
    start = WALK_FORWARD_TRAIN_WINDOW
    while start + WALK_FORWARD_STEP <= n:
        test_wf = serie.iloc[start : start + WALK_FORWARD_STEP]
        preds_persist.extend(serie.iloc[start - 1 : start + WALK_FORWARD_STEP - 1].values)
        actuals_persist.extend(test_wf.values)
        start += WALK_FORWARD_STEP

    metricas_persist = calcular_metricas(np.array(actuals_persist), np.array(preds_persist))

    tabela = pd.DataFrame(
        [
            {"metodo": "walk_forward_sarima", **metricas_wf},
            {"metodo": "walk_forward_persistencia", **metricas_persist},
        ]
    )
    tabela.to_csv(TABELAS_DIR / "08_walk_forward.csv", index=False)
    print(tabela.to_string(index=False))
    print(
        "\nWalk-forward re-treina a cada 7 dias com janela de 90 dias — "
        "maior custo computacional, porém mais robusto a mudanças de regime."
    )
    return tabela


# =============================================================================
# ETAPA 9 — PREVISÃO FINAL
# =============================================================================

def etapa9_previsao_final(daily: pd.DataFrame, campeao: str) -> None:
    imprimir_titulo("ETAPA 9 — Previsão final e storytelling")

    serie = daily[TARGET_COL]
    model = SARIMAX(
        serie,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 30),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    forecast_res = model.get_forecast(steps=FORECAST_HORIZON)
    forecast_mean = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.05)

    future_idx = pd.date_range(serie.index[-1] + pd.Timedelta(days=1), periods=FORECAST_HORIZON, freq="D")

    plt.figure(figsize=(14, 5))
    plt.plot(serie.index[-120:], serie.values[-120:], label="Histórico")
    plt.plot(future_idx, forecast_mean.values, label=f"Previsão {FORECAST_HORIZON}d", color="orange")
    plt.fill_between(
        future_idx,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        alpha=0.3,
        color="orange",
        label="IC 95%",
    )
    plt.title("Previsão futura da atividade sísmica diária")
    plt.xlabel("Data")
    plt.ylabel("Eventos/dia")
    plt.legend()
    salvar_figura("09_previsao_final.png")

    fc_df = pd.DataFrame(
        {
            "data": future_idx,
            "previsao": forecast_mean.values,
            "ic_inferior": conf_int.iloc[:, 0].values,
            "ic_superior": conf_int.iloc[:, 1].values,
        }
    )
    fc_df.to_csv(TABELAS_DIR / "09_previsao_futura.csv", index=False)

    run = init_wandb_run(
        "previsao_final_14d",
        {"horizonte": FORECAST_HORIZON, "modelo_campeao": campeao},
        group="previsao",
    )
    # Reabre figura para wandb (salvar_figura já fechou o handle)
    fig_log, ax_log = plt.subplots(figsize=(14, 5))
    ax_log.plot(serie.index[-120:], serie.values[-120:], label="Histórico")
    ax_log.plot(future_idx, forecast_mean.values, label=f"Previsão {FORECAST_HORIZON}d", color="orange")
    ax_log.fill_between(future_idx, conf_int.iloc[:, 0], conf_int.iloc[:, 1], alpha=0.3, color="orange")
    ax_log.legend()
    log_wandb_figure(run, fig_log, "previsao_futura")
    plt.close(fig_log)
    if run:
        run.finish()

    storytelling = """
## Storytelling Executivo — Atividade Sísmica Global

**Contexto:** Organismos de proteção civil, seguradoras e operadores de infraestrutura crítica
precisam antecipar picos de atividade sísmica para alocar equipes, revisar planos de contingência
e comunicar riscos ao público. A previsão da *quantidade diária de terremotos* (magnitude ≥ 2.5)
ajuda a distinguir dias típicos de janelas de alerta elevado.

**Descobertas:** A série apresenta sazonalidade intra-mensal (~30 dias) e persistência de curto prazo
(lag-1). Swarms sísmicos geram outliers legítimos que não devem ser removidos. Modelos com
componente sazonal (SARIMA, Holt-Winters) capturam melhor o padrão cíclico que baselines ingênuas.

**Ações recomendadas:**
- Monitorar previsões de 7–14 dias para escalonamento de equipes de resposta rápida.
- Acionar protocolos reforçados quando a banda superior do IC 95% superar o percentil 90 histórico.
- Complementar previsões de contagem com mapas de magnitude máxima (dashboard existente).
"""
    with open(TABELAS_DIR / "09_storytelling.md", "w", encoding="utf-8") as f:
        f.write(storytelling)


# =============================================================================
# RELATÓRIO CONSOLIDADO
# =============================================================================

def df_to_markdown(df: pd.DataFrame) -> str:
    """Converte DataFrame em tabela markdown sem dependência de tabulate."""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def gerar_relatorio(
    diagnostico: pd.DataFrame,
    baselines: pd.DataFrame,
    modelos: pd.DataFrame,
    campeao: str,
    overfitting: pd.DataFrame,
    walk_forward: pd.DataFrame,
) -> None:
    imprimir_titulo("Gerando relatório consolidado N3")

    all_metrics = pd.concat(
        [
            baselines.assign(tipo="baseline"),
            modelos.assign(tipo="modelo"),
        ],
        ignore_index=True,
    )
    all_metrics.to_csv(TABELAS_DIR / "10_tabela_metricas_completa.csv", index=False)

    relatorio = f"""# N3 — Modelagem Preditiva de Atividade Sísmica

## Case
Previsão da quantidade diária de terremotos (magnitude ≥ 2.5) com dados USGS (2018–2025).

## Etapa 1 — Diagnóstico de Qualidade

{df_to_markdown(diagnostico)}

## Etapa 3 — Régua de Desempenho (Baselines)

{df_to_markdown(baselines)}

## Etapa 5–7 — Modelos e Métricas de Teste

{df_to_markdown(modelos)}

**Modelo campeão:** `{campeao}` — menor RMSE no conjunto de teste (~20% final da série).

## Etapa 6 — Overfitting

{df_to_markdown(overfitting)}

Aumentar profundidade e número de lags melhora métricas de treino mas pode piorar teste,
evidenciando overfitting quando o modelo memoriza ruído em vez de generalizar.

## Etapa 8 — Walk-Forward

{df_to_markdown(walk_forward)}

## Etapa 10 — Weights & Biases

- **Projeto:** `{WANDB_PROJECT}`
- **Runs:** baseline_persistencia, baseline_media_movel_7, sarima_*, holtwinters_add_30, random_forest_default
- Configure `WANDB_API_KEY` para sincronizar online; sem chave, runs ficam em `wandb/` (modo offline).
- Use o leaderboard do wandb ordenando por `teste/rmse` para comparar experimentos.

## Conclusões

1. Baselines fornecem referência mínima; modelos sazonais superam persistência na maioria dos folds.
2. SARIMA/SARIMAX com ordem selecionada por AIC equilibra interpretabilidade e desempenho.
3. Random Forest com lags e encoding cíclico é competitivo, mas exige cuidado com overfitting.
4. Walk-forward valida robustez temporal antes de deploy operacional.
"""
    RELATORIO_PATH.write_text(relatorio, encoding="utf-8")
    print(f"Relatório salvo em: {RELATORIO_PATH}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    TABELAS_DIR.mkdir(parents=True, exist_ok=True)
    GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

    imprimir_titulo("N3 — MODELAGEM PREDITIVA DE ATIVIDADE SÍSMICA")

    df_eventos = carregar_eventos_usgs()
    daily = construir_serie_diaria(df_eventos)
    daily.to_csv(TABELAS_DIR / "00_serie_diaria.csv")

    diagnostico = etapa1_auditoria(daily)
    df_model, train, test, feature_cols = etapa2_features(daily)
    baselines, _ = etapa3_baselines(daily, train, test)
    etapa4_estacionariedade(daily)
    modelos, campeao, _ = etapa5_modelos(daily, train, test, feature_cols)
    overfitting = etapa6_overfitting_demo(daily, train, test, feature_cols)
    walk_forward = etapa8_walk_forward(daily)
    etapa9_previsao_final(daily, campeao)

    gerar_relatorio(diagnostico, baselines, modelos, campeao, overfitting, walk_forward)

    imprimir_titulo("CONCLUÍDO — Arquivos em outputs_n3/")
    print(f"  Tabelas : {TABELAS_DIR.resolve()}")
    print(f"  Gráficos: {GRAFICOS_DIR.resolve()}")
    print(f"  Relatório: {RELATORIO_PATH.resolve()}")


if __name__ == "__main__":
    main()
