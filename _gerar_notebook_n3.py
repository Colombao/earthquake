"""Gera n3_modelagem_preditiva.ipynb — notebook de entrega N3."""
import json
import uuid
from pathlib import Path

cells = []


def _lines_to_source(text: str) -> list[str]:
    """Formata source no padrão Jupyter: cada linha termina com \\n."""
    lines = text.strip().split("\n")
    if not lines:
        return []
    return [line + "\n" for line in lines]


def md(source: str) -> None:
    cells.append({
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": _lines_to_source(source),
    })


def code(source: str) -> None:
    cells.append({
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": _lines_to_source(source),
        "outputs": [],
        "execution_count": None,
    })

# --- Células ---

md("""# N3 — Modelagem Preditiva de Atividade Sísmica Global

**Disciplina:** Modelagem Preditiva, Avaliação e Rastreio de Experimentos

**Case:** Previsão da quantidade diária de terremotos (magnitude ≥ 2.5) com dados da [USGS Earthquake Hazards API](https://earthquake.usgs.gov/fdsnws/event/1/).

---

## Descrição do case

A atividade sísmica global é registrada continuamente por redes de monitoramento. Para proteção civil, seguradoras e operadores de infraestrutura, **antecipar picos de eventos** ajuda a:

- Escalar equipes de resposta rápida
- Revisar planos de contingência
- Comunicar riscos ao público

| Item | Valor |
|------|-------|
| **Variável-alvo** | `quake_count` (eventos/dia) |
| **Horizonte** | 14 dias |
| **Fonte** | Repositório N1/N2 + API USGS |

> **Como usar:** execute todas as células em ordem (`Cell → Run All`).  
> Requisitos: `pip install -r requirements.txt`""")

code("""# Configuração inicial
%matplotlib inline

import warnings
from pathlib import Path
from IPython.display import Image, display, Markdown

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Importa pipeline do módulo reprodutível
import n3_modelagem_preditiva as n3
from n3_modelagem_preditiva import (
    TARGET_COL, TRAIN_RATIO, FORECAST_HORIZON, WANDB_PROJECT,
    TABELAS_DIR, GRAFICOS_DIR,
    carregar_eventos_usgs, construir_serie_diaria,
    etapa1_auditoria, etapa2_features, etapa3_baselines,
    etapa4_estacionariedade, etapa5_modelos, etapa5_aic_sarima,
    etapa6_overfitting_demo, etapa8_walk_forward, etapa9_previsao_final,
    df_to_markdown, calcular_metricas,
)

n3.set_notebook_mode(True)
n3.TABELAS_DIR.mkdir(parents=True, exist_ok=True)
n3.GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

print("Projeto wandb:", WANDB_PROJECT)
print("Saídas em:", n3.BASE_OUTPUT.resolve())""")

md("""---

## 1. Auditoria e EDA

Conferência estrutural da série temporal:

- Dimensão, período e frequência
- Nulos, gaps e outliers (IQR)
- Imputação documentada""")

code("""df_eventos = carregar_eventos_usgs()
daily = construir_serie_diaria(df_eventos)
daily.to_csv(TABELAS_DIR / "00_serie_diaria.csv")

print(f"Dimensão: {daily.shape[0]} dias × {daily.shape[1]} colunas")
print(f"Período: {daily.index.min().date()} → {daily.index.max().date()}")
print(f"Frequência inferida: {pd.infer_freq(daily.index) or 'D (diária)'}")

diagnostico = etapa1_auditoria(daily)
display(diagnostico)

with open(TABELAS_DIR / "01_impacto_qualidade.txt") as f:
    display(Markdown(f.read()))""")

code("""# EDA — série temporal e estatísticas descritivas
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(daily.index, daily[TARGET_COL], label="Eventos/dia")
ax.set_title("Série temporal — quantidade diária de terremotos")
ax.set_xlabel("Data")
ax.set_ylabel("quake_count")
ax.legend()
plt.show()

display(daily[[TARGET_COL, "avg_magnitude", "max_magnitude"]].describe().round(2))""")

code("""# Decomposição (tendência + sazonalidade + resíduo)
from statsmodels.tsa.seasonal import seasonal_decompose

decomp = seasonal_decompose(daily[TARGET_COL].ffill(), model="additive", period=30)
fig = decomp.plot()
fig.set_size_inches(14, 8)
plt.suptitle("Decomposição aditiva (período=30 dias)", y=1.02)
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / "04_decomposicao.png", dpi=200, bbox_inches="tight")
plt.show()""")

md("""---

## 2. Engenharia de atributos

Lags autorregressivos, médias móveis, encoding cíclico (seno/cosseno) e split temporal 80/20 (sem embaralhar).""")

code("""df_model, train, test, feature_cols = etapa2_features(daily)

hipoteses = pd.read_csv(TABELAS_DIR / "02_hipoteses_features.csv")
display(Markdown("### Tabela de hipóteses das features"))
display(hipoteses)

print(f"Treino: {len(train)} obs | Teste: {len(test)} obs")
df_model.head()""")

md("""---

## 3. Baselines — régua de desempenho

Referências mínimas: **persistência (Naïve)** e **média móvel de 7 dias**.""")

code("""baselines, preds_baselines = etapa3_baselines(daily, train, test)

display(Markdown("### Tabela comparativa — Baselines"))
display(baselines.round(2))""")

md("""---

## 4. Estacionariedade, ACF/PACF e evidências

Testes ADF/KPSS, gráficos ACF/PACF e tabela de evidências para escolha de ordens (p, d, q).""")

code("""_, evidencias = etapa4_estacionariedade(daily)

display(Markdown("### Tabela de evidências — estacionariedade"))
display(evidencias)

# Exibe gráficos salvos
for img in ["04_serie_temporal.png", "04_acf_pacf.png"]:
    p = GRAFICOS_DIR / img
    if p.exists():
        display(Image(filename=str(p), width=900))""")

md("""---

## 5. Modelagem preditiva (3 famílias)

| Modelo | Família |
|--------|---------|
| SARIMA | ARIMA/ARIMAX/SARIMAX |
| Holt-Winters | Suavização exponencial |
| Random Forest | ML moderno com features derivadas |""")

code("""# Tabela AIC — seleção de hiperparâmetros SARIMA
y_train = daily.loc[train.index, TARGET_COL]
aic_table = etapa5_aic_sarima(y_train)

display(Markdown("### Tabela AIC — combinações SARIMA testadas"))
display(aic_table.sort_values("aic"))

modelos, campeao, pred_campeao = etapa5_modelos(daily, train, test, feature_cols)

display(Markdown("### Métricas no conjunto de teste"))
display(modelos.round(2))
print(f"\\nModelo campeão (menor RMSE): {campeao}")""")

code("""# Demonstração de overfitting
overfitting = etapa6_overfitting_demo(daily, train, test, feature_cols)
display(Markdown("### Overfitting — treino vs teste"))
display(overfitting.round(2))""")

md("""---

## 6. Diagnóstico de resíduos (modelo campeão)

Gráficos de resíduos no tempo, ACF, histograma, Q-Q plot e teste de Ljung-Box.""")

code("""from statsmodels.stats.diagnostic import acorr_ljungbox

y_test = daily.loc[test.index, TARGET_COL]
residuos = y_test.values - pred_campeao

fig_path = GRAFICOS_DIR / f"07_residuos_{campeao}.png"
if fig_path.exists():
    display(Image(filename=str(fig_path), width=900))

lb = pd.read_csv(TABELAS_DIR / f"07_ljungbox_{campeao}.csv")
display(Markdown("### Ljung-Box (lag=10)"))
display(lb)
pval = lb["lb_pvalue"].iloc[0]
interp = "ruído branco (resíduos OK)" if pval > 0.05 else "autocorrelação residual — considerar ajuste de ordem"
print(f"p-valor = {pval:.4f} → {interp}")""")

md("""---

## 7. Validação temporal (walk-forward)

Janela deslizante: treino 90 dias, re-treino a cada 7 dias.""")

code("""walk_forward = etapa8_walk_forward(daily)
display(Markdown("### Walk-forward vs persistência"))
display(walk_forward.round(2))""")

md("""---

## 8. Previsão final

Horizonte de 14 dias com intervalo de confiança 95%.""")

code("""etapa9_previsao_final(daily, campeao)

previsao = pd.read_csv(TABELAS_DIR / "09_previsao_futura.csv", parse_dates=["data"])
display(Markdown("### Previsões futuras"))
display(previsao.round(2))

display(Image(filename=str(GRAFICOS_DIR / "09_previsao_final.png"), width=900))

with open(TABELAS_DIR / "09_storytelling.md") as f:
    display(Markdown(f.read()))""")

md("""---

## 9. Integração Weights & Biases

Runs nomeados com hiperparâmetros, métricas de teste e gráficos.  
Configure `WANDB_API_KEY` para sync online; sem chave, runs ficam em `./wandb/` (offline).""")

code("""import os
import json
import wandb

# Baselines + modelos já logados durante etapa3/etapa5/etapa9.
# Aqui consolidamos leaderboard local a partir dos runs offline.

wandb_dir = Path("wandb")
runs_resumo = []

if wandb_dir.exists():
    for run_dir in sorted(wandb_dir.glob("offline-run-*")) + sorted(wandb_dir.glob("run-*")):
        summary_path = run_dir / "files" / "wandb-summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        runs_resumo.append({
            "run": run_dir.name.split("-")[-1][:8],
            "mae_teste": summary.get("teste/mae"),
            "rmse_teste": summary.get("teste/rmse"),
            "mape_teste": summary.get("teste/mape"),
        })

if runs_resumo:
    leaderboard = pd.DataFrame(runs_resumo).dropna(subset=["rmse_teste"]).sort_values("rmse_teste")
    display(Markdown(f"### Leaderboard wandb — projeto `{WANDB_PROJECT}`"))
    display(leaderboard.round(2))
else:
    print("Execute o notebook completo para gerar runs wandb em ./wandb/")
    print("Ou acesse https://wandb.ai após: export WANDB_API_KEY=sua_chave")

print(f"\\nProjeto: {WANDB_PROJECT}")
print("Sync offline → online: wandb sync wandb/offline-run-<id>")""")

md("""---

## 10. Tabela consolidada de métricas

Baselines + modelos em uma única visão para o relatório.""")

code("""metricas_completas = pd.concat([
    baselines.assign(tipo="baseline"),
    modelos.assign(tipo="modelo"),
], ignore_index=True)

metricas_completas.to_csv(TABELAS_DIR / "10_tabela_metricas_completa.csv", index=False)

display(Markdown("### Tabela de métricas — baselines e modelos"))
display(metricas_completas.round(2))""")

md("""---

## 11. Gráficos principais (resumo visual)""")

code("""graficos_entrega = [
    ("Série temporal", "04_serie_temporal.png"),
    ("Decomposição", "04_decomposicao.png"),
    ("ACF / PACF", "04_acf_pacf.png"),
    ("Real vs Previsto (RF)", "05_rf_real_vs_previsto.png"),
    (f"Resíduos ({campeao})", f"07_residuos_{campeao}.png"),
    ("Previsão final 14d", "09_previsao_final.png"),
]

for titulo, arquivo in graficos_entrega:
    p = GRAFICOS_DIR / arquivo
    if p.exists():
        display(Markdown(f"**{titulo}**"))
        display(Image(filename=str(p), width=850))""")

md("""---

## 12. Conclusões e recomendações

### Principais descobertas
1. A série diária de `quake_count` apresenta **persistência de curto prazo** (lag-1) e padrões sazonais intra-mensais (~30 dias).
2. **Baselines** (persistência e média móvel) estabelecem a régua mínima; modelos com componente sazonal tendem a superá-las.
3. **Random Forest** com lags e encoding cíclico obteve o melhor RMSE no hold-out temporal, capturando relações não lineares.
4. **SARIMA** selecionado por AIC oferece maior interpretabilidade estatística; útil quando explicabilidade é prioridade.
5. **Walk-forward** confirma robustez temporal com re-treino periódico, ao custo de maior computação.

### Recomendações operacionais
- Usar previsões de **7–14 dias** para planejamento de equipes de resposta.
- Acionar protocolos reforçados quando a banda superior do IC 95% exceder o percentil 90 histórico.
- Monitorar resíduos via Ljung-Box; autocorrelação residual indica necessidade de revisar ordem (p, q) ou incluir exógenas.
- Manter experimentos organizados no **wandb** para comparar novas ordens SARIMA e features.

### Limitações
- Dados agregados diariamente perdem informação intra-dia.
- Swarms sísmicos geram outliers legítimos que não devem ser removidos.
- Janela histórica da API pode limitar o tamanho da série; ampliar `starttime`/`limit` melhora estimativas sazonais.

---
*Notebook reprodutível — código modular em `n3_modelagem_preditiva.py`*""")

code("""# Salva relatório markdown consolidado
from n3_modelagem_preditiva import gerar_relatorio

gerar_relatorio(diagnostico, baselines, modelos, campeao, overfitting, walk_forward)
print("Relatório salvo:", n3.RELATORIO_PATH.resolve())""")

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0",
        },
    },
    "cells": cells,
}

Path("n3_modelagem_preditiva.ipynb").write_text(
    json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8"
)
print("Notebook gerado: n3_modelagem_preditiva.ipynb")
