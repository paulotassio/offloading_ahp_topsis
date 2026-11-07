"""
topsis.py (versao revisada)
---------------------------
Decisao de offloading via TOPSIS, usada como baseline determinista e
compensatorio contra o sistema fuzzy especialista.

Tres formulacoes da matriz de decisao sao mantidas para comparacao:

V1 - "ambiente" (formulacao original da equipe)
    Criticidade e jitter entram com o MESMO valor nas tres alternativas.
    Consequencia: esses criterios sao neutralizados na distancia e so a
    latencia decide o ranking (resultado: 100% EDGE).

V2 - "exposicao + normalizacao vetorial por quadro"
    Criticidade e jitter passam a ser custos de EXPOSICAO A REDE: a
    alternativa LOCAL ja concluiu a inferencia e nao depende de nova
    rodada de comunicacao (exposicao 0), o HIBRIDO depende parcialmente
    (exposicao ALPHA) e o EDGE depende totalmente (exposicao 1).
    Problema: a normalizacao vetorial feita dentro de cada quadro
    transforma [0, a*c, c] sempre no mesmo vetor, qualquer que seja c > 0,
    e a MAGNITUDE da criticidade/jitter deixa de importar.

V3 - "exposicao + normalizacao por referencia fixa" (recomendada)
    Mesma matriz de exposicao da V2, mas cada criterio e dividido por uma
    escala de referencia FIXA - os mesmos universos de discurso do sistema
    fuzzy especialista (latencia 0-800 ms, jitter 0-400 ms, criticidade
    0-1) - e as solucoes ideais sao absolutas (A+ = custo zero,
    A- = custo maximo da escala). Assim a magnitude de cada criterio
    passa a influenciar a decisao.
"""

import sys
import numpy as np
import pandas as pd

REMOTE_INFERENCE_MS = 25.0
ALPHA_HIBRIDO = 0.5          # fracao de exposicao a rede do HIBRIDO
REF_SCALES = np.array([1.0, 400.0, 800.0])   # criticidade, jitter, latencia

CRITERIA = ("criticidade", "jitter", "latencia")
ALTERNATIVAS = ("LOCAL", "HIBRIDO", "EDGE")

# Pesos do AHP (CR = 0,033), ordem: criticidade, jitter, latencia
WEIGHTS_AHP = np.array([0.6333, 0.2605, 0.1062])


def _latencias(df, remote_inference_ms):
    lat_local = df["total_latency_ms"].to_numpy(dtype=np.float64)
    lat_edge = df["rtt_ms"].to_numpy(dtype=np.float64) + remote_inference_ms
    lat_hibrido = (lat_local + lat_edge) / 2
    return lat_local, lat_hibrido, lat_edge


def build_matrix(df, formulacao, alpha=ALPHA_HIBRIDO,
                 remote_inference_ms=REMOTE_INFERENCE_MS):
    """Retorna a matriz (n_quadros, 3 alternativas, 3 criterios)."""
    lat_l, lat_h, lat_e = _latencias(df, remote_inference_ms)
    crit = df["criticality"].to_numpy(dtype=np.float64)
    jit = df["rtt_jitter_ms"].to_numpy(dtype=np.float64)

    if formulacao == "V1":
        exp = np.array([1.0, 1.0, 1.0])          # sem exposicao diferenciada
    else:
        exp = np.array([0.0, alpha, 1.0])        # LOCAL, HIBRIDO, EDGE

    crit_m = crit[:, None] * exp[None, :]
    jit_m = jit[:, None] * exp[None, :]
    lat_m = np.stack([lat_l, lat_h, lat_e], axis=1)
    return np.stack([crit_m, jit_m, lat_m], axis=2)


def topsis_decide(df, weights, formulacao="V3", alpha=ALPHA_HIBRIDO,
                  remote_inference_ms=REMOTE_INFERENCE_MS):
    df = df.copy()
    matriz = build_matrix(df, formulacao, alpha, remote_inference_ms)

    if formulacao in ("V1", "V2"):
        # Normalizacao vetorial classica, dentro de cada quadro
        norma = np.sqrt((matriz ** 2).sum(axis=1, keepdims=True))
        norma[norma == 0] = 1.0
        ponderada = (matriz / norma) * weights
        pis = ponderada.min(axis=1, keepdims=True)
        nis = ponderada.max(axis=1, keepdims=True)
    else:
        # V3: escala de referencia fixa e ideais absolutos
        normalizada = np.clip(matriz / REF_SCALES, 0.0, 1.0)
        ponderada = normalizada * weights
        pis = np.zeros((1, 1, 3))                 # custo zero
        nis = weights.reshape(1, 1, 3)            # custo maximo da escala

    d_pos = np.sqrt(((ponderada - pis) ** 2).sum(axis=2))
    d_neg = np.sqrt(((ponderada - nis) ** 2).sum(axis=2))
    soma = d_pos + d_neg
    soma[soma == 0] = 1.0
    cc = d_neg / soma

    escolha = cc.argmax(axis=1)
    df["decisao_topsis"] = [ALTERNATIVAS[i] for i in escolha]
    df["topsis_c_star"] = cc[np.arange(len(df)), escolha]
    return df


def metricas(df):
    n = len(df)
    alta = df[df["criticality"] >= 0.6]
    return {
        "concordancia_%": 100 * (df["decisao_topsis"] == df["decisao_fuzzy"]).mean(),
        "LOCAL_%": 100 * (df["decisao_topsis"] == "LOCAL").mean(),
        "HIBRIDO_%": 100 * (df["decisao_topsis"] == "HIBRIDO").mean(),
        "EDGE_%": 100 * (df["decisao_topsis"] == "EDGE").mean(),
        "altacrit_LOCAL_%": 100 * (alta["decisao_topsis"] == "LOCAL").mean(),
        "altacrit_EDGE_%": 100 * (alta["decisao_topsis"] == "EDGE").mean(),
        "altacrit_concord_%": 100 * (alta["decisao_topsis"] == alta["decisao_fuzzy"]).mean(),
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "resultado_dataset.csv"
    formulacao = sys.argv[2] if len(sys.argv) > 2 else "V3"
    df = pd.read_csv(path)
    df = topsis_decide(df, WEIGHTS_AHP, formulacao)
    for k, v in metricas(df).items():
        print(f"{k:<22} {v:6.1f}")
