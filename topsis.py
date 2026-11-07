"""
topsis.py (versao revisada, com a formulacao V4)
------------------------------------------------
Decisao de offloading via TOPSIS, usada como baseline determinista e
compensatorio contra o sistema fuzzy especialista.

Quatro formulacoes da matriz de decisao sao mantidas para comparacao.
V1, V2 e V3 sao exatamente as do artigo; a V4 e a "Saida A" da revisao.

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

V3 - "exposicao + normalizacao por referencia fixa"
    Mesma matriz de exposicao da V2, mas cada criterio e dividido por uma
    escala de referencia FIXA - os mesmos universos de discurso do sistema
    fuzzy especialista (latencia 0-800 ms, jitter 0-400 ms, criticidade
    0-1) - e as solucoes ideais sao absolutas (A+ = custo zero,
    A- = custo maximo da escala). Assim a magnitude de cada criterio
    passa a influenciar a decisao.
    Obs.: na V2 e na V3 a latencia do HIBRIDO e a media fixa (1/2) das
    outras duas, mesmo quando alpha varia. Mantido assim para reproduzir
    o artigo; a V4 corrige o modelo do HIBRIDO.

V4 - "HIBRIDO em paralelo + energia" (Saida A, recomendada)
    Mesmo esquema da V3 (exposicao, escalas fixas, ideais absolutos), com
    duas mudancas:
    (1) O HIBRIDO segue a definicao do especialista (fuzzy_system.py):
        "mantem local, mas pode consultar a borda em paralelo". O alerta
        sai do primeiro resultado, entao a latencia do HIBRIDO e
        min(L_local, L_edge). A exposicao a rede e 0, porque o resultado
        local garante o alerta mesmo que o enlace falhe.
        Opcao hibrido_latencia="local": so o resultado local dispara o
        alerta (a borda apenas refina); nesse caso o HIBRIDO e dominado
        pelo LOCAL e nunca e escolhido.
    (2) Entra um quarto criterio de custo, a energia por quadro:
        LOCAL = E_inf, EDGE = E_tx, HIBRIDO = E_inf + E_tx.
        Com a escala de referencia E_inf + E_tx, os valores normalizados
        dependem apenas de rho = E_tx / E_inf:
            (LOCAL, HIBRIDO, EDGE) = (1/(1+rho), 1, rho/(1+rho)).
        PLACEHOLDER: rho = 1 ate a medicao de energia no smartphone.
    Sem o criterio de energia, o HIBRIDO dominaria LOCAL e EDGE nos tres
    criterios restantes e seria escolhido em todos os quadros.

Parametros nao medidos (PLACEHOLDER) - precisam de medicao antes de
qualquer numero da V4 entrar num artigo como resultado:
    ENERGY_RATIO       rho = E_tx / E_inf
    RTT_EDGE_EXTRA_MS  enlace smartphone -> borda (0 reproduz o artigo)
    REMOTE_INFERENCE_MS inferencia no servidor de borda
    AHP_JULGAMENTOS_V4 julgamentos 4 a 6 (criterio energia) sao provisorios
"""

import sys
import numpy as np
import pandas as pd

REMOTE_INFERENCE_MS = 25.0
RTT_EDGE_EXTRA_MS = 0.0      # PLACEHOLDER: enlace smartphone -> borda
ALPHA_HIBRIDO = 0.5          # fracao de exposicao a rede do HIBRIDO (V2/V3)
REF_SCALES = np.array([1.0, 400.0, 800.0])          # criticidade, jitter, latencia
REF_SCALES_V4 = np.array([1.0, 400.0, 800.0, 1.0])  # + energia (ja normalizada)
ENERGY_RATIO = 1.0           # PLACEHOLDER: rho = E_tx / E_inf
HIBRIDO_LATENCIA = "min"     # "min" (alerta do 1o resultado) ou "local"

CRITERIA = ("criticidade", "jitter", "latencia")
CRITERIA_V4 = ("criticidade", "jitter", "latencia", "energia")
ALTERNATIVAS = ("LOCAL", "HIBRIDO", "EDGE")

# Pesos do AHP (CR = 0,033), ordem: criticidade, jitter, latencia (artigo)
WEIGHTS_AHP = np.array([0.6333, 0.2605, 0.1062])

# --------------------------------------------------------------------
# AHP 4x4 da V4. Valor a_ij = quanto o criterio i e mais importante que j
# para a seguranca do usuario (escala de Saaty, 1 a 9).
# Julgamentos 1-3: os mesmos do artigo.
# Julgamentos 4-6 (energia): PROVISORIOS, a confirmar pelos autores ou
# por um painel. A energia afeta a autonomia do aparelho, nao o risco
# imediato, por isso e o criterio menos importante.
AHP_JULGAMENTOS_V4 = {
    ("criticidade", "jitter"): 3,
    ("criticidade", "latencia"): 5,
    ("jitter", "latencia"): 3,
    ("criticidade", "energia"): 7,
    ("jitter", "energia"): 3,
    ("latencia", "energia"): 2,
}

# Indice aleatorio de Saaty
RI_SAATY = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
            7: 1.32, 8: 1.41, 9: 1.45}


def matriz_ahp(julgamentos, criterios=CRITERIA_V4):
    """Monta a matriz reciproca n x n a partir dos julgamentos a_ij (i > j)."""
    idx = {c: k for k, c in enumerate(criterios)}
    n = len(criterios)
    A = np.ones((n, n))
    for (ci, cj), valor in julgamentos.items():
        i, j = idx[ci], idx[cj]
        A[i, j] = float(valor)
        A[j, i] = 1.0 / float(valor)
    return A


def pesos_ahp(A):
    """Pesos pelo autovetor principal, lambda_max, CI e CR (Saaty)."""
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    autoval, autovet = np.linalg.eig(A)
    k = int(np.argmax(autoval.real))
    w = np.abs(autovet[:, k].real)
    w = w / w.sum()
    lam = float(autoval[k].real)
    ci = (lam - n) / (n - 1) if n > 2 else 0.0
    ri = RI_SAATY.get(n, 1.45)
    cr = ci / ri if ri > 0 else 0.0
    return w, lam, ci, cr


WEIGHTS_AHP_V4 = pesos_ahp(matriz_ahp(AHP_JULGAMENTOS_V4))[0]


def _latencias(df, remote_inference_ms, rtt_extra_ms=0.0):
    lat_local = df["total_latency_ms"].to_numpy(dtype=np.float64)
    lat_edge = (df["rtt_ms"].to_numpy(dtype=np.float64)
                + rtt_extra_ms + remote_inference_ms)
    lat_hibrido = (lat_local + lat_edge) / 2
    return lat_local, lat_hibrido, lat_edge


def energia_normalizada(rho=ENERGY_RATIO):
    """Energia por quadro (LOCAL, HIBRIDO, EDGE) / (E_inf + E_tx)."""
    rho = float(rho)
    return np.array([1.0 / (1.0 + rho), 1.0, rho / (1.0 + rho)])


def build_matrix(df, formulacao, alpha=ALPHA_HIBRIDO,
                 remote_inference_ms=REMOTE_INFERENCE_MS,
                 rho=ENERGY_RATIO, rtt_extra_ms=RTT_EDGE_EXTRA_MS,
                 hibrido_latencia=HIBRIDO_LATENCIA,
                 crit_col="criticality", jit_col="rtt_jitter_ms",
                 lat_col=None):
    """Retorna a matriz (n_quadros, 3 alternativas, n_criterios).

    V1-V3: 3 criterios (criticidade, jitter, latencia).
    V4:    4 criterios (criticidade, jitter, latencia, energia).
    lat_col permite trocar a latencia local (ex.: decisao causal com a
    latencia do quadro anterior); por padrao usa total_latency_ms.
    """
    if lat_col is not None and lat_col != "total_latency_ms":
        df = df.assign(total_latency_ms=df[lat_col])
    lat_l, lat_h, lat_e = _latencias(df, remote_inference_ms, rtt_extra_ms)
    crit = df[crit_col].to_numpy(dtype=np.float64)
    jit = df[jit_col].to_numpy(dtype=np.float64)

    if formulacao == "V1":
        exp = np.array([1.0, 1.0, 1.0])          # sem exposicao diferenciada
    elif formulacao in ("V2", "V3"):
        exp = np.array([0.0, alpha, 1.0])        # LOCAL, HIBRIDO, EDGE
    elif formulacao == "V4":
        exp = np.array([0.0, 0.0, 1.0])          # local garante o alerta
        if hibrido_latencia == "min":
            lat_h = np.minimum(lat_l, lat_e)
        elif hibrido_latencia == "local":
            lat_h = lat_l.copy()
        else:
            raise ValueError("hibrido_latencia deve ser 'min' ou 'local'")
    else:
        raise ValueError(f"Formulacao desconhecida: {formulacao}")

    crit_m = crit[:, None] * exp[None, :]
    jit_m = jit[:, None] * exp[None, :]
    lat_m = np.stack([lat_l, lat_h, lat_e], axis=1)
    if formulacao != "V4":
        return np.stack([crit_m, jit_m, lat_m], axis=2)

    ener_m = np.broadcast_to(energia_normalizada(rho), lat_m.shape)
    return np.stack([crit_m, jit_m, lat_m, ener_m], axis=2)


def topsis_cc(matriz, weights, formulacao="V3", ref_scales=None):
    """Coeficiente de proximidade (n_quadros, n_alternativas).

    V1/V2: normalizacao vetorial e ideais relativos, dentro de cada quadro.
    V3/V4: escala de referencia fixa (truncada em 1) e ideais absolutos.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if formulacao in ("V1", "V2"):
        norma = np.sqrt((matriz ** 2).sum(axis=1, keepdims=True))
        norma[norma == 0] = 1.0
        ponderada = (matriz / norma) * weights
        pis = ponderada.min(axis=1, keepdims=True)
        nis = ponderada.max(axis=1, keepdims=True)
    else:
        if ref_scales is None:
            ref_scales = REF_SCALES_V4 if formulacao == "V4" else REF_SCALES
        normalizada = np.clip(matriz / ref_scales, 0.0, 1.0)
        ponderada = normalizada * weights
        pis = np.zeros((1, 1, matriz.shape[2]))   # custo zero
        nis = weights.reshape(1, 1, -1)           # custo maximo da escala

    d_pos = np.sqrt(((ponderada - pis) ** 2).sum(axis=2))
    d_neg = np.sqrt(((ponderada - nis) ** 2).sum(axis=2))
    soma = d_pos + d_neg
    soma[soma == 0] = 1.0
    return d_neg / soma


def topsis_decide(df, weights, formulacao="V3", alpha=ALPHA_HIBRIDO,
                  remote_inference_ms=REMOTE_INFERENCE_MS,
                  ref_scales=None, **kwargs):
    """Decisao TOPSIS por quadro. kwargs vao para build_matrix
    (rho, rtt_extra_ms, hibrido_latencia, crit_col, jit_col, lat_col)."""
    df = df.copy()
    matriz = build_matrix(df, formulacao, alpha, remote_inference_ms, **kwargs)
    cc = topsis_cc(matriz, weights, formulacao, ref_scales)

    escolha = cc.argmax(axis=1)
    df["decisao_topsis"] = [ALTERNATIVAS[i] for i in escolha]
    df["topsis_c_star"] = cc[np.arange(len(df)), escolha]
    return df


def metricas(df):
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
    pesos = WEIGHTS_AHP_V4 if formulacao == "V4" else WEIGHTS_AHP
    df = topsis_decide(df, pesos, formulacao)
    for k, v in metricas(df).items():
        print(f"{k:<22} {v:6.1f}")
