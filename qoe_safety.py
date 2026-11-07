"""
qoe_safety.py
--------------
QoE-Safety v2: probabilidade de o alerta chegar a tempo de o usuário parar.

Substitui a Eq. (4) do relatório "QoE-Safety e Risco Dinâmico 6G" (sigmoide
+ termos aditivos com pesos livres) por uma grandeza física e probabilística,
sem pesos arbitrários. Todos os termos são tempos (s), distâncias (m) ou
probabilidades medidas.

Notação (para cada alternativa de processamento k ∈ {LOCAL, HIBRIDO, EDGE}):

    d        distância ao obstáculo no instante da captura            [m]
    v        velocidade de aproximação (obstáculo estático: a do usuário) [m/s]
    d_m      margem física = distância de parada + folga               [m]
    D        prazo físico = (d - d_m) / v                              [s]
    T_k      latência do caminho k, da captura ao fim da inferência    [s]
    T_a      latência do alerta (início do áudio/TTS)                  [s]
    U        atraso de amostragem, U ~ Uniforme(0, Δ) (só prospectivo) [s]
    T_h      tempo de reação do usuário ao alerta (variável aleatória) [s]

    QoE-Safety_k = P_det,k(d) · P(T_k + T_a + U + T_h ≤ D)            (Eq. 6)
    risco_k      = 1 − QoE-Safety_k                                    (Eq. 7)

Dois modos de uso:
  * reativo     — há obstáculo detectado; d vem da própria detecção.
  * prospectivo — não há obstáculo; avalia a prontidão do caminho k para um
                  obstáculo de referência que surge a d_ref (ex.: sai de trás
                  de uma porta). Inclui o atraso de amostragem U.

A distribuição de T_k é empírica: as últimas `janela` latências observadas
(estritamente passadas) no mesmo caminho. O jitter entra aqui como dispersão
dessa distribuição, e não como parcela somada (evita a dupla contagem da
Eq. 2 + Ψ_6G do relatório original).

Nada neste módulo depende de scipy: só numpy e pandas.

IMPORTANTE — parâmetros marcados como PLACEHOLDER não foram medidos. Eles
precisam ser estimados antes de qualquer resultado entrar num artigo:
  - th_mediana_s / th_sigma_log : estudo com participantes (CEP/CONEP)
  - t_alerta_s                  : medir o início do TTS no app
  - fov_vertical_graus          : calibração da câmera (tabuleiro de xadrez)
  - ALTURA_CLASSE_M             : priors de altura por classe
  - d_margem_m, d_referencia_m  : definição de projeto, justificar no texto
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Limiares de decisão: os mesmos já decididos para o neuro-fuzzy
# (fuzzy_system.decide). Mantidos idênticos para não reabrir a decisão.
# ---------------------------------------------------------------------------
LIMIAR_EDGE = 0.35
LIMIAR_LOCAL = 0.65


# ---------------------------------------------------------------------------
# PLACEHOLDER: altura típica por classe COCO [m], para a distância pinhole.
# Valores aproximados de projeto, não medidos. Classes ausentes usam
# ALTURA_PADRAO_M e são sinaladas na coluna `altura_prior_padrao`.
# ---------------------------------------------------------------------------
ALTURA_CLASSE_M = {
    "person": 1.70,
    "bicycle": 1.10,
    "car": 1.50,
    "motorcycle": 1.20,
    "bus": 3.00,
    "truck": 3.00,
    "train": 3.80,
    "bench": 0.85,
    "chair": 0.90,
    "dog": 0.60,
    "cat": 0.30,
    "horse": 1.60,
    "cow": 1.40,
    "fire hydrant": 0.75,
    "stop sign": 2.20,
    "traffic light": 3.00,
    "potted plant": 0.80,
    "tv": 0.60,
    "sink": 0.90,
    "oven": 0.90,
    "refrigerator": 1.75,
}
ALTURA_PADRAO_M = 1.00


@dataclass(frozen=True)
class ParametrosQoE:
    # Cinemática
    v_caminhada_ms: float = 1.2          # decisão da tese (≈ 1,2 m/s)
    d_margem_m: float = 0.5              # PLACEHOLDER: parada + folga
    # Fator humano: T_h ~ LogNormal(mediana, sigma_log)
    th_mediana_s: float = 0.8            # PLACEHOLDER: medir com participantes
    th_sigma_log: float = 0.3            # PLACEHOLDER
    # Alerta
    t_alerta_s: float = 0.15             # PLACEHOLDER: medir início do TTS
    # Câmera (OV2640): FOV vertical derivado de ~66° diagonal, 4:3
    fov_vertical_graus: float = 42.6     # PLACEHOLDER: calibrar
    proximidade_truncada: float = 0.98   # caixa ocupa ~todo o quadro
    # Modo prospectivo
    d_referencia_m: float = 3.0          # PLACEHOLDER: obstáculo de referência
    p_det_referencia: float = 1.0        # PLACEHOLDER: revocação do modelo a d_ref
    # Estimação de F_k
    # Quadros passados usados como F_k. No piloto, janelas curtas (5–60)
    # previram a cauda PIOR que a climatologia da sessão (BSS < 0) e 120
    # empatou (ver avaliar_qoe_safety.py). Por isso o padrão é longo: o
    # condicionamento de curto prazo deve vir de variáveis exógenas (SINR,
    # falha de feixe), não da própria série de latência.
    janela: int = 120
    min_historico: int = 5               # abaixo disso, a predição é NaN
    n_quadratura: int = 16               # pontos para integrar U ~ Unif(0, Δ)

    def com(self, **kw) -> "ParametrosQoE":
        """Cópia com parâmetros alterados (útil para sensibilidade)."""
        return replace(self, **kw)


# ---------------------------------------------------------------------------
# Blocos da fórmula
# ---------------------------------------------------------------------------
_erf = np.vectorize(math.erf, otypes=[float])


def cdf_reacao(t_s, p: ParametrosQoE) -> np.ndarray:
    """G(t) = P(T_h ≤ t), T_h lognormal. G(t) = 0 para t ≤ 0."""
    t = np.asarray(t_s, dtype=float)
    out = np.zeros_like(t)
    pos = t > 0
    if np.any(pos):
        z = (np.log(t[pos]) - math.log(p.th_mediana_s)) / (p.th_sigma_log * math.sqrt(2.0))
        out[pos] = 0.5 * (1.0 + _erf(z))
    return out


def foco_relativo(p: ParametrosQoE) -> float:
    """f / H_quadro = 1 / (2·tan(FOV_v / 2))."""
    return 1.0 / (2.0 * math.tan(math.radians(p.fov_vertical_graus) / 2.0))


def estimar_distancia(proximidade, label, p: ParametrosQoE):
    """
    Distância pinhole: d ≈ (f / H_quadro) · H_classe / (h_caixa / H_quadro).
    `proximidade` é a razão altura_caixa / altura_quadro registrada pelo app.

    Retorna (d_m, truncada, prior_padrao). Quando a caixa ocupa quase todo o
    quadro (truncada=True) o objeto pode estar cortado e d é só um LIMITE
    SUPERIOR da distância real.
    """
    prox = np.asarray(proximidade, dtype=float)
    labels = np.asarray(label, dtype=object)
    alturas = np.array([ALTURA_CLASSE_M.get(str(l), ALTURA_PADRAO_M) for l in labels])
    prior_padrao = np.array([str(l) not in ALTURA_CLASSE_M for l in labels])
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(prox > 0, foco_relativo(p) * alturas / np.clip(prox, 1e-6, 1.0), np.inf)
    truncada = prox >= p.proximidade_truncada
    return d, truncada, prior_padrao


def tempo_ate_colisao(d_m, v_fechamento_ms):
    """TTC = d / v. Infinito quando não há obstáculo ou não há aproximação (v ≤ 0)."""
    d = np.asarray(d_m, dtype=float)
    v = np.broadcast_to(np.asarray(v_fechamento_ms, dtype=float), d.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((v > 0) & np.isfinite(d), d / np.where(v > 0, v, 1.0), np.inf)


def prazo_fisico(d_m, p: ParametrosQoE, v_fechamento_ms=None):
    """D = (d − d_m) / v = TTC − d_m / v. Pode ser negativo (já é tarde)."""
    v = p.v_caminhada_ms if v_fechamento_ms is None else v_fechamento_ms
    return tempo_ate_colisao(np.asarray(d_m, dtype=float) - p.d_margem_m, v)


def prob_alerta_no_prazo(prazo_s: float, latencias_s, p: ParametrosQoE,
                         atraso_amostragem_max_s: float = 0.0) -> float:
    """
    P(T_k + T_a + U + T_h ≤ D), com T_k amostrado das latências dadas,
    U ~ Unif(0, Δ) integrado por quadratura de ponto médio e T_h lognormal.
    Determinística: não há sorteio, logo não há semente.
    """
    if not np.isfinite(prazo_s):
        return 1.0 if prazo_s > 0 else 0.0
    lat = np.atleast_1d(np.asarray(latencias_s, dtype=float))
    lat = lat[np.isfinite(lat)]
    if lat.size == 0:
        return float("nan")
    if atraso_amostragem_max_s > 0:
        u = (np.arange(p.n_quadratura) + 0.5) / p.n_quadratura * atraso_amostragem_max_s
    else:
        u = np.zeros(1)
    folga = prazo_s - lat[:, None] - p.t_alerta_s - u[None, :]
    return float(np.mean(cdf_reacao(folga, p)))


def qoe_safety(prazo_s: float, latencias_s, p: ParametrosQoE, p_det: float = 1.0,
               atraso_amostragem_max_s: float = 0.0) -> float:
    """Eq. (6): QoE-Safety_k = P_det,k · P(alerta no prazo)."""
    if not 0.0 <= p_det <= 1.0:
        raise ValueError("p_det deve estar em [0, 1]")
    return p_det * prob_alerta_no_prazo(prazo_s, latencias_s, p, atraso_amostragem_max_s)


def risco(qoe: float) -> float:
    """Eq. (7): risco = 1 − QoE-Safety (identidade mantida da versão publicada)."""
    return 1.0 - qoe


def decide(risco_valor: float) -> str:
    """Mesmos limiares do fuzzy_system.decide (0,35 / 0,65)."""
    if not np.isfinite(risco_valor):
        return "SEM_HISTORICO"
    if risco_valor < LIMIAR_EDGE:
        return "EDGE"
    if risco_valor < LIMIAR_LOCAL:
        return "HIBRIDO"
    return "LOCAL"


def qoe_por_alternativa(prazo_s: float, latencias_por_k: dict, p: ParametrosQoE,
                        p_det_por_k: dict | None = None,
                        atraso_por_k: dict | None = None) -> dict:
    """
    Avalia a Eq. (6) para cada alternativa k. `latencias_por_k[k]` são amostras
    de T_k (s) — telemetria real ou traces NS-3. Retorna {k: QoE-Safety_k}.
    """
    p_det_por_k = p_det_por_k or {}
    atraso_por_k = atraso_por_k or {}
    return {
        k: qoe_safety(prazo_s, lat, p, p_det_por_k.get(k, 1.0), atraso_por_k.get(k, 0.0))
        for k, lat in latencias_por_k.items()
    }


# ---------------------------------------------------------------------------
# Aplicação à telemetria (DataFrame produzido por parser_logcat + features)
# ---------------------------------------------------------------------------
def _historico(serie: np.ndarray, i: int, janela: int) -> np.ndarray:
    return serie[max(0, i - janela):i]


def coluna_latencia(df: pd.DataFrame) -> str:
    """
    Coluna usada como T_k (captura -> fim da inferência).

    'e2e_latency_ms' (app + firmware com idade do quadro) inclui o tempo que
    o quadro passou no buffer do ESP32; é usada quando existe para TODOS os
    quadros. Caso contrário (logs do piloto, firmware antigo, ou mistura),
    usa 'total_latency_ms' (pedido -> fim da inferência) em todos, para não
    misturar definições na mesma distribuição empírica.
    """
    if "e2e_latency_ms" in df.columns and len(df) and df["e2e_latency_ms"].notna().all():
        return "e2e_latency_ms"
    return "total_latency_ms"


def label_obstaculo(df: pd.DataFrame) -> pd.Series:
    """
    Rótulo do objeto da coluna proximity (maior caixa): 'closest_label'
    quando existe; senão 'label' (logs do piloto, em que o rótulo é o do
    objeto de maior score e pode ser de outro objeto se houver 2+).
    """
    if "closest_label" in df.columns:
        return df["closest_label"].where(df["closest_label"].notna(), df["label"])
    return df["label"]


def avaliar_quadros(df: pd.DataFrame, p: ParametrosQoE = ParametrosQoE()) -> pd.DataFrame:
    """
    Acrescenta ao DataFrame as colunas da QoE-Safety v2 para o caminho
    OBSERVADO no piloto (ESP32-CAM → Wi-Fi → inferência no smartphone).

    Latência T_k: ver coluna_latencia(). Rótulo do obstáculo (prior de
    altura da distância pinhole): ver label_obstaculo().

    Colunas novas:
      latencia_fonte       coluna usada como T_k
      label_obstaculo      rótulo usado na distância
      detectado            há obstáculo com proximidade > 0
      dist_m               distância pinhole (NaN sem obstáculo)
      dist_truncada        caixa ≥ 98% do quadro: d é limite superior
      altura_prior_padrao  classe sem prior de altura (usa 1,0 m)
      ttc_s, prazo_s       TTC e prazo físico D
      p_prazo_realizado    DESFECHO: P(alerta no prazo | latência realizada
                           deste quadro); média só sobre T_h. Não é preditivo.
      p_prazo_lat_zero     idem, com latência do sistema = 0 e T_a = 0
                           (limite físico/humano: tarde mesmo com sistema ideal)
      qoe_pred_reativa     PREDIÇÃO com as `janela` latências passadas
      qoe_pred_prospectiva prontidão para obstáculo de referência a d_ref
      qoe_safety_v2        reativa se há obstáculo, prospectiva se não há
      risco_v2, decisao_v2
    """
    df = df.copy()
    col_lat = coluna_latencia(df)
    lat_s = df[col_lat].to_numpy(dtype=float) / 1000.0
    det_s = df["detect_ms"].to_numpy(dtype=float) / 1000.0
    sess = df["session"].to_numpy() if "session" in df.columns else np.zeros(len(df))

    prox = df["proximity"].fillna(0.0).to_numpy(dtype=float)
    detectado = (df["num_objects"].to_numpy() > 0) & (prox > 0)
    labels = label_obstaculo(df)
    d, trunc, prior_padrao = estimar_distancia(prox, labels.to_numpy(), p)
    d = np.where(detectado, d, np.nan)

    df["latencia_fonte"] = col_lat
    df["label_obstaculo"] = labels
    df["detectado"] = detectado
    df["dist_m"] = d
    df["dist_truncada"] = detectado & trunc
    df["altura_prior_padrao"] = detectado & prior_padrao
    df["ttc_s"] = np.where(detectado, tempo_ate_colisao(d, p.v_caminhada_ms), np.nan)
    prazo = np.where(detectado, prazo_fisico(d, p), np.nan)
    df["prazo_s"] = prazo

    prazo_ref = float(prazo_fisico(p.d_referencia_m, p))
    p_zero = p.com(t_alerta_s=0.0)

    real, zero, reat, prosp = ([np.nan] * len(df) for _ in range(4))
    for s in np.unique(sess):
        idx = np.flatnonzero(sess == s)
        lat_sess, det_sess = lat_s[idx], det_s[idx]
        for j, i in enumerate(idx):
            if detectado[i]:
                real[i] = prob_alerta_no_prazo(prazo[i], lat_s[i], p)
                zero[i] = prob_alerta_no_prazo(prazo[i], 0.0, p_zero)
            hist = _historico(lat_sess, j, p.janela)
            if hist.size < p.min_historico:
                continue
            delta = float(np.median(_historico(det_sess, j, p.janela)))
            prosp[i] = qoe_safety(prazo_ref, hist, p, p.p_det_referencia, delta)
            if detectado[i]:
                reat[i] = qoe_safety(prazo[i], hist, p, 1.0)

    df["p_prazo_realizado"] = real
    df["p_prazo_lat_zero"] = zero
    df["qoe_pred_reativa"] = reat
    df["qoe_pred_prospectiva"] = prosp
    df["qoe_safety_v2"] = np.where(detectado, df["qoe_pred_reativa"], df["qoe_pred_prospectiva"])
    df["risco_v2"] = 1.0 - df["qoe_safety_v2"]
    df["decisao_v2"] = df["risco_v2"].apply(decide)
    return df


# ---------------------------------------------------------------------------
# Eq. (4) do relatório original — mantida SÓ para os testes que demonstram
# por que ela foi substituída. Não usar no pipeline.
# ---------------------------------------------------------------------------
def risco_dinamico_original(gamma, u_visao, psi_6g, e_bat,
                            alpha=10.0, gamma0=0.3, beta=0.2, delta=0.2, theta=0.2):
    sig = 1.0 / (1.0 + np.exp(alpha * (np.asarray(gamma, float) - gamma0)))
    return sig + beta * np.asarray(u_visao, float) + delta * np.asarray(psi_6g, float) \
        + theta / np.asarray(e_bat, float)


def qoe_safety_original(*args, **kw):
    return 1.0 - np.minimum(1.0, risco_dinamico_original(*args, **kw))


if __name__ == "__main__":
    p = ParametrosQoE()
    print(f"f/H_quadro = {foco_relativo(p):.3f}")
    for prox, label in [(0.325, "person"), (0.60, "bench"), (0.93, "dog")]:
        d, _, _ = estimar_distancia([prox], [label], p)
        D = float(prazo_fisico(d[0], p))
        q = qoe_safety(D, [0.19], p)
        print(f"{label:<7} prox={prox:.2f}  d={d[0]:.2f} m  D={D:.2f} s  "
              f"QoE-Safety={q:.3f}  decisão={decide(risco(q))}")
