"""
test_qoe_safety.py
-------------------
Testes de propriedades da QoE-Safety v2 e demonstração numérica das falhas da
Eq. (4) do relatório original.

Rodar no terminal (qualquer um dos dois):
    python -m pytest test_qoe_safety.py -v
    python test_qoe_safety.py
"""

import math

import numpy as np
import pandas as pd

from qoe_safety import (
    ParametrosQoE, cdf_reacao, decide, estimar_distancia, foco_relativo,
    prazo_fisico, prob_alerta_no_prazo, qoe_por_alternativa, qoe_safety,
    qoe_safety_original, risco_dinamico_original, avaliar_quadros,
)

P = ParametrosQoE()
RNG_SEED = 42


# --------------------------------------------------------------------------
# 1. Propriedades básicas da nova fórmula
# --------------------------------------------------------------------------
def test_limites_zero_um():
    rng = np.random.default_rng(RNG_SEED)
    for _ in range(500):
        prazo = rng.uniform(-2, 10)
        lat = rng.lognormal(math.log(0.2), 0.6, size=rng.integers(1, 40))
        q = qoe_safety(prazo, lat, P, p_det=rng.uniform(0, 1),
                       atraso_amostragem_max_s=rng.uniform(0, 0.3))
        assert 0.0 <= q <= 1.0


def test_sem_obstaculo_e_ja_tarde():
    assert qoe_safety(math.inf, [0.2], P) == 1.0      # não há aproximação
    assert qoe_safety(0.0, [0.0], P) == 0.0           # prazo esgotado
    assert qoe_safety(-1.0, [0.0], P) == 0.0


def test_sem_piso_artificial():
    # Condição ideal: obstáculo longe, sistema rápido → QoE-Safety → 1.
    assert qoe_safety(10.0, [0.05], P) > 0.9999


def test_monotonicidade():
    lat = np.array([0.15, 0.19, 0.25, 0.40, 0.80])
    D = 1.2
    base = qoe_safety(D, lat, P)
    assert qoe_safety(D, lat + 0.1, P) < base                        # mais latência
    assert qoe_safety(D + 0.3, lat, P) > base                        # mais prazo
    assert qoe_safety(D, lat, P.com(th_mediana_s=1.0)) < base        # reação lenta
    assert qoe_safety(D, lat, P.com(t_alerta_s=0.3)) < base          # TTS lento
    assert qoe_safety(D, lat, P, atraso_amostragem_max_s=0.2) < base # amostragem
    # Prazo cresce com d e cai com v e com a margem
    assert prazo_fisico(3.0, P) > prazo_fisico(2.0, P)
    assert prazo_fisico(3.0, P, v_fechamento_ms=1.5) < prazo_fisico(3.0, P)
    assert prazo_fisico(3.0, P.com(d_margem_m=1.0)) < prazo_fisico(3.0, P)


def test_obstaculo_se_afastando_nao_gera_risco():
    assert np.isinf(prazo_fisico(2.0, P, v_fechamento_ms=-0.5))
    assert qoe_safety(float(prazo_fisico(2.0, P, v_fechamento_ms=0.0)), [0.3], P) == 1.0


def test_jitter_e_dispersao_nao_parcela():
    # Mesma média (0,2 s), dispersões diferentes.
    estavel = np.full(20, 0.20)
    instavel = np.array([0.05] * 15 + [0.65] * 5)
    assert abs(estavel.mean() - instavel.mean()) < 1e-12
    # Prazo folgado: o jitter é irrelevante (Eq. 4 original penalizaria sempre).
    assert abs(qoe_safety(6.0, estavel, P) - qoe_safety(6.0, instavel, P)) < 1e-6
    # Prazo moderado (a mediana cabe, a cauda não): o jitter reduz a QoE-Safety.
    assert qoe_safety(1.6, instavel, P) < qoe_safety(1.6, estavel, P)
    # Prazo muito apertado: o jitter pode AUMENTAR a chance de chegar a tempo
    # (75% dos quadros são rápidos). Uma parcela aditiva nunca capta isso.
    assert qoe_safety(1.0, instavel, P) > qoe_safety(1.0, estavel, P)


def test_quadratura_confere_com_monte_carlo():
    rng = np.random.default_rng(RNG_SEED)
    lat = np.array([0.12, 0.19, 0.26, 0.55])
    D, delta, n = 1.4, 0.12, 400_000
    t = (rng.choice(lat, n) + P.t_alerta_s + rng.uniform(0, delta, n)
         + rng.lognormal(math.log(P.th_mediana_s), P.th_sigma_log, n))
    mc = np.mean(t <= D)
    analitico = prob_alerta_no_prazo(D, lat, P, atraso_amostragem_max_s=delta)
    assert abs(mc - analitico) < 0.005, (mc, analitico)


def test_cdf_reacao_mediana():
    assert abs(cdf_reacao([P.th_mediana_s], P)[0] - 0.5) < 1e-12
    assert cdf_reacao([0.0], P)[0] == 0.0


def test_distancia_pinhole():
    f = foco_relativo(P)
    prox_4m = f * 1.70 / 4.0                       # pessoa a 4 m
    d, trunc, padrao = estimar_distancia([prox_4m, prox_4m / 2, 1.0, 0.3, 0.0],
                                         ["person", "person", "person", "xyz", "person"], P)
    assert abs(d[0] - 4.0) < 1e-9
    assert abs(d[1] - 8.0) < 1e-9                  # metade da caixa → dobro da distância
    assert trunc[2] and not trunc[0]               # caixa cheia → limite superior
    assert padrao[3] and not padrao[0]             # classe sem prior sinalizada
    assert np.isinf(d[4])                          # sem caixa → sem obstáculo


def test_limiares_iguais_ao_fuzzy():
    assert decide(0.3499) == "EDGE"
    assert decide(0.35) == "HIBRIDO"
    assert decide(0.6499) == "HIBRIDO"
    assert decide(0.65) == "LOCAL"


def test_alternativas_ordenadas():
    q = qoe_por_alternativa(1.3, {"LOCAL": [0.35, 0.40], "EDGE": [0.10, 0.12]}, P,
                            p_det_por_k={"LOCAL": 0.80, "EDGE": 0.95})
    assert q["EDGE"] > q["LOCAL"]


def test_predicao_sem_vazamento():
    # A predição do quadro i só pode usar quadros < i.
    n = 60
    df = pd.DataFrame({
        "frame_id": range(n), "rtt_ms": 30, "wait_ms": 20, "detect_ms": 120,
        "total_latency_ms": 170, "num_objects": 1, "label": "person",
        "score": 0.7, "proximity": 0.6, "session": "A",
    })
    a = avaliar_quadros(df, P)
    df.loc[40, "total_latency_ms"] = 5000           # altera só o quadro 40
    b = avaliar_quadros(df, P)
    assert a.loc[40, "qoe_pred_reativa"] == b.loc[40, "qoe_pred_reativa"]
    assert b.loc[41, "qoe_pred_reativa"] < a.loc[41, "qoe_pred_reativa"]
    assert b.loc[40, "p_prazo_realizado"] < a.loc[40, "p_prazo_realizado"]


# --------------------------------------------------------------------------
# 2. Por que a Eq. (4) original foi substituída (parâmetros ilustrativos)
# --------------------------------------------------------------------------
def test_original_tem_piso_theta():
    # Folga máxima, visão perfeita, rede perfeita, bateria cheia:
    q = qoe_safety_original(gamma=1.0, u_visao=0.0, psi_6g=0.0, e_bat=1.0, theta=0.2)
    assert q <= 0.8 + 1e-12                        # nunca passa de 1 − θ


def test_original_bateria_zera_seguranca():
    # Bateria a 5%: θ/E = 4 → risco satura em 1 mesmo com folga máxima.
    r = risco_dinamico_original(gamma=1.0, u_visao=0.0, psi_6g=0.0, e_bat=0.05)
    assert r > 1.0
    assert qoe_safety_original(1.0, 0.0, 0.0, 0.05) == 0.0


def test_original_penaliza_incerteza_sem_obstaculo():
    # Sem obstáculo (folga máxima), a incerteza visual ainda sobe o risco.
    a = risco_dinamico_original(1.0, 0.0, 0.0, 1.0)
    b = risco_dinamico_original(1.0, 1.0, 0.0, 1.0)
    assert b - a > 0.19
    # Na v2, sem obstáculo não há prazo a cumprir: QoE-Safety = 1.
    assert qoe_safety(math.inf, [0.2], P) == 1.0


def test_original_psi_penaliza_mesmo_com_folga():
    a = qoe_safety_original(1.0, 0.0, 0.0, 1.0)
    b = qoe_safety_original(1.0, 0.0, 1.0, 1.0)
    assert a - b > 0.19                            # rede ruim pune mesmo sem risco físico


if __name__ == "__main__":
    testes = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    falhas = 0
    for nome, fn in testes:
        try:
            fn()
            print(f"PASSOU  {nome}")
        except AssertionError as e:
            falhas += 1
            print(f"FALHOU  {nome}  {e}")
    print(f"\n{len(testes) - falhas}/{len(testes)} testes passaram.")
    raise SystemExit(1 if falhas else 0)
