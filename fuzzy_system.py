"""
fuzzy_system.py
-----------------

Entradas:
  - latencia_ms   : latência total ponta-a-ponta (rede + fila + detecção)
  - jitter_ms     : instabilidade da rede (desvio padrão móvel do RTT)
  - criticidade   : criticidade da detecção (0 a 1)

Saída:
  - risco (0 a 1) : quanto maior, maior o risco de manter/mover o
                    processamento para a borda nas condições atuais.

Decisão (a partir do risco de acidente):
    risco < 0.35            -> "EDGE"     (offload para a borda)
    0.35 <= risco < 0.65    -> "HIBRIDO"  (mantém local, mas pode
                                            consultar a borda em paralelo)
    risco >= 0.65           -> "LOCAL"    (processamento embarcado apenas)

"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def build_fuzzy_system(
    latencia_range=(0, 800),
    jitter_range=(0, 400),
):
    latencia = ctrl.Antecedent(np.linspace(*latencia_range, 200), "latencia")
    jitter = ctrl.Antecedent(np.linspace(*jitter_range, 200), "jitter")
    criticidade = ctrl.Antecedent(np.linspace(0, 1, 100), "criticidade")
    risco = ctrl.Consequent(np.linspace(0, 1, 100), "risco")

    # --- Funções de pertinência (baseadas nas faixas observadas nos dados reais) ---
    latencia["baixa"] = fuzz.trapmf(latencia.universe, [0, 0, 80, 150])
    latencia["media"] = fuzz.trimf(latencia.universe, [100, 200, 350])
    latencia["alta"] = fuzz.trapmf(latencia.universe, [250, 400, latencia_range[1], latencia_range[1]])

    jitter["estavel"] = fuzz.trapmf(jitter.universe, [0, 0, 20, 60])
    jitter["instavel"] = fuzz.trapmf(jitter.universe, [40, 100, jitter_range[1], jitter_range[1]])

    criticidade["baixa"] = fuzz.trapmf(criticidade.universe, [0, 0, 0.2, 0.4])
    criticidade["media"] = fuzz.trimf(criticidade.universe, [0.3, 0.5, 0.7])
    criticidade["alta"] = fuzz.trapmf(criticidade.universe, [0.6, 0.8, 1.0, 1.0])

    risco["baixo"] = fuzz.trapmf(risco.universe, [0, 0, 0.2, 0.4])
    risco["medio"] = fuzz.trimf(risco.universe, [0.3, 0.5, 0.7])
    risco["alto"] = fuzz.trapmf(risco.universe, [0.6, 0.8, 1.0, 1.0])

    rules = [
        # Segurança em primeiro lugar: qualquer condição crítica isolada já
        # eleva o risco (favorece processamento local).
        ctrl.Rule(latencia["alta"], risco["alto"]),
        ctrl.Rule(jitter["instavel"], risco["alto"]),
        ctrl.Rule(criticidade["alta"], risco["alto"]),

        # Condições intermediárias
        ctrl.Rule(latencia["media"] & jitter["estavel"] & criticidade["media"], risco["medio"]),
        ctrl.Rule(latencia["media"] & jitter["estavel"] & criticidade["baixa"], risco["medio"]),
        ctrl.Rule(latencia["baixa"] & jitter["estavel"] & criticidade["media"], risco["medio"]),
        ctrl.Rule(latencia["baixa"] & jitter["instavel"] & criticidade["baixa"], risco["medio"]),

        # Condição favorável para offload à borda: rede rápida, estável e
        # obstáculo pouco crítico.
        ctrl.Rule(latencia["baixa"] & jitter["estavel"] & criticidade["baixa"], risco["baixo"]),
    ]

    system = ctrl.ControlSystem(rules)
    return system, (latencia, jitter, criticidade, risco)


def compute_risk(simulation: ctrl.ControlSystemSimulation, latencia_ms: float, jitter_ms: float, criticidade: float) -> float:
    simulation.input["latencia"] = float(np.clip(latencia_ms, 0, 800))
    simulation.input["jitter"] = float(np.clip(jitter_ms, 0, 400))
    simulation.input["criticidade"] = float(np.clip(criticidade, 0, 1))
    simulation.compute()
    return float(simulation.output["risco"])


def decide(risco: float) -> str:
    if risco < 0.35:
        return "EDGE"
    elif risco < 0.65:
        return "HIBRIDO"
    else:
        return "LOCAL"


if __name__ == "__main__":
    system, _ = build_fuzzy_system()
    sim = ctrl.ControlSystemSimulation(system)

    casos = [
        ("Rede ótima, sem obstáculo crítico", 60, 10, 0.1),
        ("Rede degradada, obstáculo crítico", 500, 200, 0.9),
        ("Condição intermediária", 180, 40, 0.5),
    ]
    for nome, lat, jit, crit in casos:
        r = compute_risk(sim, lat, jit, crit)
        print(f"{nome}: risco={r:.3f} | QoE-Safety={1 - r:.3f} | decisão={decide(r)}")
