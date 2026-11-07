"""
avaliar_qoe_safety.py
----------------------
Testa a QoE-Safety v2 (qoe_safety.py) sobre a telemetria do piloto
(resultado_dataset.csv, gerado por main.py).

Uso (no terminal, na pasta codigo_python):
    python avaliar_qoe_safety.py
    python avaliar_qoe_safety.py outro_dataset.csv

Saídas:
    qoe_safety_v2_dataset.csv   — telemetria + colunas da v2 por quadro
    qoe_safety_v2_resumo.json   — todos os números citados no relatório
    qoe_safety_v2_graficos.png  — figura de 4 painéis (+ .pdf vetorial)

Análises:
  A. Desfecho: taxa esperada de alertas fora do prazo nos quadros com
     obstáculo, e quanto dela sobra com latência zero (limite físico/humano).
  B. Calibração preditiva de F_k: a janela recente de latências prevê a
     cauda da próxima latência melhor que a climatologia da sessão?
  C. Comparação com o fuzzy especialista (qoe_safety_fuzzy / decisao_fuzzy).
  D. Sensibilidade um-fator-por-vez aos parâmetros PLACEHOLDER.
  E. Distância mínima de detecção d* para QoE-Safety ≥ 0,95.

Intervalos de confiança: bootstrap de blocos móveis (quadros consecutivos
são correlacionados), 2.000 reamostragens, semente 42.

ATENÇÃO: o piloto tem um único caminho (ESP32-CAM → Wi-Fi → inferência no
smartphone), sem obstáculos com distância medida e sem tempo de reação
medido. Os resultados validam o comportamento da fórmula e mostram onde
está o gargalo; não são evidência de desfecho para um artigo.
"""

import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qoe_safety import ParametrosQoE, avaliar_quadros, coluna_latencia, prazo_fisico, qoe_safety

SEMENTE = 42
N_BOOT = 2000
TAU_S = 0.30          # evento de cauda para a calibração: latência > 300 ms
Q_ALVO = 0.95         # QoE-Safety mínima para calcular d*

# Paleta segura para daltonismo (slots 1–3 validados em todos os pares) + cinza
AZUL, LARANJA, AQUA, CINZA, TINTA = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#52514e"


def br(x, casas=2):
    """Número com vírgula decimal (pt-BR) para os rótulos da figura."""
    return f"{x:.{casas}f}".replace(".", ",").replace("-", "\u2212")


# ---------------------------------------------------------------------------
def bootstrap_blocos(valores, estatistica=np.mean, bloco=10, n=N_BOOT, semente=SEMENTE):
    """IC 95% por bootstrap de blocos móveis (percentil)."""
    v = np.asarray(valores, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan)
    bloco = max(1, min(bloco, v.size))
    rng = np.random.default_rng(semente)
    n_blocos = int(np.ceil(v.size / bloco))
    inicios_validos = v.size - bloco + 1
    est = np.empty(n)
    for b in range(n):
        ini = rng.integers(0, inicios_validos, n_blocos)
        amostra = np.concatenate([v[s:s + bloco] for s in ini])[:v.size]
        est[b] = estatistica(amostra)
    return tuple(np.percentile(est, [2.5, 97.5]))


def taxa_atraso(df):
    """Taxa esperada de alertas fora do prazo = média de 1 − P(no prazo)."""
    sub = df[df["detectado"]]
    return float(np.mean(1.0 - sub["p_prazo_realizado"]))


def d_minima(latencias, p, q=Q_ALVO, lo=0.0, hi=30.0):
    """Menor d com QoE-Safety ≥ q (bissecção; QoE cresce com d)."""
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if qoe_safety(float(prazo_fisico(mid, p)), latencias, p) >= q:
            hi = mid
        else:
            lo = mid
    return hi


JANELAS = [5, 10, 30, 60, 120]


def calibracao_janela(df, w, min_hist):
    """
    Evento: latência do próximo quadro > TAU_S. Compara a previsão da janela
    das últimas w latências com a climatologia (todas as latências passadas da
    sessão). BSS > 0: a janela é melhor. IC95 por bootstrap de blocos (30).
    """
    pj, pc, y, cob = [], [], [], []
    col_lat = coluna_latencia(df)
    for _, sub in df.groupby("session", sort=False):
        lat = sub[col_lat].to_numpy(dtype=float) / 1000.0
        for j in range(min_hist, len(lat)):
            jan = lat[max(0, j - w):j]
            pj.append(np.mean(jan > TAU_S))
            pc.append(np.mean(lat[:j] > TAU_S))
            y.append(float(lat[j] > TAU_S))
            cob.append(float(lat[j] <= np.quantile(jan, 0.9)))
    pj, pc, y = map(np.asarray, (pj, pc, y))
    bj, bc = (pj - y) ** 2, (pc - y) ** 2
    rng = np.random.default_rng(SEMENTE)
    m, bloco = len(y), 30
    boot = []
    for _ in range(N_BOOT):
        ini = rng.integers(0, m - bloco + 1, int(np.ceil(m / bloco)))
        idx = np.concatenate([np.arange(i, i + bloco) for i in ini])[:m]
        boot.append(1.0 - bj[idx].mean() / bc[idx].mean())
    return {
        "n_predicoes": int(m),
        "freq_evento": float(y.mean()),
        "brier_janela": float(bj.mean()),
        "brier_climatologia": float(bc.mean()),
        "bss": float(1.0 - bj.mean() / bc.mean()),
        "bss_ic95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        "cobertura_q90": float(np.mean(cob)),
    }


# ---------------------------------------------------------------------------
def analisar(csv_path="resultado_dataset.csv", prefixo="qoe_safety_v2"):
    p = ParametrosQoE()
    bruto = pd.read_csv(csv_path)
    df = avaliar_quadros(bruto, p)
    res = {"parametros": p.__dict__, "n_quadros": int(len(df))}
    col_lat = coluna_latencia(bruto)   # mesma coluna usada em avaliar_quadros
    res["latencia_fonte"] = col_lat

    # ---------------- A. Desfecho e decomposição ----------------
    det = df[df["detectado"]]
    res["n_detectados"] = int(len(det))
    res["n_detectados_por_sessao"] = det.groupby("session").size().astype(int).to_dict()
    res["n_dist_truncada"] = int(det["dist_truncada"].sum())
    res["n_prior_padrao"] = int(det["altura_prior_padrao"].sum())
    res["dist_m_quartis"] = [float(x) for x in det["dist_m"].quantile([0.25, 0.5, 0.75])]
    res["latencia_ms_percentis"] = {
        k: float(v) for k, v in zip(["p50", "p95", "p99", "max"],
                                    df[col_lat].quantile([0.5, 0.95, 0.99, 1.0]))}

    def bloco_desfecho(sub):
        atraso = 1.0 - sub["p_prazo_realizado"].to_numpy()
        atraso0 = 1.0 - sub["p_prazo_lat_zero"].to_numpy()
        return {
            "n": int(len(sub)),
            "taxa_atraso": float(atraso.mean()),
            "taxa_atraso_ic95": [float(x) for x in bootstrap_blocos(atraso)],
            "taxa_atraso_lat_zero": float(atraso0.mean()),
            "taxa_atraso_lat_zero_ic95": [float(x) for x in bootstrap_blocos(atraso0)],
            "parcela_sistema": float(atraso.mean() - atraso0.mean()),
            "parcela_sistema_ic95": [float(x) for x in bootstrap_blocos(atraso - atraso0)],
            "frac_quadros_p_menor_095": float(np.mean(sub["p_prazo_realizado"] < 0.95)),
        }

    res["desfecho"] = {"total": bloco_desfecho(det)}
    atr = (1.0 - det["p_prazo_realizado"])
    # Agrupa pelo rótulo usado na distância (objeto da maior caixa quando disponível)
    por_classe = atr.groupby(det["label_obstaculo"]).agg(["size", "sum"]).sort_values("sum", ascending=False)
    res["atraso_por_classe"] = {
        str(k): {"n": int(r["size"]), "fracao_do_atraso": float(r["sum"] / atr.sum()),
                 "dist_mediana_m": float(det.loc[det["label_obstaculo"] == k, "dist_m"].median())}
        for k, r in por_classe.head(6).iterrows()}
    res["fracao_atraso_d_menor_2m"] = float(atr[det["dist_m"] < 2.0].sum() / atr.sum())
    res["n_d_menor_2m"] = int((det["dist_m"] < 2.0).sum())
    for s, sub in det.groupby("session"):
        res["desfecho"][f"sessao_{s}"] = bloco_desfecho(sub)

    # ---------------- B. Calibração preditiva de F_k ----------------
    res["calibracao"] = {"tau_s": TAU_S, "por_janela": {}}
    for w in JANELAS:
        r = calibracao_janela(df, w, p.min_historico)
        res["calibracao"]["por_janela"][str(w)] = r
    res["calibracao"]["janela_usada"] = p.janela
    res["calibracao"]["freq_evento"] = res["calibracao"]["por_janela"][str(JANELAS[0])]["freq_evento"]

    # ---------------- C. Comparação com o fuzzy especialista ----------------
    ok = df["qoe_safety_v2"].notna()
    res["comparacao_fuzzy"] = {
        "spearman_todos": float(df.loc[ok, ["qoe_safety_fuzzy", "qoe_safety_v2"]]
                                .corr(method="spearman").iloc[0, 1]),
        "spearman_detectados": float(df.loc[ok & df["detectado"], ["qoe_safety_fuzzy", "qoe_safety_v2"]]
                                     .corr(method="spearman").iloc[0, 1]),
        "decisoes_fuzzy_pct": (df["decisao_fuzzy"].value_counts(normalize=True) * 100).round(1).to_dict(),
        "decisoes_v2_pct": (df.loc[ok, "decisao_v2"].value_counts(normalize=True) * 100).round(1).to_dict(),
        "n_sem_historico": int((~ok).sum()),
        "tabela_cruzada": pd.crosstab(df.loc[ok, "decisao_fuzzy"], df.loc[ok, "decisao_v2"]).to_dict(),
        "qoe_prospectiva_mediana": float(df["qoe_pred_prospectiva"].median()),
        "qoe_prospectiva_min": float(df["qoe_pred_prospectiva"].min()),
    }

    # ---------------- D. Sensibilidade um-fator-por-vez ----------------
    base = taxa_atraso(df)
    variacoes = [
        ("Reação T_h (mediana 0,5–1,2 s)", "th_mediana_s", 0.5, 1.2),
        ("Dispersão de T_h (σ 0,2–0,5)", "th_sigma_log", 0.2, 0.5),
        ("Margem d_m (0,3–1,0 m)", "d_margem_m", 0.3, 1.0),
        ("Velocidade (0,9–1,5 m/s)", "v_caminhada_ms", 0.9, 1.5),
        ("FOV vertical (38–48°)", "fov_vertical_graus", 38.0, 48.0),
        ("Alerta/TTS (0,05–0,30 s)", "t_alerta_s", 0.05, 0.30),
    ]
    sens = []
    for rotulo, campo, lo, hi in variacoes:
        a = taxa_atraso(avaliar_quadros(bruto, p.com(**{campo: lo})))
        b = taxa_atraso(avaliar_quadros(bruto, p.com(**{campo: hi})))
        sens.append({"fator": rotulo, "baixo": a, "alto": b})
    for rotulo, fator in [("Latência do sistema (×0 a ×2)", None)]:
        zero = float(np.mean(1.0 - det["p_prazo_lat_zero"]))
        dobro = bruto.copy()
        dobro[col_lat] = dobro[col_lat] * 2
        sens.append({"fator": rotulo, "baixo": zero, "alto": taxa_atraso(avaliar_quadros(dobro, p))})
    res["sensibilidade"] = {"base": base, "fatores": sens}

    # ---------------- E. Distância mínima de detecção ----------------
    lat_all = df[col_lat].to_numpy(dtype=float) / 1000.0
    res["d_minima_m"] = {
        "latencia_zero": d_minima([0.0], p.com(t_alerta_s=0.0)),
        "piloto_empirica": d_minima(lat_all, p),
        "piloto_p99_constante": d_minima([np.quantile(lat_all, 0.99)], p),
        "piloto_x2": d_minima(2 * lat_all, p),
    }

    # ---------------- Saídas ----------------
    df.to_csv(f"{prefixo}_dataset.csv", index=False)
    with open(f"{prefixo}_resumo.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)
    _figura(df, res, p, lat_all, prefixo)
    _imprimir(res)
    return df, res


# ---------------------------------------------------------------------------
def _figura(df, res, p, lat_all, prefixo):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": CINZA, "axes.labelcolor": TINTA,
                         "xtick.color": TINTA, "ytick.color": TINTA, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                         "grid.linewidth": 0.6, "lines.linewidth": 2})
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.2))
    from matplotlib.ticker import FuncFormatter
    virgula = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "\u2212"))
    for eixo in ax.flat:
        eixo.xaxis.set_major_formatter(virgula)
        eixo.yaxis.set_major_formatter(virgula)

    # (a) QoE-Safety × distância
    a = ax[0, 0]
    d = np.linspace(0.5, 6.0, 221)
    curvas = [
        ("Latência zero (limite)", [0.0], p.com(t_alerta_s=0.0), CINZA, "--"),
        ("Piloto (distribuição empírica)", lat_all, p, AZUL, "-"),
        ("Piloto, p99 constante", [np.quantile(lat_all, 0.99)], p, LARANJA, "-"),
        ("Piloto com latência ×2", 2 * lat_all, p, AQUA, "-"),
    ]
    for rot, lat, pp, cor, ls in curvas:
        y = [qoe_safety(float(prazo_fisico(x, pp)), lat, pp) for x in d]
        a.plot(d, y, color=cor, ls=ls, label=rot)
    dist = df.loc[df["detectado"], "dist_m"].clip(upper=6.0)
    a.plot(dist, np.full(len(dist), -0.05), "|", color=TINTA, ms=7, alpha=0.35)
    a.text(6.0, -0.05, "  distâncias estimadas\n  no piloto", va="center", fontsize=7, color=TINTA)
    a.axhline(Q_ALVO, color=CINZA, lw=0.8, ls=":")
    a.set_xlim(0.5, 6.0)
    a.set_ylim(-0.1, 1.03)
    a.set_xlabel("Distância ao obstáculo na captura, d (m)")
    a.set_ylabel("QoE-Safety = P(alerta no prazo)")
    dm = res["d_minima_m"]
    a.set_title(f"a  Latência do piloto exige detectar {br(dm['piloto_empirica'] - dm['latencia_zero'])} m mais longe",
                loc="left", fontsize=9.5)
    a.legend(fontsize=7.5, frameon=False, loc="center right")

    # (b) Decomposição do atraso
    b = ax[0, 1]
    grupos = [k for k in ["sessao_A", "sessao_B", "total"] if k in res["desfecho"]]
    nomes = {"sessao_A": "Sessão A\n(praça)", "sessao_B": "Sessão B\n(residência)", "total": "Total"}
    x = np.arange(len(grupos))
    for off, chave, cor, rot in [(-0.18, "taxa_atraso", AZUL, "Latência observada"),
                                 (0.18, "taxa_atraso_lat_zero", CINZA, "Latência zero")]:
        val = np.array([res["desfecho"][g][chave] for g in grupos]) * 100
        ic = np.array([res["desfecho"][g][chave + "_ic95"] for g in grupos]) * 100
        b.bar(x + off, val, width=0.34, color=cor, label=rot)
        b.errorbar(x + off, val, yerr=[val - ic[:, 0], ic[:, 1] - val], fmt="none",
                   ecolor=TINTA, elinewidth=1, capsize=3)
        for xi, vi, top in zip(x + off, val, ic[:, 1]):
            b.text(xi, top + 1.0, br(vi, 1), ha="center", fontsize=7.5, color=TINTA)
    b.set_xticks(x, [f"{nomes[g]}\nn = {res['desfecho'][g]['n']}" for g in grupos])
    b.set_ylabel("Alertas fora do prazo, esperado (%)")
    tt = res["desfecho"]["total"]
    b.set_title(f"b  {tt['taxa_atraso_lat_zero'] / tt['taxa_atraso'] * 100:.0f}% do atraso esperado persiste com latência zero",
                loc="left", fontsize=9.5)
    b.legend(fontsize=7.5, frameon=False)
    b.grid(axis="x", visible=False)

    # (c) Calibração: BSS por tamanho de janela
    c = ax[1, 0]
    pj = res["calibracao"]["por_janela"]
    ws = [int(k) for k in pj]
    bss = np.array([pj[str(w)]["bss"] for w in ws])
    ic = np.array([pj[str(w)]["bss_ic95"] for w in ws])
    xs = np.arange(len(ws))
    c.axhline(0, color=TINTA, lw=0.8)
    c.errorbar(xs, bss, yerr=[bss - ic[:, 0], ic[:, 1] - bss], fmt="o", color=AZUL,
               ecolor=AZUL, elinewidth=1.5, capsize=3, ms=6)
    for xi, vi in zip(xs, bss):
        c.text(xi + 0.12, vi - 0.012, br(abs(vi) if abs(vi) < 5e-4 else vi, 3), va="center",
               fontsize=7.5, color=TINTA)
    c.set_xticks(xs, [str(w) for w in ws])
    c.set_xlabel("Tamanho da janela de latências passadas (quadros)")
    c.set_ylabel("Brier skill score vs. climatologia")
    c.text(len(ws) - 1, -0.19, "BSS > 0: janela melhor que a climatologia\nBSS < 0: janela pior",
           ha="right", va="bottom", fontsize=7.5, color=TINTA)
    c.set_title(f"c  Nenhuma janela prevê a cauda (> {int(TAU_S*1000)} ms) melhor que a climatologia",
                loc="left", fontsize=9.5)
    c.grid(axis="x", visible=False)

    # (d) Sensibilidade
    dd = ax[1, 1]
    base = res["sensibilidade"]["base"] * 100
    fat = sorted(res["sensibilidade"]["fatores"],
                 key=lambda f: abs(f["alto"] - f["baixo"]))
    for i, f in enumerate(fat):
        lo, hi = f["baixo"] * 100, f["alto"] * 100
        dd.plot([lo, hi], [i, i], color=CINZA, lw=6, solid_capstyle="round", alpha=0.45)
        dd.plot(lo, i, "o", color=AZUL, ms=6)
        dd.plot(hi, i, "o", color=LARANJA, ms=6)
    dd.axvline(base, color=TINTA, lw=0.8, ls=":")
    dd.set_yticks(range(len(fat)), [f["fator"] for f in fat], fontsize=7.5)
    dd.set_xlabel("Alertas fora do prazo, esperado (%)")
    dd.set_title("d  Sensibilidade um-fator-por-vez da taxa de atraso", loc="left", fontsize=9.5)
    dd.plot([], [], "o", color=AZUL, label="Valor baixo do fator")
    dd.plot([], [], "o", color=LARANJA, label="Valor alto do fator")
    dd.legend(fontsize=7.5, frameon=False, loc="lower right")
    dd.grid(axis="y", visible=False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{prefixo}_graficos.{ext}", dpi=200)
    plt.close(fig)


def _imprimir(res):
    t = res["desfecho"]["total"]
    print("=" * 62)
    print(f"QoE-Safety v2 — {res['n_quadros']} quadros, {res['n_detectados']} com obstáculo")
    print("=" * 62)
    print("Distância estimada (quartis, m): "
          + ", ".join(f"{x:.2f}" for x in res["dist_m_quartis"]))
    print(f"Alertas fora do prazo (esperado): {t['taxa_atraso']*100:.1f}% "
          f"[IC95 {t['taxa_atraso_ic95'][0]*100:.1f}–{t['taxa_atraso_ic95'][1]*100:.1f}]")
    print(f"  com latência zero:              {t['taxa_atraso_lat_zero']*100:.1f}%")
    print(f"  parcela do sistema:             {t['parcela_sistema']*100:.1f} p.p. "
          f"[IC95 {t['parcela_sistema_ic95'][0]*100:.1f}–{t['parcela_sistema_ic95'][1]*100:.1f}]")
    print(f"Calibração P(lat > {TAU_S*1000:.0f} ms), BSS da janela vs climatologia:")
    for w, r in res["calibracao"]["por_janela"].items():
        print(f"  janela {w:>3}: BSS {r['bss']:+.3f} [IC95 {r['bss_ic95'][0]:+.3f} a {r['bss_ic95'][1]:+.3f}]"
              f"  cobertura q90 {r['cobertura_q90']*100:.1f}%")
    cf = res["comparacao_fuzzy"]
    print(f"Spearman fuzzy × v2: todos {cf['spearman_todos']:.3f}; "
          f"com obstáculo {cf['spearman_detectados']:.3f}")
    print(f"Decisões fuzzy (%): {cf['decisoes_fuzzy_pct']}")
    print(f"Decisões v2    (%): {cf['decisoes_v2_pct']}")
    print("d* para QoE-Safety ≥ 0,95 (m): "
          + ", ".join(f"{k} {v:.2f}" for k, v in res["d_minima_m"].items()))
    print("Sensibilidade (baixo → alto, % atrasado):")
    for f in res["sensibilidade"]["fatores"]:
        print(f"  {f['fator']:<34} {f['baixo']*100:5.1f} → {f['alto']*100:5.1f}")
    print("=" * 62)


if __name__ == "__main__":
    try:  # console do Windows redirecionado para arquivo pode não aceitar "→"
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    analisar(sys.argv[1] if len(sys.argv) > 1 else "resultado_dataset.csv")
