"""
analise_mcdm.py
---------------
Analises da formulacao V4 do TOPSIS ("Saida A" da revisao do artigo da
disciplina de Decisao Multicriterio).

O que o script faz, em ordem:
  1. Carrega o piloto (resultado_dataset.csv) e a nova coleta (CSVs do app),
     calcula o jitter dentro de cada sessao e as decisoes do fuzzy
     especialista (Mamdani, fuzzy_system.py).
  2. Confere se a V1-V3 reproduzem a Tabela 16 do artigo.
  3. AHP 4x4 da V4: pesos (autovetor), lambda_max, CI e CR.
  4. Compara Fuzzy, V1, V2, V3, V4, SAW com a normalizacao da V4, regra de
     duas linhas e politicas fixas (sempre local / hibrido / borda).
  5. V4 por sessao e intervalos de 95% por bootstrap de blocos moveis.
  6. Mapa do espaco de pesos da V4 (amostras uniformes do simplex):
     onde vale a propriedade P1 = "nunca EDGE com C >= 0,6" na grade de
     estados, concordancia e fracao de HIBRIDO.
  7. Sensibilidade da V4 a um parametro por vez e aos julgamentos do AHP.
  8. Teste de inversao de ordem (retirar o HIBRIDO).
  9. Decisao causal: criticidade e latencia do quadro anterior.
 10. Figuras: mapa dos pesos e plano criticidade x jitter.

Uso (no terminal, dentro de codigo_python):
    python analise_mcdm.py
    python analise_mcdm.py --piloto resultado_dataset.csv \
        --nova ../RESULTADOS_SMARTPHONE --saida resultados_mcdm
Opcoes:
    --amostras-pesos N   amostras do simplex (padrao 4000)
    --bootstrap N        reamostragens (padrao 2000)
    --refazer-fuzzy      ignora o cache das decisoes fuzzy

Semente fixa: 42. Parametros nao medidos (rho, enlace ate a borda,
inferencia remota, julgamentos 4-6 do AHP) estao em topsis.py.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import skfuzzy.control as ctrl

import topsis as T
from parser_logcat import load_dataset_from_csv
from fuzzy_system import build_fuzzy_system, compute_risk, decide

SEED = 42
JANELA_JITTER = 10
LIMIAR_ALTA = 0.6
LIMIAR_JITTER_REGRA = 40.0   # inicio de "instavel" no especialista (ms)
ALT = np.array(T.ALTERNATIVAS)
CORES = {"LOCAL": "#D55E00", "HIBRIDO": "#E69F00", "EDGE": "#0072B2"}  # Okabe-Ito
# Figuras dimensionadas para a largura do texto do artigo (fonte de 7-8 pt)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 6.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "pdf.fonttype": 42})
ROTULOS = {"LOCAL": "Local", "HIBRIDO": "Híbrido", "EDGE": "Borda"}

# Valores publicados no artigo (Tabela 16): L, H, B (%), concordancia (%)
REF_ARTIGO = {
    ("piloto", "Fuzzy"): (42.0, 54.3, 3.7, None),
    ("piloto", "V1"): (0.0, 0.0, 100.0, 3.7),
    ("piloto", "V2"): (99.9, 0.0, 0.1, 42.0),
    ("piloto", "V3"): (47.1, 25.7, 27.3, 56.0),
    ("nova", "Fuzzy"): (22.5, 72.2, 5.3, None),
    ("nova", "V1"): (0.0, 0.0, 100.0, 5.3),
    ("nova", "V2"): (100.0, 0.0, 0.0, 22.5),
    ("nova", "V3"): (38.4, 16.4, 45.2, 32.8),
}


# =====================================================================
# 1. Dados
# =====================================================================
def jitter_por_sessao(df):
    return (df.groupby("session", sort=False)["rtt_ms"]
              .transform(lambda s: s.rolling(JANELA_JITTER, min_periods=1).std())
              .fillna(0.0))


def colunas_causais(df):
    """Criticidade e latencia do quadro anterior, dentro da sessao.
    1o quadro da sessao: criticidade 0 e a propria latencia."""
    g = df.groupby("session", sort=False)
    df["crit_ant"] = g["criticality"].shift(1).fillna(0.0)
    df["lat_ant"] = g["total_latency_ms"].shift(1)
    df["lat_ant"] = df["lat_ant"].fillna(df["total_latency_ms"])
    return df


def riscos_fuzzy(lat, jit, crit):
    sistema, _ = build_fuzzy_system()
    sim = ctrl.ControlSystemSimulation(sistema)
    r = np.empty(len(lat))
    for i, (l, j, c) in enumerate(zip(lat, jit, crit)):
        r[i] = compute_risk(sim, l, j, c)
    return r


def adicionar_fuzzy(df, pasta_cache, nome, refazer):
    """Decisoes fuzzy (atual e causal), com cache em CSV."""
    cache = pasta_cache / f"cache_fuzzy_{nome}.csv"
    if cache.exists() and not refazer:
        c = pd.read_csv(cache)
        if len(c) == len(df) and (c["frame_id"].to_numpy() == df["frame_id"].to_numpy()).all():
            for col in ("risco_fuzzy", "risco_fuzzy_causal"):
                df[col] = c[col].to_numpy()
            df["decisao_fuzzy"] = [decide(r) for r in df["risco_fuzzy"]]
            df["decisao_fuzzy_causal"] = [decide(r) for r in df["risco_fuzzy_causal"]]
            print(f"    fuzzy de '{nome}' lido do cache.")
            return df
    t0 = time.time()
    df["risco_fuzzy"] = riscos_fuzzy(df["total_latency_ms"].to_numpy(),
                                     df["rtt_jitter_ms"].to_numpy(),
                                     df["criticality"].to_numpy())
    df["risco_fuzzy_causal"] = riscos_fuzzy(df["lat_ant"].to_numpy(),
                                            df["rtt_jitter_ms"].to_numpy(),
                                            df["crit_ant"].to_numpy())
    df["decisao_fuzzy"] = [decide(r) for r in df["risco_fuzzy"]]
    df["decisao_fuzzy_causal"] = [decide(r) for r in df["risco_fuzzy_causal"]]
    df[["frame_id", "session", "risco_fuzzy", "risco_fuzzy_causal"]].to_csv(cache, index=False)
    print(f"    fuzzy de '{nome}' calculado em {time.time() - t0:.0f} s.")
    return df


def carregar_piloto(path, pasta_cache, refazer):
    df = pd.read_csv(path)
    # Versao publicada (jitter com a janela atravessando as duas sessoes)
    df["rtt_jitter_publicado"] = df["rtt_jitter_ms"]
    df["decisao_fuzzy_publicado"] = df["decisao_fuzzy"]
    # Versao corrigida: jitter dentro de cada sessao
    df["rtt_jitter_ms"] = jitter_por_sessao(df)
    df = colunas_causais(df)
    return adicionar_fuzzy(df, pasta_cache, "piloto", refazer)


def carregar_nova(pasta, pasta_cache, refazer):
    arquivos = sorted(Path(pasta).glob("iavision_*.csv"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum iavision_*.csv em '{pasta}'.")
    dfs = []
    for a in arquivos:
        d = load_dataset_from_csv(str(a))
        d["session"] = (d["scenario"].iloc[0] if "scenario" in d.columns
                        and len(d) else a.stem)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df["criticality"] = df["proximity"].fillna(0.0).clip(0.0, 1.0)
    df["rtt_jitter_ms"] = jitter_por_sessao(df)
    df = colunas_causais(df)
    return adicionar_fuzzy(df, pasta_cache, "nova", refazer)


# =====================================================================
# 2. Decisoes e metricas
# =====================================================================
def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in ALT)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def latencia_alerta(df, dec, tau=T.REMOTE_INFERENCE_MS, rtt_extra=0.0):
    """Latencia estimada do alerta com a semantica da V4 para todas as
    politicas: LOCAL = L, EDGE = RTT + extra + tau, HIBRIDO = min(L, EDGE)."""
    L = df["total_latency_ms"].to_numpy(float)
    B = df["rtt_ms"].to_numpy(float) + rtt_extra + tau
    return np.select([dec == "LOCAL", dec == "HIBRIDO"], [L, np.minimum(L, B)], B)


def resumo(df, dec, ref_col="decisao_fuzzy", rtt_extra=0.0):
    dec = np.asarray(dec)
    ref = df[ref_col].to_numpy()
    C = df["criticality"].to_numpy()
    alta, obst = C >= LIMIAR_ALTA, C > 0
    lat = latencia_alerta(df, dec, rtt_extra=rtt_extra)
    pct = lambda m: float(100 * m.mean()) if m.size else float("nan")
    return {
        "n": int(len(df)),
        "LOCAL_%": pct(dec == "LOCAL"),
        "HIBRIDO_%": pct(dec == "HIBRIDO"),
        "EDGE_%": pct(dec == "EDGE"),
        "concord_%": pct(dec == ref),
        "kappa": kappa(dec, ref),
        "alta_n": int(alta.sum()),
        "alta_EDGE_%": pct(dec[alta] == "EDGE"),
        "alta_LOCAL_%": pct(dec[alta] == "LOCAL"),
        "alta_HIBRIDO_%": pct(dec[alta] == "HIBRIDO"),
        "obst_EDGE_%": pct(dec[obst] == "EDGE"),
        "lat_mediana_ms": float(np.median(lat)),
        "lat_p95_ms": float(np.percentile(lat, 95)),
        "lat_alta_mediana_ms": float(np.median(lat[alta])) if alta.any() else float("nan"),
    }


def dec_v4(df, w=None, rho=T.ENERGY_RATIO, tau=T.REMOTE_INFERENCE_MS,
           rtt_extra=T.RTT_EDGE_EXTRA_MS, sJ=400.0, sL=800.0,
           hib=T.HIBRIDO_LATENCIA, crit_col="criticality", lat_col=None):
    w = T.WEIGHTS_AHP_V4 if w is None else np.asarray(w, float)
    m = T.build_matrix(df, "V4", remote_inference_ms=tau, rho=rho,
                       rtt_extra_ms=rtt_extra, hibrido_latencia=hib,
                       crit_col=crit_col, lat_col=lat_col)
    escalas = np.array([1.0, sJ, sL, 1.0])
    cc = T.topsis_cc(m, w, "V4", escalas)
    return ALT[cc.argmax(axis=1)]


def dec_saw_v4(df, w=None):
    """Soma ponderada (SAW) com a mesma matriz e normalizacao da V4."""
    w = T.WEIGHTS_AHP_V4 if w is None else np.asarray(w, float)
    m = T.build_matrix(df, "V4")
    r = np.clip(m / T.REF_SCALES_V4, 0.0, 1.0)
    return ALT[(r * w).sum(axis=2).argmin(axis=1)]


def dec_regra(df):
    """Regra de duas linhas: obstaculo -> LOCAL; senao, jitter decide."""
    C = df["criticality"].to_numpy()
    J = df["rtt_jitter_ms"].to_numpy()
    return np.where(C > 0, "LOCAL", np.where(J < LIMIAR_JITTER_REGRA, "EDGE", "LOCAL"))


def dec_topsis(df, form, w=None):
    w = T.WEIGHTS_AHP if w is None else w
    return T.topsis_decide(df, w, form)["decisao_topsis"].to_numpy()


def metodos(df):
    n = len(df)
    return {
        "Fuzzy": df["decisao_fuzzy"].to_numpy(),
        "V1": dec_topsis(df, "V1"),
        "V2": dec_topsis(df, "V2"),
        "V3": dec_topsis(df, "V3"),
        "V4": dec_v4(df),
        "SAW (normalização V4)": dec_saw_v4(df),
        "Regra de duas linhas": dec_regra(df),
        "Sempre local": np.full(n, "LOCAL"),
        "Sempre híbrido": np.full(n, "HIBRIDO"),
        "Sempre borda": np.full(n, "EDGE"),
    }


# =====================================================================
# 3. Bootstrap por blocos moveis (dentro de cada sessao)
# =====================================================================
def bootstrap_blocos(df, func, B=2000, bloco=100, seed=SEED):
    rng = np.random.default_rng(seed)
    s = df["session"].to_numpy()
    grupos = [np.flatnonzero(s == g) for g in pd.unique(s)]
    vals = []
    for _ in range(B):
        idx = []
        for g in grupos:
            n = len(g)
            b = min(bloco, n)
            k = int(np.ceil(n / b))
            ini = rng.integers(0, n - b + 1, size=k)
            sel = (ini[:, None] + np.arange(b)[None, :]).ravel()[:n]
            idx.append(g[sel])
        vals.append(func(np.concatenate(idx)))
    vals = np.asarray(vals)
    return np.percentile(vals, 2.5, axis=0), np.percentile(vals, 97.5, axis=0)


# =====================================================================
# 4. Grade de estados (pior caso, independente dos dados)
# =====================================================================
def grade_estados():
    C = np.round(np.arange(0.0, 1.0001, 0.01), 2)
    J = [0, 5, 10, 20, 30, 40, 60, 80, 100, 150, 200, 300, 400]
    LL = [110, 150, 190, 250, 300, 400, 500, 600, 800]
    RTT = [10, 20, 30, 45, 60, 80, 100, 150, 200, 300, 400, 600, 800]
    rede = [(j, l, r) for j in J for l in LL for r in RTT if r <= l]
    rede = np.array(rede, dtype=float)
    cc, rr = np.meshgrid(C, np.arange(len(rede)), indexing="ij")
    g = pd.DataFrame({
        "criticality": cc.ravel(),
        "rtt_jitter_ms": rede[rr.ravel(), 0],
        "total_latency_ms": rede[rr.ravel(), 1],
        "rtt_ms": rede[rr.ravel(), 2],
    })
    return g, C


def c_estrela(dec, C_grade, n_rede):
    """Menor C* tal que nenhum estado com C >= C* escolhe EDGE.
    Devolve nan se ha EDGE ate C = 1."""
    edge_por_c = (dec.reshape(len(C_grade), n_rede) == "EDGE").any(axis=1)
    if not edge_por_c.any():
        return 0.0
    ultimo = int(np.flatnonzero(edge_por_c).max())
    return float("nan") if ultimo == len(C_grade) - 1 else float(C_grade[ultimo + 1])


# =====================================================================
# 5. Figuras
# =====================================================================
def salvar(fig, pasta, nome):
    for ext in ("pdf", "png"):
        fig.savefig(pasta / f"{nome}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figura_mapa_pesos(mp, w_ahp, pasta):
    """Dimensionada para a largura do texto do modelo SBC (~16 cm)."""
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.75), layout="constrained")
    mp = mp.iloc[1:]   # a linha 0 e o proprio AHP (marcado com a estrela)
    ok = mp["P1"].to_numpy(bool)
    ax[0].scatter(mp.loc[ok, "w_C"], mp.loc[ok, "w_L"], s=2, c="#56B4E9",
                  label="P1 vale", rasterized=True)
    ax[0].scatter(mp.loc[~ok, "w_C"], mp.loc[~ok, "w_L"], s=2, c="#D55E00",
                  label="P1 falha", rasterized=True)
    ax[0].set_xlabel("Peso da criticidade $w_C$")
    ax[0].set_ylabel("Peso da latência $w_L$")
    ax[0].set_title("(a) Nunca só Borda com $C_t \\geq 0{,}6$")
    sc = ax[1].scatter(mp["w_L"], mp["w_E"], s=2, c=mp["HIBRIDO_nova_%"], cmap="viridis",
                       vmin=0, vmax=100, rasterized=True)
    fig.colorbar(sc, ax=ax[1], fraction=0.05, label="Quadros no Híbrido (%)")
    ax[1].set_xlabel("Peso da latência $w_L$")
    ax[1].set_ylabel("Peso da energia $w_E$")
    ax[1].set_title("(b) Uso do Híbrido na nova coleta")
    idx = {"w_C": 0, "w_J": 1, "w_L": 2, "w_E": 3}
    for a, (x, y) in zip(ax, [("w_C", "w_L"), ("w_L", "w_E")]):
        a.scatter(w_ahp[idx[x]], w_ahp[idx[y]], marker="*", s=90, c="black",
                  edgecolors="white", linewidths=0.6, zorder=5)
        a.annotate("AHP", (w_ahp[idx[x]], w_ahp[idx[y]]), xytext=(5, 5),
                   textcoords="offset points", fontsize=7, weight="bold")
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.set_aspect("equal")
    ax[0].legend(loc="upper right", markerscale=4, handletextpad=0.2)
    salvar(fig, pasta, "fig_mapa_pesos_v4")


def figura_plano(pasta, L=190.0, RTT=45.0):
    raiz = np.linspace(0, 1, 81)
    C = raiz ** 2
    J = np.linspace(0, 200, 61)
    cc, jj = np.meshgrid(C, J, indexing="xy")
    g = pd.DataFrame({"criticality": cc.ravel(), "rtt_jitter_ms": jj.ravel(),
                      "total_latency_ms": L, "rtt_ms": RTT})
    paineis = {
        "(a) Fuzzy especialista": np.array([decide(r) for r in riscos_fuzzy(
            g["total_latency_ms"].to_numpy(), g["rtt_jitter_ms"].to_numpy(),
            g["criticality"].to_numpy())]),
        "(b) TOPSIS V3": dec_topsis(g, "V3"),
        "(c) TOPSIS V4, $\\rho = 1$": dec_v4(g, rho=1.0),
        "(d) TOPSIS V4, $\\rho = 0{,}25$": dec_v4(g, rho=0.25),
    }
    cmap = ListedColormap([CORES[a] for a in ALT])
    codigo = {a: i for i, a in enumerate(ALT)}
    fig, ax = plt.subplots(2, 2, figsize=(6.3, 4.6), sharex=True, sharey=True,
                           layout="constrained")
    for k, (a, (titulo, dec)) in enumerate(zip(ax.ravel(), paineis.items())):
        z = np.vectorize(codigo.get)(dec).reshape(cc.shape)
        a.pcolormesh(raiz, J, z, cmap=cmap, vmin=-0.5, vmax=2.5, shading="nearest",
                     rasterized=True)
        a.axvline(np.sqrt(LIMIAR_ALTA), ls="--", c="white", lw=0.8)
        ticks = [0, 0.05, 0.25, 0.6, 1]
        a.set_xticks(np.sqrt(ticks))
        a.set_xticklabels(["0", "0,05", "0,25", "0,6", "1"])
        if k >= 2:
            a.set_xlabel("Criticidade $C_t$ (eixo em raiz)")
        if k % 2 == 0:
            a.set_ylabel("Jitter $J_t$ (ms)")
        a.set_title(titulo)
    alcas = [plt.Rectangle((0, 0), 1, 1, color=CORES[a]) for a in ALT]
    fig.legend(alcas, [ROTULOS[a] for a in ALT], loc="outside lower center", ncol=3,
               frameon=False)
    salvar(fig, pasta, "fig_plano_decisao")


# =====================================================================
# 6. Chegada de obstaculos (decisao causal) em todas as configuracoes
# =====================================================================
def analise_chegada(d, cfgs, pesos, decs_atual, pasta, B):
    """Relaciona, para cada politica com decisao causal:
      x = % de quadros enviados so a borda quando o quadro anterior nao
          tinha obstaculo (C_{t-1} = 0);
      y = % dos primeiros quadros criticos (C_t >= 0,6 e C_{t-1} < 0,6)
          enviados so a borda.
    Se a chegada do obstaculo nao depende do estado da rede, y ~ x."""
    C = d["criticality"].to_numpy()
    c_ant = d["crit_ant"].to_numpy()
    sem_obst = c_ant == 0
    prim = (C >= LIMIAR_ALTA) & (c_ant < LIMIAR_ALTA)
    m_caus = T.build_matrix(d, "V4", crit_col="crit_ant", lat_col="lat_ant")
    reg = []

    def registra(rot, tipo, dv):
        reg.append({"configuracao": rot, "tipo": tipo,
                    "x_EDGE_sem_obstaculo_anterior_%": 100 * (dv[sem_obst] == "EDGE").mean(),
                    "y_EDGE_primeiros_criticos_%": 100 * (dv[prim] == "EDGE").mean(),
                    "EDGE_alta_%": 100 * (dv[C >= LIMIAR_ALTA] == "EDGE").mean()})

    registra("Fuzzy especialista", "especialista", d["decisao_fuzzy_causal"].to_numpy())
    registra("TOPSIS V3", "V3", T.topsis_decide(d, T.WEIGHTS_AHP, "V3", crit_col="crit_ant",
                                                lat_col="lat_ant")["decisao_topsis"].to_numpy())
    dregra = d.assign(criticality=d["crit_ant"])
    registra("Regra de duas linhas", "regra", dec_regra(dregra))
    for rotulo, kw, wk, _ in cfgs:
        registra(f"V4: {rotulo}", "V4 sensibilidade",
                 dec_v4(d, w=wk, crit_col="crit_ant", lat_col="lat_ant", **kw))
    for k, wk in enumerate(pesos[1:]):
        registra(f"V4: pesos amostra {k + 1}", "V4 pesos",
                 ALT[T.topsis_cc(m_caus, wk, "V4").argmax(axis=1)])
    tab = pd.DataFrame(reg)
    tab.round(3).to_csv(pasta / "chegada_obstaculos.csv", index=False)

    # IC 95% (bootstrap de blocos) para V4 base e especialista
    ic = {}
    for rot, dv in (("V4 base", dec_v4(d, crit_col="crit_ant", lat_col="lat_ant")),
                    ("Fuzzy especialista", d["decisao_fuzzy_causal"].to_numpy())):
        f = lambda idx, dv=dv: np.array([
            100 * (dv[idx][prim[idx]] == "EDGE").mean() if prim[idx].any() else np.nan])
        rng = np.random.default_rng(SEED)
        s = d["session"].to_numpy()
        grupos = [np.flatnonzero(s == g) for g in pd.unique(s)]
        vals = []
        for _ in range(B):
            idx = []
            for g in grupos:
                n, b = len(g), min(100, len(g))
                ini = rng.integers(0, n - b + 1, size=int(np.ceil(n / b)))
                idx.append(g[(ini[:, None] + np.arange(b)[None, :]).ravel()[:n]])
            vals.append(f(np.concatenate(idx))[0])
        ic[rot] = [round(float(np.nanpercentile(vals, 2.5)), 1),
                   round(float(np.nanpercentile(vals, 97.5)), 1)]

    v4 = tab[tab["tipo"].str.startswith("V4")]
    x, y = v4["x_EDGE_sem_obstaculo_anterior_%"], v4["y_EDGE_primeiros_criticos_%"]
    res = {"eventos_primeiro_critico": int(prim.sum()),
           "configuracoes_V4": int(len(v4)),
           "correlacao_x_y_V4": round(float(np.corrcoef(x, y)[0, 1]), 3),
           "media_abs_y_menos_x_pp": round(float(np.mean(np.abs(y - x))), 1),
           "ic95_y_primeiros_EDGE_%": ic,
           "especialista": tab.iloc[0, 2:].astype(float).round(1).to_dict(),
           "V3": tab.iloc[1, 2:].astype(float).round(1).to_dict(),
           "regra": tab.iloc[2, 2:].astype(float).round(1).to_dict(),
           "V4_base": tab[tab["configuracao"] == "V4: base"].iloc[0, 2:]
                      .astype(float).round(1).to_dict()}

    fig, ax = plt.subplots(figsize=(3.6, 3.4), layout="constrained")
    pw = tab[tab["tipo"] == "V4 pesos"]
    ax.scatter(pw.iloc[:, 2], pw.iloc[:, 3], s=1.5, c="#BBBBBB", rasterized=True,
               label="V4, pesos amostrados")
    ps = tab[tab["tipo"] == "V4 sensibilidade"]
    ax.scatter(ps.iloc[:, 2], ps.iloc[:, 3], s=12, c="#0072B2", edgecolors="white",
               linewidths=0.4, label="V4, parâmetros e julgamentos")
    for tipo, mk, cor, rot in (("V3", "^", "#E69F00", "V3"),
                               ("regra", "s", "#009E73", "Regra de duas linhas"),
                               ("especialista", "*", "black", "Fuzzy especialista")):
        r = tab[tab["tipo"] == tipo].iloc[0]
        ax.scatter(r.iloc[2], r.iloc[3], marker=mk, s=70 if mk == "*" else 30, c=cor,
                   edgecolors="white", linewidths=0.5, zorder=5, label=rot)
    ax.plot([0, 100], [0, 100], ls="--", c="#666666", lw=0.8, label="$y = x$")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_xlabel("Só Borda quando o quadro anterior\nnão tinha obstáculo (%)")
    ax.set_ylabel("Chegadas de obstáculo só na Borda (%)")
    ax.legend(loc="upper left", markerscale=1.5, handletextpad=0.2, borderaxespad=0.3)
    salvar(fig, pasta, "fig_chegada_obstaculos")
    return res


# =====================================================================
# Principal
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--piloto", default="resultado_dataset.csv")
    ap.add_argument("--nova", default=str(Path("..") / "RESULTADOS_SMARTPHONE"))
    ap.add_argument("--saida", default="resultados_mcdm")
    ap.add_argument("--amostras-pesos", type=int, default=4000)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--refazer-fuzzy", action="store_true")
    args = ap.parse_args()

    out = Path(args.saida)
    out.mkdir(parents=True, exist_ok=True)
    t_ini = time.time()
    resumo_json = {"semente": SEED}

    # ---------------- 1. Dados ----------------
    print("[1/11] Carregando dados e calculando o especialista...")
    piloto = carregar_piloto(args.piloto, out, args.refazer_fuzzy)
    nova = carregar_nova(args.nova, out, args.refazer_fuzzy)
    dados = {"piloto": piloto, "nova": nova}
    print(f"    piloto: {len(piloto)} quadros; nova coleta: {len(nova)} quadros, "
          f"{nova['session'].nunique()} sessões.")

    # ---------------- 2. Reproducao do artigo ----------------
    print("[2/11] Conferindo a reprodução da Tabela 16 do artigo...")
    pil_pub = piloto.assign(rtt_jitter_ms=piloto["rtt_jitter_publicado"],
                            decisao_fuzzy=piloto["decisao_fuzzy_publicado"])
    linhas = []
    for nome, d in (("piloto", pil_pub), ("nova", nova)):
        for metodo in ("Fuzzy", "V1", "V2", "V3"):
            dec = d["decisao_fuzzy"].to_numpy() if metodo == "Fuzzy" else dec_topsis(d, metodo)
            r = resumo(d, dec)
            ref = REF_ARTIGO[(nome, metodo)]
            obtido = (r["LOCAL_%"], r["HIBRIDO_%"], r["EDGE_%"],
                      None if metodo == "Fuzzy" else r["concord_%"])
            ok = all(abs(o - e) <= 0.15 for o, e in zip(obtido, ref) if e is not None)
            linhas.append({"conjunto": nome, "metodo": metodo,
                           "L_%": round(obtido[0], 1), "H_%": round(obtido[1], 1),
                           "B_%": round(obtido[2], 1),
                           "concord_%": None if obtido[3] is None else round(obtido[3], 1),
                           "artigo": ref, "confere": ok})
    rep = pd.DataFrame(linhas)
    rep.to_csv(out / "reproducao_artigo.csv", index=False)
    print(rep.to_string(index=False))
    resumo_json["reproducao_confere"] = bool(rep["confere"].all())

    # ---------------- 3. AHP 4x4 ----------------
    print("[3/11] AHP 4x4 da V4...")
    A = T.matriz_ahp(T.AHP_JULGAMENTOS_V4)
    w, lam, ci, cr = T.pesos_ahp(A)
    w_media = (A / A.sum(axis=0)).mean(axis=1)
    ahp = {"criterios": list(T.CRITERIA_V4), "matriz": A.round(4).tolist(),
           "julgamentos": {f"{i}/{j}": v for (i, j), v in T.AHP_JULGAMENTOS_V4.items()},
           "pesos_autovetor": w.round(4).tolist(),
           "pesos_media_linhas": w_media.round(4).tolist(),
           "lambda_max": round(lam, 4), "CI": round(ci, 4), "CR": round(cr, 4),
           "RI": T.RI_SAATY[4]}
    (out / "ahp_v4.json").write_text(json.dumps(ahp, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"    pesos (C, J, L, E) = {w.round(4)}; lambda_max = {lam:.4f}; CR = {cr:.4f}")
    resumo_json["ahp_v4"] = ahp

    # ---------------- 4. Metodos ----------------
    print("[4/11] Comparando métodos...")
    tab = []
    decs = {}
    for nome, d in dados.items():
        decs[nome] = metodos(d)
        for metodo, dec in decs[nome].items():
            r = resumo(d, dec)
            r.update({"conjunto": nome, "metodo": metodo,
                      "igual_a_V4_%": float(100 * (dec == decs[nome]["V4"]).mean())})
            tab.append(r)
    tab = pd.DataFrame(tab)
    cols = ["conjunto", "metodo"] + [c for c in tab.columns if c not in ("conjunto", "metodo")]
    tab = tab[cols]
    tab.round(2).to_csv(out / "metodos.csv", index=False)
    print(tab[["conjunto", "metodo", "LOCAL_%", "HIBRIDO_%", "EDGE_%", "concord_%",
               "kappa", "alta_EDGE_%", "obst_EDGE_%", "igual_a_V4_%"]]
          .round(1).to_string(index=False))

    # ---------------- 5. Por sessao e bootstrap ----------------
    print("[5/11] V4 por sessão e bootstrap por blocos...")
    sess = []
    for nome, d in dados.items():
        dv4 = decs[nome]["V4"]
        for s in pd.unique(d["session"]):
            m = (d["session"] == s).to_numpy()
            r = resumo(d[m], dv4[m])
            r.update({"conjunto": nome, "sessao": s})
            sess.append(r)
    sess = pd.DataFrame(sess)
    sess.round(2).to_csv(out / "v4_por_sessao.csv", index=False)
    sn = sess[sess["conjunto"] == "nova"]
    resumo_json["v4_nova_concord_sessoes_media_dp"] = [
        round(sn["concord_%"].mean(), 1), round(sn["concord_%"].std(ddof=1), 1)]
    resumo_json["v4_nova_kappa_sessoes_media_dp"] = [
        round(sn["kappa"].mean(), 2), round(sn["kappa"].std(ddof=1), 2)]

    ic = {}
    for nome, d in dados.items():
        dv4, ref = decs[nome]["V4"], d["decisao_fuzzy"].to_numpy()
        f = lambda idx: np.array([100 * (dv4[idx] == ref[idx]).mean(),
                                  100 * (dv4[idx] == "EDGE").mean(),
                                  100 * (dv4[idx] == "HIBRIDO").mean()])
        lo, hi = bootstrap_blocos(d, f, B=args.bootstrap)
        ic[nome] = {"concord_%": [round(lo[0], 1), round(hi[0], 1)],
                    "EDGE_%": [round(lo[1], 1), round(hi[1], 1)],
                    "HIBRIDO_%": [round(lo[2], 1), round(hi[2], 1)]}
    resumo_json["v4_ic95_bootstrap_blocos"] = ic
    print(f"    IC 95%: {ic}")

    # ---------------- 6. Mapa do espaco de pesos ----------------
    print(f"[6/10] Mapa do espaço de pesos ({args.amostras_pesos} amostras)...")
    grade, C_grade = grade_estados()
    n_rede = len(grade) // len(C_grade)
    m_grade = T.build_matrix(grade, "V4")
    m_nova = T.build_matrix(nova, "V4")
    ref_nova = nova["decisao_fuzzy"].to_numpy()
    rng = np.random.default_rng(SEED)
    pesos = np.vstack([T.WEIGHTS_AHP_V4, rng.dirichlet(np.ones(4), size=args.amostras_pesos)])
    reg = []
    for k, wk in enumerate(pesos):
        dg = ALT[T.topsis_cc(m_grade, wk, "V4").argmax(axis=1)]
        cst = c_estrela(dg, C_grade, n_rede)
        dn = ALT[T.topsis_cc(m_nova, wk, "V4").argmax(axis=1)]
        reg.append({"amostra": k, "w_C": wk[0], "w_J": wk[1], "w_L": wk[2], "w_E": wk[3],
                    "C_estrela": cst,
                    "P1": bool(np.isfinite(cst) and cst <= LIMIAR_ALTA + 1e-9),
                    "concord_nova_%": 100 * (dn == ref_nova).mean(),
                    "EDGE_nova_%": 100 * (dn == "EDGE").mean(),
                    "HIBRIDO_nova_%": 100 * (dn == "HIBRIDO").mean()})
    mp = pd.DataFrame(reg)
    mp.round(4).to_csv(out / "mapa_pesos_v4.csv", index=False)
    ahp_row, amostras = mp.iloc[0], mp.iloc[1:]
    falha = amostras[~amostras["P1"]]
    dist = (np.sqrt(((falha[["w_C", "w_J", "w_L", "w_E"]].to_numpy()
                      - T.WEIGHTS_AHP_V4) ** 2).sum(axis=1)).min()
            if len(falha) else float("nan"))
    resumo_json["mapa_pesos"] = {
        "amostras": int(len(amostras)),
        "P1_vale_no_AHP": bool(ahp_row["P1"]),
        "C_estrela_AHP": ahp_row["C_estrela"],
        "fracao_simplex_P1_%": round(100 * amostras["P1"].mean(), 1),
        # P1 vale em todas as amostras com w_C acima deste valor
        "w_C_acima_do_qual_P1_sempre_vale": round(falha["w_C"].max(), 3) if len(falha) else 0.0,
        # O HIBRIDO so passa de 20% dos quadros com w_E abaixo deste valor
        "HIBRIDO_acima_de_20%_so_com_w_E_ate": round(
            amostras.loc[amostras["HIBRIDO_nova_%"] > 20, "w_E"].max(), 3)
        if (amostras["HIBRIDO_nova_%"] > 20).any() else None,
        "concord_nova_maxima_%": round(amostras["concord_nova_%"].max(), 1),
        "pesos_da_concord_maxima": np.round(amostras.loc[
            amostras["concord_nova_%"].idxmax(), ["w_C", "w_J", "w_L", "w_E"]]
            .to_numpy(dtype=float), 3).tolist(),
        "distancia_AHP_ate_falha_mais_proxima": round(float(dist), 3),
        "concord_nova_simplex_p5_p50_p95": np.percentile(
            amostras["concord_nova_%"], [5, 50, 95]).round(1).tolist(),
    }
    print(f"    {resumo_json['mapa_pesos']}")
    figura_mapa_pesos(mp, T.WEIGHTS_AHP_V4, out)

    # ---------------- 7. Sensibilidade ----------------
    print("[7/11] Sensibilidade da V4...")
    cfgs = [("base", {}, None, None)]
    cfgs += [(f"rho = {v}", {"rho": v}, None, None) for v in (0.25, 0.5, 2.0, 4.0)]
    cfgs += [(f"inferência remota {v:.0f} ms", {"tau": v}, None, None) for v in (0.0, 50.0, 100.0)]
    cfgs += [(f"enlace até a borda +{v:.0f} ms", {"rtt_extra": v}, None, None) for v in (50.0, 100.0, 200.0)]
    cfgs += [(f"escala do jitter {v:.0f} ms", {"sJ": v}, None, None) for v in (200.0, 800.0)]
    cfgs += [(f"escala da latência {v:.0f} ms", {"sL": v}, None, None) for v in (400.0, 1600.0)]
    cfgs += [("Híbrido: só o local alerta", {"hib": "local"}, None, None)]
    cfgs += [("pesos iguais (1/4)", {}, np.full(4, 0.25), None)]
    for par, v in T.AHP_JULGAMENTOS_V4.items():
        for nv in (v - 1, v + 1):
            if 1 <= nv <= 9:
                j = dict(T.AHP_JULGAMENTOS_V4)
                j[par] = nv
                wj, _, _, crj = T.pesos_ahp(T.matriz_ahp(j))
                cfgs.append((f"julgamento {par[0]}/{par[1]}: {v} -> {nv}", {}, wj, crj))
    sens = []
    for rotulo, kw, wk, crk in cfgs:
        linha = {"configuracao": rotulo, "CR": None if crk is None else round(crk, 3)}
        dg = dec_v4(grade, w=wk, **kw)
        linha["C_estrela_grade"] = c_estrela(dg, C_grade, n_rede)
        for nome, d in dados.items():
            dv = dec_v4(d, w=wk, **kw)
            r = resumo(d, dv, rtt_extra=kw.get("rtt_extra", 0.0))
            for c in ("concord_%", "kappa", "LOCAL_%", "HIBRIDO_%", "EDGE_%",
                      "alta_EDGE_%", "obst_EDGE_%", "lat_mediana_ms"):
                linha[f"{nome}_{c}"] = r[c]
            if nome == "nova":
                cs = [100 * (dv[(d["session"] == s).to_numpy()]
                             == d.loc[d["session"] == s, "decisao_fuzzy"].to_numpy()).mean()
                      for s in pd.unique(d["session"])]
                linha["nova_concord_sessoes_media"] = float(np.mean(cs))
                linha["nova_concord_sessoes_dp"] = float(np.std(cs, ddof=1))
        sens.append(linha)
    sens = pd.DataFrame(sens)
    sens.round(3).to_csv(out / "sensibilidade_v4.csv", index=False)
    print(sens[["configuracao", "CR", "C_estrela_grade", "nova_concord_%",
                "nova_LOCAL_%", "nova_HIBRIDO_%", "nova_EDGE_%", "nova_alta_EDGE_%",
                "piloto_concord_%"]].round(1).to_string(index=False))

    # ---------------- 8. Inversao de ordem ----------------
    print("[8/11] Teste de inversão de ordem (retirando o Híbrido)...")
    inv = []
    for nome, d in dados.items():
        for form, wk in (("V1", T.WEIGHTS_AHP), ("V2", T.WEIGHTS_AHP),
                         ("V3", T.WEIGHTS_AHP), ("V4", T.WEIGHTS_AHP_V4)):
            m = T.build_matrix(d, form)
            cc3 = T.topsis_cc(m, wk, form)
            cc2 = T.topsis_cc(m[:, [0, 2], :], wk, form)
            s3 = np.sign(cc3[:, 0] - cc3[:, 2])
            s2 = np.sign(cc2[:, 0] - cc2[:, 1])
            valid = (s3 != 0) & (s2 != 0)
            inv.append({"conjunto": nome, "formulacao": form,
                        "inversoes_%": float(100 * (s3[valid] != s2[valid]).mean()),
                        "quadros": int(valid.sum())})
    inv = pd.DataFrame(inv)
    inv.round(2).to_csv(out / "inversao_ordem.csv", index=False)
    print(inv.round(2).to_string(index=False))

    # ---------------- 9. Decisao causal ----------------
    print("[9/11] Decisão causal (quadro anterior)...")
    caus = []
    for nome, d in dados.items():
        C = d["criticality"].to_numpy()
        alta = C >= LIMIAR_ALTA
        primeiro = alta & (d["crit_ant"].to_numpy() < LIMIAR_ALTA)
        variantes = {
            "Fuzzy (atual)": d["decisao_fuzzy"].to_numpy(),
            "Fuzzy (causal)": d["decisao_fuzzy_causal"].to_numpy(),
            "V3 (atual)": decs[nome]["V3"],
            "V3 (causal)": T.topsis_decide(d, T.WEIGHTS_AHP, "V3", crit_col="crit_ant",
                                           lat_col="lat_ant")["decisao_topsis"].to_numpy(),
            "V4 (atual)": decs[nome]["V4"],
            "V4 (causal)": dec_v4(d, crit_col="crit_ant", lat_col="lat_ant"),
        }
        for rot, dv in variantes.items():
            caus.append({"conjunto": nome, "politica": rot,
                         "quadros_alta": int(alta.sum()),
                         "alta_EDGE_%": float(100 * (dv[alta] == "EDGE").mean()),
                         "primeiros_alta": int(primeiro.sum()),
                         "primeiros_alta_EDGE_%": float(100 * (dv[primeiro] == "EDGE").mean())
                         if primeiro.any() else float("nan"),
                         "primeiros_alta_LOCAL_%": float(100 * (dv[primeiro] == "LOCAL").mean())
                         if primeiro.any() else float("nan")})
    caus = pd.DataFrame(caus)
    caus.round(2).to_csv(out / "decisao_causal.csv", index=False)
    print(caus.round(1).to_string(index=False))

    # ---------------- 10. Figura do plano ----------------
    print("[10/11] Figura do plano criticidade x jitter...")
    figura_plano(out)

    # ---------------- 11. Chegada de obstaculos em todas as configuracoes ----
    print("[11/11] Chegada de obstáculos em todas as configurações (causal)...")
    resumo_json["chegada_obstaculos"] = analise_chegada(nova, cfgs, pesos, decs["nova"],
                                                        out, args.bootstrap)
    print(f"    {resumo_json['chegada_obstaculos']}")

    resumo_json["tempo_s"] = round(time.time() - t_ini, 1)
    (out / "resumo.json").write_text(json.dumps(resumo_json, indent=2, ensure_ascii=False,
                                                default=float), encoding="utf-8")
    print(f"\nPronto em {resumo_json['tempo_s']} s. Saídas em '{out}'.")


if __name__ == "__main__":
    main()
