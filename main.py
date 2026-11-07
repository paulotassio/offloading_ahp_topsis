"""
main.py
--------

Uso — um arquivo por vez:
    python main.py RESULTADOS_LOGS/Xiaomi-..._181229.logcat

Uso — dois arquivos combinados (recomendado para reproduzir o artigo):
    python main.py RESULTADOS_LOGS/Xiaomi-..._181229.logcat ^
                   RESULTADOS_LOGS/Xiaomi-..._183131.logcat

Uso — CSVs gravados pelo app no celular (pasta Download/IAVision copiada
para o PC). Aceita arquivos .csv avulsos ou a pasta inteira; na pasta, usa
todos os iavision_*.csv em ordem cronologica, um por sessao:
    python main.py IAVision
    python main.py IAVision/iavision_20261025_100000_praca_castanhal.csv

Saidas geradas:
    resultado_dataset.csv     — telemetria + decisoes fuzzy e ANFIS por frame
    resultado_graficos.png    — 4 graficos do pipeline
"""

import glob
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parser_logcat import load_dataset
from features import add_features
from fuzzy_system import build_fuzzy_system, compute_risk, decide
from anfis_model import ANFIS, train_anfis


# =====================================================================
def run_pipeline(log_paths: list, output_prefix: str = "resultado"):
    """
    Executa o pipeline completo sobre um ou mais arquivos de log.
    Quando mais de um arquivo e passado, os dados sao concatenados
    antes do treinamento do ANFIS.
    """

    # 1. Carregar e combinar todos os logs
    dfs = []
    for path in log_paths:
        df_i = load_dataset(path)   # .logcat (Android Studio) ou .csv (celular)
        if df_i.empty:
            print(f"[AVISO] Nenhuma linha DATASET_NEURO_FUZZY encontrada em '{path}'.")
            continue
        # Marca a sessao de origem para rastreabilidade. As sessoes A e B
        # do estudo piloto (.logcat) mantem o rotulo do artigo; as demais
        # usam o nome do arquivo (ex.: iavision_20261025_100000_praca_castanhal).
        # A regra A/B vale so para .logcat: um CSV gravado as 18:12:29 de
        # outro dia nao pode virar "Sessao A".
        is_logcat = not path.lower().endswith(".csv")
        label = ("A" if is_logcat and "181229" in path
                 else "B" if is_logcat and "183131" in path
                 else os.path.splitext(os.path.basename(path))[0])
        df_i["session"] = label
        cenario = (f", cenario {df_i['scenario'].iloc[0]}"
                   if "scenario" in df_i.columns else "")
        print(f"[1/5] {len(df_i)} frames carregados de '{path}' (Sessao {label}{cenario}).")
        dfs.append(df_i)

    if not dfs:
        print("Nenhum dado valido encontrado. Encerrando.")
        return

    df = pd.concat(dfs, ignore_index=True)
    print(f"[1/5] Total combinado: {len(df)} frames "
          f"({' + '.join(str(len(d)) for d in dfs)} quadros).\n")

    # 2. Features
    df = add_features(df)
    print("[2/5] Features derivadas: rtt_jitter_ms, criticality.")

    # 3. Fuzzy especialista (professor / bootstrap do ANFIS)
    system, _ = build_fuzzy_system()
    import skfuzzy.control as ctrl_mod

    riscos_fuzzy = []
    for _, row in df.iterrows():
        sim = ctrl_mod.ControlSystemSimulation(system)
        r = compute_risk(sim,
                         row["total_latency_ms"],
                         row["rtt_jitter_ms"],
                         row["criticality"])
        riscos_fuzzy.append(r)

    df["risco_fuzzy"]     = riscos_fuzzy
    df["qoe_safety_fuzzy"] = 1 - df["risco_fuzzy"]
    df["decisao_fuzzy"]   = df["risco_fuzzy"].apply(decide)
    print("[3/5] Sistema fuzzy especialista avaliado para todos os frames.")

    # 4. Treinar ANFIS sobre o dataset combinado (metodo hibrido de Jang)
    lat_max  = max(df["total_latency_ms"].max(), 1e-6)
    jit_max  = max(df["rtt_jitter_ms"].max(), 1e-6)

    x = np.stack([
        df["total_latency_ms"].to_numpy(dtype=np.float32) / lat_max,
        df["rtt_jitter_ms"].to_numpy(dtype=np.float32)    / jit_max,
        df["criticality"].to_numpy(dtype=np.float32),
    ], axis=1).astype(np.float32)
    y = df["risco_fuzzy"].to_numpy(dtype=np.float32)

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    torch.manual_seed(42)                       # reproducibilidade
    model = ANFIS(n_inputs=3, n_mfs_per_input=3)
    model.init_membership_from_ranges([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])

    print("[4/5] Treinando ANFIS (metodo hibrido de Jang, 100 epocas)...")
    history = train_anfis(
        model, x_tensor, y_tensor,
        epochs=100,
        lr_antecedents=0.0005,
        verbose=True,
    )

    with torch.no_grad():
        df["risco_anfis"]     = model(x_tensor).clamp(0, 1).numpy()
    df["qoe_safety_anfis"] = 1 - df["risco_anfis"]
    df["decisao_anfis"]    = df["risco_anfis"].apply(decide)

    mae  = np.mean(np.abs(df["risco_fuzzy"] - df["risco_anfis"]))
    rmse = np.sqrt(np.mean((df["risco_fuzzy"] - df["risco_anfis"]) ** 2))
    print(f"      MAE  (fuzzy vs ANFIS): {mae:.4f}")
    print(f"      RMSE (fuzzy vs ANFIS): {rmse:.4f}")

    # 5. Exportar CSV + graficos
    csv_path = f"{output_prefix}_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"[5/5] CSV exportado: {csv_path}")

    _plot_results(df, history, output_prefix)
    _print_summary(df)

    return df, model


# =====================================================================
def _plot_results(df: pd.DataFrame, history: list, output_prefix: str):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) Curva de treinamento do ANFIS
    axes[0, 0].plot(history, color="#6D2E46")
    axes[0, 0].set_title("Convergencia do ANFIS (aprendizado hibrido de Jang)")
    axes[0, 0].set_xlabel("Epoca")
    axes[0, 0].set_ylabel("MSE (ANFIS vs. fuzzy especialista)")

    # (b) QoE-Safety ao longo dos frames, com marcador de sessao
    axes[0, 1].plot(df["qoe_safety_fuzzy"].values,
                    label="Fuzzy especialista", alpha=0.7, linewidth=0.7)
    axes[0, 1].plot(df["qoe_safety_anfis"].values,
                    label="ANFIS", alpha=0.7, linewidth=0.7)
    # Linha vertical marcando o inicio de cada sessao apos a primeira
    # (com os logs A e B do piloto, uma unica linha em "Sessao B →")
    if "session" in df.columns and df["session"].nunique() > 1:
        inicios = np.flatnonzero(df["session"].ne(df["session"].shift()).to_numpy())[1:]
        for sep in inicios:
            axes[0, 1].axvline(sep, color="gray", linestyle="--",
                               linewidth=0.8, label=f"Sessao {df['session'].iloc[sep]} →")
    axes[0, 1].set_title("Metrica QoE-Safety ao longo do tempo")
    axes[0, 1].set_xlabel("Indice do quadro (Sessao A concatenada com Sessao B)")
    axes[0, 1].set_ylabel("QoE-Safety (0-1)")
    axes[0, 1].legend(fontsize=8)

    # (c) Distribuicao de decisoes (fuzzy)
    counts = df["decisao_fuzzy"].value_counts()
    cores  = {"EDGE": "#55A868", "HIBRIDO": "#DD8452", "LOCAL": "#C44E52"}
    axes[1, 0].bar(counts.index,
                   counts.values,
                   color=[cores.get(d, "#999") for d in counts.index])
    axes[1, 0].set_title("Distribuicao de decisoes — Fuzzy especialista")
    axes[1, 0].set_ylabel("Numero de quadros")

    # (d) Latencia x QoE-Safety, colorida por decisao
    for decisao, cor in cores.items():
        sub = df[df["decisao_fuzzy"] == decisao]
        axes[1, 1].scatter(sub["total_latency_ms"], sub["qoe_safety_fuzzy"],
                           label=decisao, color=cor, alpha=0.5, s=10)
    axes[1, 1].set_title("Latencia total vs. QoE-Safety, por decisao")
    axes[1, 1].set_xlabel("Latencia total do pipeline (ms)")
    axes[1, 1].set_ylabel("QoE-Safety")
    axes[1, 1].legend()

    plt.tight_layout()
    fig_path = f"{output_prefix}_graficos.png"
    plt.savefig(fig_path, dpi=150)
    print(f"      Graficos salvos: {fig_path}")
    plt.close()


# =====================================================================
def _print_summary(df: pd.DataFrame):
    n = len(df)
    print(f"\n{'='*50}")
    print(f"RESUMO — {n} quadros no total")
    print(f"{'='*50}")

    # Por sessao (se houver)
    if "session" in df.columns and df["session"].nunique() > 1:
        for s in sorted(df["session"].unique()):
            sub = df[df["session"] == s]
            # Fracao = detect_medio / total_medio (ratio das medias, nao media dos ratios)
            frac = sub["detect_ms"].mean() / sub["total_latency_ms"].mean() * 100
            print(f"\n  Sessao {s} ({len(sub)} quadros):")
            print(f"    Latencia total mediana: {sub['total_latency_ms'].median():.1f} ms")
            print(f"    Fracao deteccao:        {frac:.1f}%")
        print()

    # Metricas globais
    mae  = np.mean(np.abs(df["risco_fuzzy"] - df["risco_anfis"]))
    rmse = np.sqrt(np.mean((df["risco_fuzzy"] - df["risco_anfis"]) ** 2))
    print(f"  MAE  ANFIS vs. fuzzy:  {mae:.4f}")
    print(f"  RMSE ANFIS vs. fuzzy:  {rmse:.4f}")
    print(f"  QoE-Safety medio fuzzy:{df['qoe_safety_fuzzy'].mean():.4f}")
    print(f"  QoE-Safety medio ANFIS:{df['qoe_safety_anfis'].mean():.4f}")

    print("\n  Distribuicao de decisoes (fuzzy especialista):")
    dist = df["decisao_fuzzy"].value_counts(normalize=True).mul(100).round(1)
    for d, v in dist.items():
        print(f"    {d:<8} {v}%")
    print(f"{'='*50}\n")


# =====================================================================
def _expandir_entradas(args: list) -> list:
    """Troca cada pasta pelos iavision_*.csv que ela contem (ordem cronologica)."""
    paths = []
    for a in args:
        if os.path.isdir(a):
            encontrados = sorted(glob.glob(os.path.join(a, "iavision_*.csv")))
            if not encontrados:
                print(f"[AVISO] Nenhum arquivo iavision_*.csv na pasta '{a}'.")
            paths.extend(encontrados)
        else:
            paths.append(a)
    return paths


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python main.py log1.logcat [log2.logcat ...]")
        print("     python main.py sessao1.csv [sessao2.csv ...]")
        print("     python main.py pasta_IAVision")
        sys.exit(1)

    log_paths = _expandir_entradas(sys.argv[1:])
    run_pipeline(log_paths, output_prefix="resultado")