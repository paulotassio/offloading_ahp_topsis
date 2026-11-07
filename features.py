"""
features.py
------------
Deriva, a partir dos dados brutos de telemetria, as três variáveis de
entrada usadas pela métrica:

  1. Latência total (ms)         -> já vem pronta do log (total_latency_ms)
  2. Estabilidade da rede        -> jitter: desvio padrão móvel do RTT
                                     numa janela deslizante de frames
  3. Criticidade da detecção     -> se o log tiver a coluna 'proximity'
                                     (razão altura da caixa / altura do
                                     frame, disponível a partir da versão
                                     atual do app), ela é usada
                                     diretamente como criticidade. Caso
                                     contrário (logs antigos), cai para o
                                     proxy baseado em score x nº objetos.
"""

import pandas as pd


def add_features(df: pd.DataFrame, jitter_window: int = 10) -> pd.DataFrame:
    df = df.copy()

    # --- Estabilidade da rede: desvio padrão móvel do RTT (jitter) ---
    df["rtt_jitter_ms"] = (
        df["rtt_ms"]
        .rolling(window=jitter_window, min_periods=1)
        .std()
        .fillna(0.0)
    )

    # --- Criticidade da detecção ---
    has_real_proximity = "proximity" in df.columns and df["proximity"].notna().any()

    if has_real_proximity:
        # Proximidade real (0 a 1): objeto ocupando maior parte do frame
        # = mais próximo = mais crítico. Preenche frames sem detecção
        # (proximity NaN/None) com 0 (nenhum obstáculo == sem criticidade).
        df["criticality"] = df["proximity"].fillna(0.0).clip(0.0, 1.0)
        df["criticality_source"] = "proximidade_real"
    else:
        # Proxy (logs antigos, sem a coluna de proximidade): score de
        # confiança ponderado pelo número de objetos no quadro.
        df["criticality"] = (df["score"] * (1.0 + 0.15 * df["num_objects"])).clip(0.0, 1.0)
        df["criticality_source"] = "proxy_score_num_objetos"

    return df


if __name__ == "__main__":
    from parser_logcat import load_dataset_from_logcat

    df = load_dataset_from_logcat("sample_log.logcat")
    df = add_features(df)
    print(f"Fonte da criticidade usada: {df['criticality_source'].iloc[0]}")
    print(df[["frame_id", "total_latency_ms", "rtt_jitter_ms", "criticality"]].head(15))
    print("\nEstatísticas:")
    print(df[["total_latency_ms", "rtt_jitter_ms", "criticality"]].describe())
