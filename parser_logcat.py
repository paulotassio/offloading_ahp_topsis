"""
parser_logcat.py
-----------------
Lê a telemetria do app IAVision e devolve um pandas.DataFrame com uma
linha por quadro. Aceita duas origens:

  1. Exportação do Logcat (formato JSON do Android Studio, extensão
     .logcat): extrai as linhas da tag DATASET_NEURO_FUZZY.
     -> load_dataset_from_logcat()

  2. CSV gravado pelo próprio app no celular (Download/IAVision/
     iavision_AAAAMMDD_HHMMSS_<cenario>.csv, uma sessão por arquivo), sem
     passar pelo Android Studio.
     -> load_dataset_from_csv()

  load_dataset() escolhe o leitor pela extensão do arquivo.

Formatos de log já utilizados no projeto (Logcat):

  Formato original (7 campos):
    frame,RTT_ms,proc_ms,detect_ms,num_objetos,label_principal,score_principal

  Formato com pipeline desacoplado (8 campos):
    frame_id,rtt_rede_ms,espera_fila_ms,deteccao_ms,latencia_total_ms,
    num_objetos,label,score

  Formato atual, com proximidade real do obstáculo (9 campos):
    frame_id,rtt_rede_ms,espera_fila_ms,deteccao_ms,latencia_total_ms,
    num_objetos,label,score,proximidade (razão altura_caixa/altura_frame)

CSV do celular: os mesmos 9 campos do formato atual, já com os nomes de
coluna do DataFrame, mais as colunas:
    timestamp_ms  instante (epoch, ms) em que o quadro chegou ao celular
    delegate      acelerador da inferência naquele quadro (GPU, NNAPI, CPU)
    scenario      rótulo do cenário da coleta, normalizado no app
                  (ex.: "praca_castanhal"; "sem_rotulo" se ficou em branco).
    closest_label, closest_score
                  objeto de MAIOR CAIXA (o mesmo de proximity e do alerta de
                  voz). label/score continuam sendo os de maior score.
    frame_age_ms  idade do quadro no ESP32 quando o pedido chegou (relógio
                  do ESP32). Vazia (NaN) com firmware antigo.
    e2e_latency_ms
                  frame_age_ms + total_latency_ms: captura -> fim da
                  detecção (limite superior). Vazia (NaN) com firmware antigo.

CSVs gravados por versões anteriores do app não têm as colunas mais novas
e continuam sendo lidos normalmente.
"""

import io
import json
import pandas as pd


# Colunas do CSV gravado pelo app (DatasetFileLogger.kt). As 9 primeiras
# são as mesmas devolvidas por load_dataset_from_logcat().
CSV_CORE_COLUMNS = [
    "frame_id", "rtt_ms", "wait_ms", "detect_ms", "total_latency_ms",
    "num_objects", "label", "score", "proximity",
]
CSV_EXTRA_COLUMNS = ["timestamp_ms", "delegate", "scenario",
                     "closest_label", "closest_score", "frame_age_ms", "e2e_latency_ms"]

# Colunas que podem vir vazias numa linha válida (firmware sem idade do quadro)
_OPTIONAL_COLUMNS = ["frame_age_ms", "e2e_latency_ms"]

_INT_COLUMNS = ["frame_id", "rtt_ms", "wait_ms", "detect_ms",
                "total_latency_ms", "num_objects", "timestamp_ms"]
_FLOAT_COLUMNS = ["score", "proximity", "closest_score"]


def load_dataset_from_logcat(filepath: str) -> pd.DataFrame:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    messages = data.get("logcatMessages", [])
    rows = []

    for m in messages:
        if m.get("header", {}).get("tag") != "DATASET_NEURO_FUZZY":
            continue

        parts = m["message"].strip().split(",")

        if len(parts) == 9:
            frame_id, rtt, wait, detect, total, n_obj, label, score, proximity = parts
            rows.append({
                "frame_id": int(frame_id),
                "rtt_ms": int(rtt),
                "wait_ms": int(wait),
                "detect_ms": int(detect),
                "total_latency_ms": int(total),
                "num_objects": int(n_obj),
                "label": label,
                "score": float(score),
                "proximity": float(proximity),
            })
        elif len(parts) == 8:
            frame_id, rtt, wait, detect, total, n_obj, label, score = parts
            rows.append({
                "frame_id": int(frame_id),
                "rtt_ms": int(rtt),
                "wait_ms": int(wait),
                "detect_ms": int(detect),
                "total_latency_ms": int(total),
                "num_objects": int(n_obj),
                "label": label,
                "score": float(score),
                "proximity": None,  # não disponível neste formato de log
            })
        elif len(parts) == 7:
            # Formato antigo: sem coluna de fila nem de proximidade.
            frame_id, rtt, proc, detect, n_obj, label, score = parts
            rtt_i, detect_i = int(rtt), int(detect)
            rows.append({
                "frame_id": int(frame_id),
                "rtt_ms": rtt_i,
                "wait_ms": 0,
                "detect_ms": detect_i,
                "total_latency_ms": rtt_i + detect_i,
                "num_objects": int(n_obj),
                "label": label,
                "score": float(score),
                "proximity": None,
            })
        else:
            continue  # linha em formato inesperado, ignora

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("frame_id").reset_index(drop=True)
    return df


def load_dataset_from_csv(filepath: str) -> pd.DataFrame:
    """
    Lê um CSV de sessão gravado pelo app no celular.

    Robustez:
      - Se o app foi encerrado à força durante a gravação, a última linha
        pode ter ficado pela metade. O app sempre termina cada linha com
        '\\n'; uma última linha sem '\\n' é descartada com aviso.
      - Linhas com campos faltando também são descartadas com aviso.
      - Arquivo vazio ou só com cabeçalho (conexão que falhou) devolve um
        DataFrame vazio.
    """
    with open(filepath, "r", encoding="utf-8", newline="") as f:
        text = f.read()

    if text and not text.endswith("\n"):
        text = text[: text.rfind("\n") + 1]
        print(f"[AVISO] Última linha incompleta descartada em '{filepath}'.")

    if not text.strip():
        return pd.DataFrame(columns=CSV_CORE_COLUMNS + CSV_EXTRA_COLUMNS)

    raw = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)

    missing = [c for c in CSV_CORE_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"'{filepath}' não tem as colunas esperadas do IAVision: faltam {missing}"
        )

    required = CSV_CORE_COLUMNS + [c for c in CSV_EXTRA_COLUMNS
                                   if c in raw.columns and c not in _OPTIONAL_COLUMNS]
    incomplete = raw[required].isna().any(axis=1) | (raw[required] == "").any(axis=1)
    if incomplete.any():
        print(f"[AVISO] {int(incomplete.sum())} linha(s) incompleta(s) descartada(s) em '{filepath}'.")
        raw = raw[~incomplete]

    df = raw.copy()
    for c in _INT_COLUMNS:
        if c in df.columns:
            df[c] = df[c].astype("int64")
    for c in _FLOAT_COLUMNS:
        if c in df.columns:
            df[c] = df[c].astype("float64")
    for c in _OPTIONAL_COLUMNS:
        if c in df.columns:
            # vazio -> NaN; qualquer outro texto não numérico ainda gera erro
            df[c] = pd.to_numeric(df[c].mask(df[c] == ""), errors="raise").astype("float64")

    if not df.empty:
        df = df.sort_values("frame_id").reset_index(drop=True)
    return df


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Carrega um arquivo de telemetria do IAVision escolhendo o leitor pela
    extensão: .csv (gravado no celular) ou .logcat (exportado do Android
    Studio). As 9 colunas principais saem com os mesmos nomes e tipos nos
    dois casos.
    """
    if str(filepath).lower().endswith(".csv"):
        return load_dataset_from_csv(filepath)
    return load_dataset_from_logcat(filepath)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_log.logcat"
    df = load_dataset(path)
    print(f"Linhas carregadas: {len(df)}")
    print(f"Tem coluna de proximidade real: {df['proximity'].notna().any() if not df.empty else 'N/A'}")
    if "scenario" in df.columns and not df.empty:
        print(f"Cenário: {df['scenario'].unique().tolist()}")
    if "delegate" in df.columns and not df.empty:
        print(f"Delegates na sessão: {df['delegate'].value_counts().to_dict()}")
    if "frame_age_ms" in df.columns and df["frame_age_ms"].notna().any():
        print(f"Idade do quadro (ms): mediana {df['frame_age_ms'].median():.0f}, "
              f"p95 {df['frame_age_ms'].quantile(0.95):.0f}")
    print(df.head(10))
    print(df.describe())
