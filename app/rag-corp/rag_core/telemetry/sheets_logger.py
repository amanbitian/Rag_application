# app/rag-corp/rag_core/telemetry/sheets_logger.py
from __future__ import annotations
import os, json, time, socket, platform
from typing import Iterable, Dict, Any, List

COLS = [
    "date",
    "session_id",
    "file_name",
    "models_selected",
    "prompt",
    "model_name",
    "model_output",
    "latency_wall_s",
    "prompt_tokens",
    "gen_tokens",
    "load_duration_s",
    "total_duration_s",
    "tokens_per_s",
    "device",
]

def log_rows(rows: Iterable[Dict[str, Any]]) -> None:
    """
    Append rows to Google Sheets (append-only). No-op if not configured.

    Each input row may include:
      - date (auto)
      - session_id (str)
      - file_name (str)
      - models_selected (comma-separated str)
      - prompt (str)
      - model_name (str)
      - model_output (str)
      - metrics (dict)  # will be flattened into metric columns
      - device (auto)
    """
    sheet_id = os.getenv("SHEET_ID")
    sheet_tab = os.getenv("SHEET_TAB", "Sheet1")
    if not sheet_id:
        return

    client = _get_gspread_client()
    if client is None:
        return

    sh = client.open_by_key(sheet_id)
    ws = _ensure_worksheet_with_columns(sh, sheet_tab, COLS)

    now = _now()
    device = _device_label()

    values: List[List[str]] = []
    for r in rows:
        m = _flatten_metrics(r.get("metrics", {}))
        row = [
            r.get("date", now),
            r.get("session_id", ""),
            r.get("file_name", ""),
            r.get("models_selected", ""),
            r.get("prompt", ""),
            r.get("model_name", ""),
            r.get("model_output", ""),
            m.get("latency_wall_s", ""),
            m.get("prompt_tokens", ""),
            m.get("gen_tokens", ""),
            m.get("load_duration_s", ""),
            m.get("total_duration_s", ""),
            m.get("tokens_per_s", ""),
            r.get("device", device),
        ]
        values.append(row)

    # APPEND-ONLY
    ws.append_rows(values, value_input_option="RAW")


# --------------------------- helpers ---------------------------

def _flatten_metrics(stats: Dict[str, Any]) -> Dict[str, str]:
    """Normalize/rename Ollama stats → our fixed columns."""
    return {
        "latency_wall_s": _fmt(stats.get("latency_wall_s")),
        "prompt_tokens": _fmt(stats.get("prompt_eval_count")),
        "gen_tokens": _fmt(stats.get("eval_count")),
        "load_duration_s": _fmt(_ns_to_s(stats.get("load_duration")) or stats.get("load_duration_s")),
        "total_duration_s": _fmt(_ns_to_s(stats.get("total_duration")) or stats.get("total_duration_s")),
        "tokens_per_s": _fmt(stats.get("tokens_per_s")),
    }

def _fmt(v: Any) -> str:
    if v is None: return ""
    if isinstance(v, float): return f"{v:.4f}"
    return str(v)

def _ns_to_s(ns: Any) -> float | None:
    try:
        return float(ns) / 1e9
    except Exception:
        return None

def _ensure_worksheet_with_columns(sh, title: str, columns: List[str]):
    # create or open sheet
    try:
        ws = sh.worksheet(title)
    except Exception:
        ws = sh.add_worksheet(title=title, rows=1000, cols=max(10, len(columns)))
        ws.append_row(columns, value_input_option="RAW")
        return ws

    # ensure header exists & includes all required columns (append missing at end)
    try:
        header = ws.row_values(1)
    except Exception:
        header = []
    if not header:
        ws.update("A1", [columns])
    else:
        missing = [c for c in columns if c not in header]
        if missing:
            new_header = header + missing
            ws.update("A1", [new_header])
    return ws

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _device_label() -> str:
    return os.getenv("DEVICE_NAME") or f"{platform.system()} {platform.release()} | {platform.machine()} | host:{socket.gethostname()}"

def _get_gspread_client():
    # lazy imports so app runs even if logging is not configured
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except Exception:
        return None

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT")
    b64 = os.getenv("GOOGLE_SERVICE_ACCOUNT_B64")

    # 1) file path
    if path:
        p = path.strip()
        try:
            if p.startswith("{") and p.endswith("}"):
                data = json.loads(p)
                creds = Credentials.from_service_account_info(data, scopes=scopes)
            else:
                creds = Credentials.from_service_account_file(p, scopes=scopes)
            return gspread.authorize(creds)
        except Exception:
            pass
    # 2) raw json
    if raw:
        try:
            data = json.loads(raw)
            creds = Credentials.from_service_account_info(data, scopes=scopes)
            return gspread.authorize(creds)
        except Exception:
            pass
    # 3) base64 json
    if b64:
        try:
            import base64
            data = json.loads(base64.b64decode(b64).decode("utf-8"))
            creds = Credentials.from_service_account_info(data, scopes=scopes)
            return gspread.authorize(creds)
        except Exception:
            pass
    return None
