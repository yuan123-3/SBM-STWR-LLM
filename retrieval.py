"""Shared retrieval & evidence assembly (RAG V2), with pluggable data source.

Two data sources:
  - default: the frozen snapshot in 结果文件/ (used by the evaluation harness)
  - session: DataFrames from the running Streamlit app (used by the deployed app)

Both go through the SAME record builders / serializer, so "evaluation == deployment".

Records schema:
  type, region, year, factor, value, p
Serialization adds a stable evidence id (E0001, ...).
"""
from pathlib import Path
import hashlib
import json
import math
import re

import pandas as pd

_HERE = Path(__file__).resolve().parent

# The result snapshot may be shipped as 结果文件.zip (keeps the repo slim).
# If the folder is missing, extract it next to this file on first import.
_SNAP = _HERE / "结果文件"
if not _SNAP.exists():
    _ZIP = _HERE / "结果文件.zip"
    if _ZIP.exists():
        try:
            import zipfile as _zf
            with _zf.ZipFile(_ZIP) as _z:
                _z.extractall(_HERE)
        except Exception:
            pass

# when the code lives in a subfolder, look for the data one level up
ROOT = _HERE if (_HERE / "结果文件").exists() else _HERE.parent
RES = ROOT / "结果文件"
SBM_PATH = RES / "SBM_Details.csv"
STWR_DIR = RES / "STWR_Package" / "Stage2_STWR"
PATH_MATRIX = RES / "Province_Path_Matrix.csv"
PATH_STATS = RES / "Path_Stats_Summary .csv"

EVIDENCE_BUDGET = 200_000

# ------------------------------------------------------------------ default snapshot (OPTIONAL)
# The repository does NOT ship the precomputed snapshot. If it is absent, the module
# still imports: the app can then inject the user's own results via store_from_session().
def _read_csv_safe(path, cols):
    try:
        if Path(path).exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame(columns=cols)


sbm = _read_csv_safe(SBM_PATH, ["Region", "Year", "Efficiency"])
pm = _read_csv_safe(PATH_MATRIX, ["ID"])
ps = _read_csv_safe(PATH_STATS, ["Path", "Freq", "Avg_Eff"])
STWR = {}
if STWR_DIR.exists():
    for d in sorted(STWR_DIR.iterdir()):
        if d.is_dir() and d.name.isdigit():
            try:
                STWR[int(d.name)] = (pd.read_csv(d / "stwr_params.csv"),
                                     pd.read_csv(d / "stwr_p-Values.csv"))
            except Exception:
                continue

PROVINCES = sorted(sbm["Region"].unique().tolist()) if ("Region" in sbm.columns and not sbm.empty) else []
YEARS = sorted(int(y) for y in sbm["Year"].unique()) if ("Year" in sbm.columns and not sbm.empty) else []
INTENTS = ("numeric", "trend", "driver", "path", "policy", "comprehensive")

_ALIASES = {"inner mongol": "Inner Mongolia", "xizang": "Xizang", "tibet": "Xizang"}


def _default_store():
    coef_rows = []
    for y, (par, pv) in STWR.items():
        m = par.merge(pv, on="Ename", suffixes=("_c", "_p"))
        for _, r in m.iterrows():
            for v in par.columns[3:]:
                try:
                    coef_rows.append((r["Ename"], y, v,
                                      round(float(r[v + "_c"]), 4),
                                      round(float(r[v + "_p"]), 4)))
                except Exception:
                    continue
    coef = pd.DataFrame(coef_rows, columns=["region", "year", "factor", "coef", "p"])
    sbm_n = sbm.rename(columns={"Region": "region", "Year": "year", "Efficiency": "efficiency"})[
        ["region", "year", "efficiency"]].copy()
    sbm_n["year"] = pd.to_numeric(sbm_n["year"], errors="coerce")
    path_n = pm.rename(columns={"ID": "region"})
    stats_n = ps.rename(columns={"Path": "path", "Freq": "freq", "Avg_Eff": "avg_eff"})
    return {"sbm": sbm_n, "path": path_n, "stats": stats_n, "coef": coef, "_full": None}


STORE = _default_store()


def set_store(store):
    global STORE
    STORE = store


def get_store():
    return STORE


def store_from_session(sbm=None, sbm_id=None, sbm_time=None,
                       path_raw=None, path_id=None,
                       path_matrix=None, path_stats=None):
    """Build a Store from the running app's session DataFrames."""
    s = pd.DataFrame(columns=["region", "year", "efficiency"])
    if sbm is not None and sbm_id in sbm.columns and "Efficiency" in sbm.columns and sbm_time in sbm.columns:
        s = pd.DataFrame({
            "region": sbm[sbm_id].astype(str).str.strip(),
            "year": pd.to_numeric(sbm[sbm_time], errors="coerce"),
            "efficiency": pd.to_numeric(sbm["Efficiency"], errors="coerce"),
        }).dropna(subset=["year", "efficiency"])
        s["year"] = s["year"].astype(int)

    coef = pd.DataFrame(columns=["region", "year", "factor", "coef", "p"])
    if path_raw is not None and path_id in path_raw.columns and "stm" in path_raw.columns:
        rows = []
        coef_cols = [c for c in path_raw.columns if str(c).startswith("Coef_")]
        for _, r in path_raw.iterrows():
            for c in coef_cols:
                v = str(c)[5:]
                try:
                    cv = float(r[c])
                except Exception:
                    continue
                try:
                    pvv = float(r.get("Pval_" + v))
                except Exception:
                    pvv = float("nan")
                rows.append((str(r[path_id]).strip(), int(r["stm"]), v, round(cv, 4), pvv))
        if rows:
            coef = pd.DataFrame(rows, columns=["region", "year", "factor", "coef", "p"])

    path = pd.DataFrame(columns=["region"])
    if path_matrix is not None and not path_matrix.empty:
        idc = "ID" if "ID" in path_matrix.columns else path_matrix.columns[0]
        path = path_matrix.rename(columns={idc: "region"})

    stats = pd.DataFrame(columns=["path", "freq", "avg_eff"])
    if path_stats is not None and not path_stats.empty:
        ren = {}
        if "Path" in path_stats.columns:
            ren["Path"] = "path"
        if "Freq" in path_stats.columns:
            ren["Freq"] = "freq"
        if "Avg_Eff" in path_stats.columns:
            ren["Avg_Eff"] = "avg_eff"
        stats = path_stats.rename(columns=ren)

    return {"sbm": s, "path": path, "stats": stats, "coef": coef, "_full": None}


# ------------------------------------------------------------------ query understanding
def _provinces(store):
    try:
        return sorted(store["sbm"]["region"].astype(str).unique().tolist())
    except Exception:
        return PROVINCES


def detect_province(text, store=None):
    store = store or STORE
    t = (text or "").lower()
    for p in _provinces(store):
        if p.lower() in t:
            return p
    for alias, p in _ALIASES.items():
        if alias in t:
            return p
    return None


def detect_years(text):
    return sorted({int(y) for y in re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", text or "")})


def infer_intent(q):
    ql = (q or "").lower()
    if "dominant" in ql or "driving factor" in ql or "driver" in ql:
        return "driver"
    if "evolve" in ql or "transition" in ql or "path" in ql:
        return "path"
    if "measure" in ql or "polic" in ql or "intervention" in ql:
        return "policy"
    if "how did" in ql or "change" in ql or "trend" in ql:
        return "trend"
    if re.search(r"\bcee\b", ql) or "what was" in ql:
        return "numeric"
    return "comprehensive"


def plan(question, province=None, years=None, intent=None, store=None):
    store = store or STORE
    return {"question": question,
            "province": province or detect_province(question, store),
            "years": years if years is not None else detect_years(question),
            "intent": intent or infer_intent(question)}


# ------------------------------------------------------------------ record builders
def _sbm_records(store, province=None, y0=None, y1=None):
    df = store["sbm"]
    if province:
        df = df[df["region"].astype(str) == str(province)]
    if y0 is not None:
        df = df[df["year"] >= y0]
    if y1 is not None:
        df = df[df["year"] <= y1]
    return [dict(type="SBM_Efficiency", region=r["region"], year=int(r["year"]),
                 factor="Efficiency", value=round(float(r["efficiency"]), 4), p="")
            for _, r in df.sort_values(["region", "year"]).iterrows()]


def _stwr_records(store, province=None, years=None):
    df = store["coef"]
    if province:
        df = df[df["region"].astype(str) == str(province)]
    if years:
        df = df[df["year"].isin([int(y) for y in years])]
    recs = []
    for _, r in df.iterrows():
        recs.append(dict(type="STWR_Coefficient", region=r["region"], year=int(r["year"]),
                         factor=r["factor"], value=round(float(r["coef"]), 4),
                         p=(round(float(r["p"]), 4) if pd.notna(r["p"]) else "")))
    return recs


def _path_records(store, province=None, y0=None, y1=None):
    df = store["path"]
    if province and "region" in df.columns:
        df = df[df["region"].astype(str) == str(province)]
    recs = []
    for _, r in df.iterrows():
        for col in df.columns:
            if not str(col).startswith("Path_"):
                continue
            period = str(col)[5:]
            try:
                a, b = period.split("-")
                if y0 is not None and int(a) < y0:
                    continue
                if y1 is not None and int(b) > y1:
                    continue
            except Exception:
                pass
            val = r[col]
            if " -> " not in str(val):
                continue
            eff = r.get("DeltaEff_" + period)
            recs.append(dict(type="Path_Transition", region=r.get("region", ""), year=period,
                             factor=str(val),
                             value=round(float(eff), 4) if pd.notna(eff) else "", p=""))
    return recs


def _stats_records(store):
    df = store["stats"]
    recs = []
    for _, r in df.iterrows():
        try:
            recs.append(dict(type="Path_Frequency", region="", year="", factor=str(r["path"]),
                             value=round(float(r["avg_eff"]), 4), p=int(r["freq"])))
        except Exception:
            continue
    return recs


# ------------------------------------------------------------------ full vs retrieval
def full_records(store=None):
    store = store or STORE
    if store.get("_full") is None:
        store["_full"] = (_sbm_records(store) + _stwr_records(store)
                          + _path_records(store) + _stats_records(store))
    return store["_full"]


def retrieve_records(question, province=None, years=None, intent=None, store=None):
    store = store or STORE
    pl = plan(question, province, years, intent, store)
    province, years, intent = pl["province"], pl["years"], pl["intent"]
    y0 = min(years) if years else None
    y1 = max(years) if years else None

    # region-specific intents require a detected province; otherwise: insufficient evidence
    if intent in ("numeric", "trend", "driver", "path") and not province:
        return []

    if intent == "numeric":
        recs = _sbm_records(store, province, y0, y1)
    elif intent == "trend":
        recs = _sbm_records(store, province, y0, y1)
    elif intent == "driver":
        recs = (_sbm_records(store, province) + _stwr_records(store, province, set(years) if years else None)
                + _path_records(store, province))
    elif intent == "path":
        recs = _path_records(store, province) + _sbm_records(store, province)
    elif intent == "policy":
        recs = _stats_records(store) + _path_records(store, province)
    else:
        recs = (_sbm_records(store, province, y0, y1) + _stwr_records(store, province, set(years) if years else None)
                + _path_records(store, province) + _stats_records(store))
    return recs  # no silent fallback: empty stays empty (caller reports insufficient evidence)


def retrieve_task_records(task, store=None):
    """Task-level retrieval (per sub-task scope), with optional factor filtering."""
    store = store or STORE
    region = task.get("region")
    years = task.get("years") or []
    intent = task.get("intent", "comprehensive")
    factors = task.get("factors") or []
    recs = retrieve_records(task.get("goal", ""), province=region, years=years, intent=intent, store=store)
    if factors:
        recs = [r for r in recs if r.get("factor") in factors or r.get("type") == "SBM_Efficiency"]
    return recs


# ------------------------------------------------------------------ serialization (stable JSON)
def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return ""
    return v


def generate_evidence_id(record):
    """Stable id from the record's identifying fields (same record -> same id across tasks)."""
    key = "|".join(str(_clean(record.get(k, ""))) for k in
                   ("type", "region", "year", "factor", "value", "p"))
    return "EV_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _to_items(records):
    items = []
    for r in records:
        it = {k: _clean(v) for k, v in r.items()}
        it["evidence_id"] = generate_evidence_id(it)
        items.append(it)
    return items


def _json(items):
    return json.dumps(items, ensure_ascii=False, allow_nan=False, default=str)


def _fit_json(items, budget):
    full = _json(items)
    if len(full) <= budget:
        return full
    lo, hi = 0, len(items)
    while lo < hi:  # binary search: largest prefix that fits (never cut a record)
        mid = (lo + hi + 1) // 2
        if len(_json(items[:mid])) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return _json(items[:lo])


def serialize(records, budget=EVIDENCE_BUDGET):
    return _fit_json(_to_items(records), budget)


def serialize_strict(records, budget=EVIDENCE_BUDGET):
    """Raise if the full evidence exceeds the budget (for strict/evaluation use)."""
    items = _to_items(records)
    if len(_json(items)) > budget:
        raise ValueError("Evidence exceeds the configured budget.")
    return _json(items)


def full_evidence(store=None):
    return serialize(full_records(store))


def retrieve_evidence(question, province=None, years=None, intent=None, store=None):
    recs = retrieve_records(question, province, years, intent, store)
    return serialize(recs) if recs else ""


def evidence_for_group(group, item, store=None):
    store = store or STORE
    if group == "A":
        return ""
    if group in ("B", "D"):
        return full_evidence(store)
    return retrieve_evidence(item.get("question", ""), item.get("province"), item.get("years"),
                             item.get("category"), store)


# ------------------------------------------------------------------ verification helpers
def fact_lookup(records):
    keys = set()
    for r in records:
        try:
            keys.add((str(r["region"]).lower(), str(r["year"]), round(float(r["value"]), 3)))
        except Exception:
            continue
    return keys


if __name__ == "__main__":
    q = "What was the carbon emission efficiency (CEE) of Fujian in 2020?"
    print("default plan:", plan(q))
    print("default records:", len(retrieve_records(q)))
    # simulated session store (subset, renamed columns like the app)
    sess = store_from_session(sbm=sbm, sbm_id="Region", sbm_time="Year",
                              path_matrix=pm)
    set_store(sess)
    print("session plan:", plan(q))
    print("session evidence head:")
    print(retrieve_evidence(q)[:200])
    set_store(_default_store())
