import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import time
import io
import traceback
import zipfile
import os
import json
import requests
import re
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
from stwr.gwr import GWR, STWR
from stwr.sel_bw import Sel_BW, Sel_Spt_BW
from core import robust_load_data, solve_sbm_undesirable, run_stwr_full_process, analyze_paths, prepare_sankey_data

try:
    from stwr_multislice import run_stwr_multislice, export_stage2_zip
except Exception:
    run_stwr_multislice = None
    export_stage2_zip = None

try:
    import agent_layer as _agent
    import verification as _verification
except Exception:
    _agent = None
    _verification = None

# resolve the shapefile whether the code sits in the project root or a subfolder
from pathlib import Path as _Path
_BASE = _Path(__file__).resolve().parent
_PROJ = _BASE if (_BASE / "ne_10m_admin_1_states_provinces").exists() else _BASE.parent
SHP_PATH = str(_PROJ / "ne_10m_admin_1_states_provinces" / "ne_10m_admin_1_states_provinces.shp")
from prompts import (
    SYSTEM_PROMPT, RETRIEVAL_SYSTEM_PROMPT, ANALYSIS_SYSTEM_PROMPT, OUTPUT_CONTRACT,
    RESULT_META, SYNTHESIS_PROMPT, SYNTHESIS_NAME, FILE_ANALYSIS_PROMPTS,
)

try:
    import retrieval as _retrieval  # V2 fine-grained retrieval (shared with eval)
except Exception:
    _retrieval = None

# ==========================================
# 0. Global styles
# ==========================================
st.set_page_config(page_title="SBM-STWR Full Pipeline Analysis V37.0", layout="wide")


def safe_rerun():
    """Safely trigger a rerun across Streamlit versions."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


st.markdown("""
    <style>
    html, body, [class*="css"], .stMarkdown, p, span, label, input, button, textarea, .stSelectbox div {
        font-family: "Times New Roman", serif !important;
    }
    .stApp { background-color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #e3f2fd !important; border-right: 1px solid #bbdefb; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #0d47a1 !important; }
    .step-box {
        padding: 20px; border: 1px solid #bbdefb; border-radius: 12px;
        background-color: #f1f8ff; margin-bottom: 15px; border-left: 6px solid #2196f3;
    }
    .step-box b { color: #0d47a1; font-size: 1.1em; }
    .ai-msg {
        background-color: #f1f8ff; padding: 18px; border-radius: 15px 15px 15px 0px;
        margin-bottom: 12px; border: 1px solid #bbdefb; border-left: 5px solid #2196f3;
        color: #0d47a1; overflow-x: auto;
    }
    .ai-msg table { border-collapse: collapse; width: 100%; margin: 10px 0; background-color: white; border: 1px solid #bbdefb; }
    .ai-msg th, .ai-msg td { border: 1px solid #bbdefb; padding: 8px; text-align: center; font-size: 14px; }
    .ai-msg th { background-color: #e3f2fd; color: #0d47a1; }
    .user-msg {
        background-color: #f5f7f9; padding: 15px; border-radius: 15px 15px 0px 15px;
        margin-bottom: 12px; color: #37474f; border: 1px solid #cfd8dc;
    }
    .stButton>button { border-radius: 20px; border: 1px solid #64b5f6; background-color: #ffffff; color: #1976d2; }
    .stButton>button:hover { background-color: #e3f2fd; border-color: #1976d2; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# Province name mapping (module-level, supports CN + EN → CN full name for GeoJSON)
# ==========================================
PROVINCE_MAP = {
    "北京": "北京市", "天津": "天津市", "河北": "河北省", "山西": "山西省", "内蒙古": "内蒙古自治区",
    "辽宁": "辽宁省", "吉林": "吉林省", "黑龙江": "黑龙江省", "上海": "上海市", "江苏": "江苏省",
    "浙江": "浙江省", "安徽": "安徽省", "福建": "福建省", "江西": "江西省", "山东": "山东省",
    "河南": "河南省", "湖北": "湖北省", "湖南": "湖南省", "广东": "广东省", "广西": "广西壮族自治区",
    "海南": "海南省", "重庆": "重庆市", "四川": "四川省", "贵州": "贵州省", "云南": "云南省",
    "西藏": "西藏自治区", "陕西": "陕西省", "甘肃": "甘肃省", "青海": "青海省", "宁夏": "宁夏回族自治区",
    "新疆": "新疆维吾尔自治区",
    "Beijing": "北京市", "Tianjin": "天津市", "Hebei": "河北省", "Shanxi": "山西省",
    "Inner Mongolia": "内蒙古自治区", "InnerMongolia": "内蒙古自治区",
    "Liaoning": "辽宁省", "Jilin": "吉林省", "Heilongjiang": "黑龙江省",
    "Shanghai": "上海市", "Jiangsu": "江苏省", "Zhejiang": "浙江省",
    "Anhui": "安徽省", "Fujian": "福建省", "Jiangxi": "江西省", "Shandong": "山东省",
    "Henan": "河南省", "Hubei": "湖北省", "Hunan": "湖南省",
    "Guangdong": "广东省", "Guangxi": "广西壮族自治区", "Hainan": "海南省",
    "Chongqing": "重庆市", "Sichuan": "四川省", "Guizhou": "贵州省",
    "Yunnan": "云南省", "Tibet": "西藏自治区", "Xizang": "西藏自治区",
    "Shaanxi": "陕西省", "Gansu": "甘肃省", "Qinghai": "青海省",
    "Ningxia": "宁夏回族自治区", "Xinjiang": "新疆维吾尔自治区",
}

# Factor name translation: Chinese → English (for map titles, colorbar, Sankey legend)
FACTOR_NAME_MAP = {
    "产业结构": "Industrial structure", "能源结构": "Energy structure",
    "人口数量": "Population", "能源消费强度": "Energy intensity",
    "能源强度": "Energy intensity", "人均GDP": "GDP per capita",
    "城镇化率": "Urbanization rate", "对外开放度": "Trade openness",
    "技术进步": "Technological progress", "环境规制": "Environmental regulation",
    "电力消费": "Electricity consumption", "碳排放强度": "Carbon intensity",
    "维修": "Maintenance", "煤耗": "Coal consumption",
    "标准煤量": "Standard coal", "发电装机容量": "Installed capacity",
    "电力投资额": "Power investment", "总发电量": "Total generation",
    "碳排放": "Carbon emissions", "地区": "Region", "年份": "Year",
    "Intercept": "Intercept",
}

def to_english(name):
    """Return English display name in sentence case (only first word capitalized)."""
    key = name.strip()
    result = FACTOR_NAME_MAP.get(key, key)
    # If not in map, fall back to auto sentence-case
    if result == key:
        words = key.split()
        if len(words) > 1:
            result = words[0].capitalize() + " " + " ".join(w.lower() for w in words[1:])
        else:
            result = key.capitalize()
    return result

# ==========================================
# 1. State initialization
# ==========================================
try:
    from local_secrets import DEEPSEEK_API_KEY as INTERNAL_API_KEY
except Exception:
    INTERNAL_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
ENDPOINT_ID = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")

keys = [
    "chat_messages", "sbm_res_df", "stwr_res_pkg", "path_res_df",
    "path_matrix_df", "path_raw_df", "stwr_diag_summary", "stwr_data",
    "need_ai_response", "last_id_col", "last_tm_col"
]
for k in keys:
    if k not in st.session_state: st.session_state[k] = None
if st.session_state.chat_messages is None: st.session_state.chat_messages = []


def update_ai_context():
    """Aggregate all computed results for the AI assistant."""
    ctx = "### [Full Research Data Report]\n\n"
    if st.session_state.sbm_res_df is not None:
        ctx += "#### [Table 1] SBM Efficiency Results\n" + st.session_state.sbm_res_df.head(2000).to_csv(index=False) + "\n\n"
    if st.session_state.get("stwr_diag_summary"):
        ctx += "#### [Table 2] Model Diagnostics Summary (RSS/AICc/Sigma/R2)\n"
        for name, df in st.session_state.stwr_diag_summary.items():
            ctx += f"**Metric: {name}**\n{df.to_csv(index=False)}\n"
    if st.session_state.path_raw_df is not None:
        ctx += "#### [Table 3] STWR Coefficient Details\n"
        ctx += st.session_state.path_raw_df.head(1000).to_csv(index=False) + "\n\n"
    if st.session_state.path_res_df is not None:
        ctx += "#### [Table 4] Path Frequency Statistics (Step 3)\n"
        ctx += st.session_state.path_res_df.to_csv(index=False) + "\n\n"
    if st.session_state.path_matrix_df is not None:
        ctx += "#### [Table 5] Province Evolution Path Matrix (Step 3)\n"
        ctx += st.session_state.path_matrix_df.head(1000).to_csv(index=False) + "\n\n"
    st.session_state.ai_full_context = ctx


def get_dynamic_data_context(query):
    """Detect provinces/years in the query and return matching data rows."""
    id_col = st.session_state.get('last_id_col')
    tm_col = st.session_state.get('last_tm_col')
    if not id_col or not tm_col: return ""

    detected_provinces = []
    if st.session_state.sbm_res_df is not None:
        all_p = st.session_state.sbm_res_df[id_col].unique().astype(str).tolist()
        for p in all_p:
            if p in query:
                detected_provinces.append(p)
            elif p.replace('省', '').replace('市', '') in query:
                detected_provinces.append(p)
            elif PROVINCE_MAP.get(p, '') in query:
                detected_provinces.append(p)
            elif p in PROVINCE_MAP and PROVINCE_MAP[p] in query:
                detected_provinces.append(p)
    detected_years = [int(y) for y in re.findall(r'\d{4}', query)]

    dynamic_ctx = "\n### [Precision Retrieval Results]\n"
    if st.session_state.sbm_res_df is not None:
        df = st.session_state.sbm_res_df
        mask = pd.Series(False, index=df.index)
        if detected_provinces: mask |= df[id_col].astype(str).isin(detected_provinces)
        if detected_years: mask |= df[tm_col].astype(int).isin(detected_years)
        if mask.any(): dynamic_ctx += "#### Matched SBM Details:\n" + df[mask].to_csv(index=False) + "\n"
    return dynamic_ctx


# ==========================================
# 1b. RAG layer: sub-result catalog, retrieval & strategy-aware synthesis
# ==========================================
RESULT_ROW_LIMIT = 2000


def _serialize_subresult(obj, max_rows=RESULT_ROW_LIMIT):
    if obj is None:
        return "[empty]"
    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            return "[empty]"
        return obj.head(max_rows).to_csv(index=False)
    return str(obj)


def _load_residuals():
    """Extract per-year STWR residuals (with localR2) from the in-memory results zip."""
    pkg = st.session_state.get("stwr_res_pkg")
    if not pkg:
        return None
    try:
        frames = []
        with zipfile.ZipFile(io.BytesIO(pkg)) as zf:
            for name in sorted(zf.namelist()):
                if name.endswith("stwr_residuals.csv"):
                    year = name.split("/")[-2]
                    df = pd.read_csv(io.BytesIO(zf.read(name)))
                    df.insert(0, "Year", year)
                    frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else None
    except Exception:
        return None


def _build_synthesis_data():
    """Concatenate the four evidence classes for the cross-category synthesis chain."""
    ss = st.session_state
    parts = ["=== [1] Efficiency (SBM) ===\n" + _serialize_subresult(ss.sbm_res_df)]

    diag = ss.get("stwr_diag_summary") or {}
    diag_block = []
    for m in ["R2", "AICc", "RSS", "Sigma"]:
        diag_block.append(f"--- {m} ---\n" + _serialize_subresult(diag.get(f"{m}.csv")))
    parts.append("=== [2] Model diagnostics (OLS/GWR/STWR) ===\n" + "\n".join(diag_block))

    parts.append("=== [3] STWR local coefficients (Coef_/Pval_) ===\n" + _serialize_subresult(ss.path_raw_df))
    parts.append("=== [4] STWR residuals / localR2 ===\n" + _serialize_subresult(_load_residuals()))
    parts.append("=== [5] Path frequency ===\n" + _serialize_subresult(ss.path_res_df))
    parts.append("=== [6] Path matrix ===\n" + _serialize_subresult(ss.path_matrix_df))

    return "\n\n".join(parts)


def build_result_catalog():
    """Fixed-named registry of every analysis sub-result: name -> {description, strategy, getter}.

    Descriptions and analysis strategies are imported from prompts.RESULT_META;
    edit prompts.py to tune them. This function only binds each name to its data getter.
    """
    ss = st.session_state

    def meta(name):
        m = RESULT_META.get(name, {})
        return {"description": m.get("description", name), "strategy": m.get("strategy", "")}

    catalog = {}

    catalog["SBM_Efficiency"] = {**meta("SBM_Efficiency"), "getter": lambda: ss.sbm_res_df}

    diag = ss.get("stwr_diag_summary") or {}
    for metric in ["R2", "AICc", "RSS", "Sigma"]:
        name = f"STWR_Diagnostics_{metric}"
        catalog[name] = {**meta(name), "getter": (lambda d=diag.get(f"{metric}.csv"): d)}

    catalog["STWR_Coefficients"] = {**meta("STWR_Coefficients"), "getter": lambda: ss.path_raw_df}
    catalog["Path_Frequency"] = {**meta("Path_Frequency"), "getter": lambda: ss.path_res_df}
    catalog["Path_Matrix"] = {**meta("Path_Matrix"), "getter": lambda: ss.path_matrix_df}
    catalog[SYNTHESIS_NAME] = {**meta(SYNTHESIS_NAME), "getter": _build_synthesis_data}

    return catalog


def _call_llm(system_prompt, user_prompt, temperature=0.1, timeout=180):
    """Single non-streaming LLM call, returns assistant text."""
    payload = {
        "model": ENDPOINT_ID,
        "temperature": temperature,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    r = requests.post(
        API_URL,
        headers={"Authorization": "Bearer " + INTERNAL_API_KEY, "Content-Type": "application/json"},
        json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


def _parse_name_list(text, catalog):
    """Parse the retrieval LLM output into a validated list of catalog names."""
    if not text:
        return []
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\[.*\]", t, re.DOTALL)
    names = []
    if m:
        try:
            names = [str(x).strip() for x in json.loads(m.group(0))]
        except Exception:
            names = []
    if not names and "ALL" in t.upper():
        return list(catalog.keys())
    lower_map = {k.lower(): k for k in catalog}
    valid = []
    for n in names:
        if n.upper() == "ALL":
            return list(catalog.keys())
        key = lower_map.get(n.lower())
        if key and key not in valid:
            valid.append(key)
    return valid


def retrieve_subresults(query, catalog):
    """Retrieval LLM: return the list of sub-result names needed for the query."""
    menu = "\n".join(f"- {name}: {meta['description']}" for name, meta in catalog.items())
    user_prompt = (f"Catalog of sub-results:\n{menu}\n\n"
                   f"User question:\n{query}\n\n"
                   f"Return the JSON array of needed sub-result names.")
    return _parse_name_list(_call_llm(RETRIEVAL_SYSTEM_PROMPT, user_prompt), catalog)


def build_strategy_context(selected, catalog):
    """Assemble retrieved data + per-sub-result analysis strategy for the analysis LLM."""
    blocks = []
    for name in selected:
        meta = catalog.get(name)
        if not meta:
            continue
        blocks.append(
            f"#### Sub-result: {name}\n"
            f"- Description: {meta['description']}\n"
            f"- Analysis strategy: {meta['strategy']}\n"
            f"- Data:\n{_serialize_subresult(meta['getter']())}\n"
        )
    return "\n".join(blocks)


# ==========================================
# 2. Sidebar
# ==========================================
with st.sidebar:
    st.header("Configuration")

    sbm_rts = st.radio("Returns to Scale (RTS):", ["VRS", "CRS"], index=0,
                       help="VRS (Variable Returns to Scale): preferred in most academic papers. CRS: Constant Returns to Scale.")

    st.divider()

    st.subheader("STWR model")
    use_ms = st.checkbox("Use multi-slice STWR (temporal)", value=False,
                         help="Accumulate historical yearly slices so the temporal kernel is active. "
                              "If unchecked, the original single-slice STWR (GWR-like) is used.")
    ms_history = st.number_input("STWR history window (0 = all years)", 0, 30, 0, 1,
                                 help="0: use every year up to the target; N: rolling window of N years.")
    ms_search = st.checkbox("Search STWR bandwidth", value=True,
                            help="Golden-section search for alpha/sita/bw. If it fails, fixed parameters are used.")

    st.divider()

    st.subheader("Agent mode")
    st.session_state.agent_mode = st.checkbox(
        "Agent-driven planning + numerical verification", value=False,
        help="Plan tasks, call tools (reuse/run SBM-STWR-Path), retrieve evidence, "
             "then verify numerical claims deterministically before answering.")

    st.divider()

    p_thresh = st.number_input("P-value Significance Threshold:", 0.001, 0.5, 0.05, 0.01,
                               help="Only variables with p-values below this threshold enter Step 3 path evolution analysis.")

    freq_thresh = st.number_input("Pathway Occurrences:", 1, 100, 1, 1,
                                   help="Paths appearing fewer times than this will be filtered out.")

    st.divider()

    st.subheader("Intelligent Q&A")
    if st.session_state.sbm_res_df is not None:
        with st.form("sidebar_ai_form", clear_on_submit=True):
            user_input = st.text_area("Enter your question:",
                                      placeholder="e.g., Analyze the carbon emission efficiency evolution pattern of Fujian province...",
                                      height=150)
            if st.form_submit_button("Send & Search"):
                st.session_state.chat_messages.append({"role": "user", "content": user_input})
                st.session_state.need_ai_response = True
                safe_rerun()
    else:
        st.info("Please complete the computation before asking questions.")

# ==========================================
# 3. Main interface
# ==========================================
st.title("LLM-assisted SBM-STWR Workflow")

c1, c2 = st.columns(2)
with c1: sbm_f = st.file_uploader("1. Upload SBM Data (xlsx/csv)", type=["xlsx", "csv"])
with c2: stwr_fs = st.file_uploader("2. Upload STWR Data", type=["xlsx", "csv"], accept_multiple_files=True)

df_sbm_raw = None
if sbm_f: df_sbm_raw, _ = robust_load_data(sbm_f)

if stwr_fs:
    ls = [robust_load_data(f)[0] for f in stwr_fs if robust_load_data(f)[0] is not None]
    if ls: st.session_state.stwr_data = pd.concat(ls, ignore_index=True)

if df_sbm_raw is not None:
    st.divider()
    cols = list(df_sbm_raw.columns)
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        id_s, tm_s = st.selectbox("ID Column", cols, 0), st.selectbox("Time Column", cols, 1)
        st.session_state.last_id_col, st.session_state.last_tm_col = id_s, tm_s
        sel_y = st.slider("Analysis Range:", int(df_sbm_raw[tm_s].min()), int(df_sbm_raw[tm_s].max()),
                          (int(df_sbm_raw[tm_s].min()), int(df_sbm_raw[tm_s].max())))
    with sc2:
        in_v = st.multiselect("SBM Inputs", cols)
        go_v = st.multiselect("SBM Desirable Outputs", cols)
        bo_v = st.multiselect("SBM Undesirable Outputs", cols)

    if st.session_state.stwr_data is not None:
        scols = list(st.session_state.stwr_data.columns)
        with sc1:
            id_t = st.selectbox("ID (STWR Match)", scols, 0)
            tm_t = st.selectbox("Time (STWR Match)", scols, 1)
        with sc3:
            lon = st.selectbox("Longitude", scols)
            lat = st.selectbox("Latitude", scols)
            ex_v = st.multiselect("Explanatory variables", scols)

    st.subheader("Computation Stage")
    bc1, bc2, bc3 = st.columns(3)

    # ---- Step 1 ----
    with bc1:
        st.markdown('<div class="step-box"><b>Step 1: Efficiency Evaluation</b></div>', unsafe_allow_html=True)
        if st.button("Run SBM Calculation", use_container_width=True):
            with st.spinner("Solving..."):
                d_c = df_sbm_raw[(df_sbm_raw[tm_s] >= sel_y[0]) & (df_sbm_raw[tm_s] <= sel_y[1])].copy().sort_values(
                    [tm_s, id_s])
                d_c['Efficiency'] = solve_sbm_undesirable(
                    d_c[in_v].values.T, d_c[go_v].values.T, d_c[bo_v].values.T, sbm_rts)
                st.session_state.sbm_res_df = d_c
                st.session_state.raw_sbm_df = df_sbm_raw.copy()
                st.session_state.sbm_config = {"id": id_s, "time": tm_s, "start": int(sel_y[0]), "end": int(sel_y[1]),
                                               "inputs": list(in_v), "good": list(go_v), "bad": list(bo_v), "rts": sbm_rts}
                update_ai_context()
                safe_rerun()
        if st.session_state.sbm_res_df is not None:
            st.success("SBM completed!")
            st.download_button("Download SBM results", st.session_state.sbm_res_df.to_csv(index=False).encode('utf-8-sig'),
                               "SBM_Details.csv", use_container_width=True)

    # ---- Step 2 ----
    with bc2:
        st.markdown('<div class="step-box"><b>Step 2: Spatiotemporal Regression</b></div>', unsafe_allow_html=True)
        is_ready = st.session_state.sbm_res_df is not None and st.session_state.stwr_data is not None
        if st.button("Run STWR Regression", use_container_width=True, disabled=not is_ready):
            with st.spinner("Executing..."):
                sbm_sub = st.session_state.sbm_res_df[[id_s, tm_s, 'Efficiency']].copy()
                sbm_sub.columns = ['sid', 'stm', 'Efficiency']
                stwr_c = st.session_state.stwr_data.copy()
                stwr_c['sid'] = stwr_c[id_t].astype(str).str.strip()
                stwr_c['stm'] = stwr_c[tm_t].astype(int)
                dm = pd.merge(stwr_c, sbm_sub, on=['sid', 'stm'], how='inner')
                if use_ms and run_stwr_multislice is not None:
                    hist = "all" if int(ms_history) == 0 else int(ms_history)
                    res = run_stwr_multislice(
                        dm, id_col=id_t, time_col='stm', lat_col=lat, lon_col=lon,
                        y_col='Efficiency', x_cols=ex_v,
                        history=hist, search=ms_search, standardize=True, spherical=True)
                    z_stwr = export_stage2_zip(res)
                    df_path_full = res['path_df']
                else:
                    z_stwr, _, df_path_full = run_stwr_full_process(dm, id_t, 'stm', lat, lon, 'Efficiency', ex_v)

                diag_dict = {}
                for target in ["R2.csv", "RSS.csv", "AICc.csv", "Sigma.csv"]:
                    path = f"Stage2_STWR/{target}"
                    if path in z_stwr:
                        content = z_stwr[path]
                        if isinstance(content, pd.DataFrame):
                            diag_dict[target] = content
                        else:
                            diag_dict[target] = pd.read_csv(io.StringIO(content.decode('utf-8-sig')))
                st.session_state.stwr_diag_summary = diag_dict

                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for n, c in z_stwr.items():
                        zf.writestr(n, c.to_csv(index=False).encode('utf-8-sig') if isinstance(c, pd.DataFrame) else c)
                st.session_state.stwr_res_pkg = buf.getvalue()
                st.session_state.path_raw_df = df_path_full
                st.session_state.stwr_config = {"id": id_t, "time": tm_t, "lat": lat, "lon": lon, "x": list(ex_v),
                                                "sbm_id": id_s, "sbm_time": tm_s, "use_ms": bool(use_ms),
                                                "history": ms_history, "search": bool(ms_search)}
                update_ai_context()
                safe_rerun()
        if st.session_state.stwr_res_pkg is not None:
            st.success("STWR completed!")
            st.download_button("Download STWR results", st.session_state.stwr_res_pkg, "STWR_Package.zip",
                               use_container_width=True)

    # ---- STWR Coefficient Map (ArcGIS journal style) ----
    if st.session_state.path_raw_df is not None:
        st.divider()
        st.header("Spatial distribution of STWR coefficient estimates")

        df_m = st.session_state.path_raw_df.copy()
        target_id = locals().get('id_t', 'sid')
        if target_id not in df_m.columns:
            candidates = [c for c in df_m.columns
                          if not c.startswith(('Coef_', 'Pval_')) and c != 'stm']
            target_id = candidates[0] if candidates else df_m.columns[0]
        df_m[target_id] = df_m[target_id].astype(str).str.strip().replace(PROVINCE_MAP)

        m_c1, _ = st.columns([1, 4])
        with m_c1:
            m_yr = st.selectbox("Select Map Year", sorted(df_m['stm'].unique()), key="map_yr_top")
            m_var = st.selectbox("Select Coefficient", [c for c in df_m.columns if c.startswith('Coef_')], key="map_var_top")

        try:
            import geopandas as gpd
            from shapely.validation import make_valid

            gdf = gpd.read_file(SHP_PATH)

            # Mainland China (31 provinces, exclude Paracel Islands)
            china_main = gdf[(gdf['admin'] == 'China') & (gdf['name'] != 'Paracel Islands')].copy()
            china_main['name_cn'] = china_main['name_zh']

            # Taiwan – dissolve all counties into one province
            taiwan = gdf[gdf['admin'] == 'Taiwan'].copy()
            if len(taiwan) > 0:
                tw_geom = taiwan.dissolve().geometry.iloc[0]
                tw_row = gpd.GeoDataFrame(
                    {'name_cn': ['台湾省'], 'geometry': [tw_geom]}, crs=gdf.crs)
                china_main = pd.concat([china_main, tw_row], ignore_index=True)

            # Hong Kong – dissolve districts
            hk = gdf[gdf['admin'] == 'Hong Kong S.A.R.'].copy()
            if len(hk) > 0:
                hk_geom = hk.dissolve().geometry.iloc[0]
                hk_row = gpd.GeoDataFrame(
                    {'name_cn': ['香港特别行政区'], 'geometry': [hk_geom]}, crs=gdf.crs)
                china_main = pd.concat([china_main, hk_row], ignore_index=True)

            # Macau
            macau = gdf[gdf['admin'] == 'Macau S.A.R'].copy()
            if len(macau) > 0:
                mc_geom = macau.dissolve().geometry.iloc[0]
                mc_row = gpd.GeoDataFrame(
                    {'name_cn': ['澳门特别行政区'], 'geometry': [mc_geom]}, crs=gdf.crs)
                china_main = pd.concat([china_main, mc_row], ignore_index=True)

            # Repair, dissolve, simplify
            china_main["geometry"] = china_main["geometry"].apply(make_valid)
            china_all = china_main.dissolve(by="name_cn").reset_index()
            china_all["geometry"] = china_all["geometry"].simplify(tolerance=0.02, preserve_topology=True)

            # Full geojson (all provinces + HK/Macau/Taiwan) — used as outline layer
            geo_json_full = json.loads(china_all.to_json())

            df_plot = df_m[df_m["stm"] == m_yr].copy()
            var_display = to_english(m_var.replace("Coef_", "").replace("Intercept", "Intercept"))
            data_locations = set(df_plot[target_id].dropna().unique())

            # Filtered geojson (only provinces with data) — used as colored layer
            data_features = [
                f for f in geo_json_full["features"]
                if f["properties"]["name_cn"] in data_locations
            ]
            geo_json_data = {"type": "FeatureCollection", "features": data_features}

            if not data_features:
                st.warning(
                    "No province names in the STWR results matched the map. "
                    f"Examples of unmatched names: {sorted(data_locations)[:5]}. "
                    "Check that the ID column contains province names (e.g., Shanxi / 山西)."
                )

            cn_to_en = {v: k for k, v in PROVINCE_MAP.items()}

            # Hardcoded label positions: (lon, lat) at visual center of each province
            LABEL_POSITIONS = {
                "Heilongjiang": (127.5, 48.0), "Jilin": (126.0, 43.7),
                "Liaoning": (123.0, 41.8), "Inner Mongolia": (114.0, 44.0),
                "InnerMongolia": (114.0, 44.0),
                "Xinjiang": (85.0, 41.5), "Tibet": (90.0, 32.0),
                "Xizang": (90.0, 32.0),
                "Qinghai": (95.0, 36.0), "Gansu": (103.5, 38.0),
                "Sichuan": (103.5, 30.5), "Yunnan": (102.0, 25.5),
                "Guizhou": (107.0, 26.8), "Guangxi": (108.0, 23.5),
                "Hunan": (111.7, 27.8), "Hubei": (113.0, 31.0),
                "Jiangxi": (116.0, 27.5), "Anhui": (117.0, 32.0),
                "Zhejiang": (120.5, 29.2), "Fujian": (118.0, 26.0),
                "Guangdong": (113.5, 23.5), "Hainan": (110.0, 19.0),
                "Henan": (113.5, 33.8), "Shandong": (118.5, 36.3),
                "Jiangsu": (119.8, 33.0), "Shanghai": (121.45, 31.2),
                "Beijing": (116.3, 39.95), "Tianjin": (117.3, 39.2),
                "Hebei": (116.0, 38.5), "Shanxi": (112.3, 37.5),
                "Shaanxi": (108.8, 35.8), "Ningxia": (106.0, 37.2),
                "Chongqing": (106.8, 30.0),
            }

            label_data = []
            for feat in geo_json_full["features"]:
                name_cn = feat["properties"]["name_cn"]
                en_label = cn_to_en.get(name_cn)
                if en_label is None:
                    continue
                pos = LABEL_POSITIONS.get(en_label)
                if pos is not None:
                    label_data.append({"lon": pos[0], "lat": pos[1], "label": en_label})
            df_labels = pd.DataFrame(label_data)

            fig_m = go.Figure()

            # ── Outline layer: black borders for ALL China incl. HK/Macau/Taiwan ──
            fig_m.add_trace(go.Choropleth(
                geojson=geo_json_full,
                locations=[f["properties"]["name_cn"] for f in geo_json_full["features"]],
                z=[0] * len(geo_json_full["features"]),
                featureidkey="properties.name_cn",
                colorscale=[[0, "rgba(255,255,255,0)"], [1, "rgba(255,255,255,0)"]],
                showscale=False,
                marker_line=dict(color="#222222", width=0.8),
                hoverinfo="skip",
            ))

            # ── Colored layer: only provinces with data ──
            fig_m.add_trace(go.Choropleth(
                geojson=geo_json_data,
                locations=df_plot[target_id],
                z=df_plot[m_var],
                featureidkey="properties.name_cn",
                colorscale=[
                    [0.0, "#e8b4b8"],
                    [0.5, "#faf8f0"],
                    [1.0, "#a3c9b8"],
                ],
                colorbar=dict(
                    title=dict(
                        text=var_display,
                        font=dict(size=14, family="Arial", color="#111"),
                        side="right",
                    ),
                    thickness=20,
                    len=0.6,
                    x=0.92,
                    y=0.5,
                    tickfont=dict(size=10, family="Arial", color="#222"),
                    outlinewidth=0,
                    borderwidth=0,
                    ticklen=3,
                    tickwidth=1,
                    xpad=0,
                ),
                marker_line=dict(color="#333333", width=0.6),
                zmid=0,
                zmin=df_plot[m_var].min(),
                zmax=df_plot[m_var].max(),
                showscale=True,
            ))

            # Province labels
            fig_m.add_trace(go.Scattergeo(
                lon=df_labels["lon"], lat=df_labels["lat"],
                text=df_labels["label"], mode="text",
                textfont=dict(size=9, family="Arial", color="#000000"),
                textposition="middle center",
                hoverinfo="skip", showlegend=False,
            ))

            fig_m.update_geos(
                visible=True,
                fitbounds=False,
                projection_type="natural earth",
                projection_scale=1.15,
                center={"lon": 104, "lat": 37},
                lataxis_range=[17, 54],
                lonaxis_range=[73, 135],
                resolution=50,
                showcoastlines=True,
                coastlinecolor="rgba(120,120,120,0.6)",
                coastlinewidth=0.6,
                showland=False,
                showocean=True,
                oceancolor="rgba(220,235,245,1)",
                showlakes=True,
                lakecolor="rgba(220,235,245,1)",
                showrivers=False,
                showcountries=True,
                countrycolor="rgba(120,120,120,0.4)",
                countrywidth=0.4,
                bgcolor="rgba(220,235,245,1)",
            )
            fig_m.update_layout(
                height=500,
                margin={"r": 10, "t": 30, "l": 5, "b": 10},
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(family="Arial", size=10),
            )
            st.plotly_chart(
                fig_m, use_container_width=True,
                config={
                    "toImageButtonOptions": {"format": "png", "filename": f"STWR_{m_var}_{m_yr}", "scale": 3},
                    "displayModeBar": True,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
            )

        except FileNotFoundError:
            st.warning("china_provinces.geojson not found. Please run download_geojson.py to download the map data.")
        except Exception as e:
            st.error(f"Map display failed: {e}")
            st.code(traceback.format_exc())

    # ---- Step 3 ----
    with bc3:
        st.markdown('<div class="step-box"><b>Step 3: Evolution Analysis</b></div>', unsafe_allow_html=True)
        if st.session_state.path_raw_df is None:
            st.info("Please complete Step 2 first.")
        else:
            if st.button("Discover Evolution Paths", use_container_width=True):
                detail_df, stat_p = analyze_paths(
                    st.session_state.path_raw_df, id_t, 'stm', 'Efficiency', ex_v, p_thresh, freq_thresh)
                st.session_state.path_matrix_df = detail_df
                st.session_state.path_res_df = stat_p
                st.session_state.path_config = {"id": id_t, "time": "stm", "y": "Efficiency", "x": list(ex_v),
                                                "p": p_thresh, "freq": freq_thresh}
                update_ai_context()
                safe_rerun()
        if st.session_state.path_res_df is not None:
            st.success("Evolution complete!")
            st.download_button("Download evolution matrix",
                               st.session_state.path_matrix_df.to_csv(index=False).encode('utf-8-sig'),
                               "Province_Path_Matrix.csv", use_container_width=True)
            st.download_button("Download path statistics",
                               st.session_state.path_res_df.to_csv(index=False).encode('utf-8-sig'),
                               "Path_Stats_Summary.csv", use_container_width=True)

# ---- Factor abbreviations (shared by Sankey legend and pathway analysis) ----
FACTOR_ABBREV = {
    "产业结构": "IS", "Industrial structure": "IS",
    "能源结构": "ES", "Energy structure": "ES",
    "人口数量": "P", "Population": "P",
    "能源消费强度": "EI", "Energy intensity": "EI",
    "能源强度": "EI",
    "人均GDP": "GDP", "GDP per capita": "GDP",
    "城镇化率": "UR", "Urbanization rate": "UR",
    "对外开放度": "TO", "Trade openness": "TO",
    "技术进步": "TP", "Technological progress": "TP",
    "环境规制": "ERS", "Environmental regulation": "ERS",
    "电力消费": "EC", "Electricity consumption": "EC",
    "碳排放强度": "CI", "Carbon intensity": "CI",
    "Intercept": "Int", "截距": "Int",
    "煤耗": "Coal", "标准煤量": "Coal",
    "发电装机容量": "Cap", "电力投资额": "Inv", "总发电量": "Gen",
}


def to_abbrev(name):
    key = name.strip()
    suffix = ""
    if key.endswith("+"):
        suffix = "+"
        key = key[:-1]
    elif key.endswith("-"):
        suffix = "-"
        key = key[:-1]

    if key in FACTOR_ABBREV:
        return FACTOR_ABBREV[key] + suffix
    for full, abbr in FACTOR_ABBREV.items():
        if full.lower() in key.lower() or key.lower() in full.lower():
            return abbr + suffix
    return key + suffix


# ---- Sankey: legend-based, clean nodes ----
if st.session_state.path_matrix_df is not None:
    st.divider()
    st.header("Inter-annual transition pathways of dominant driving factors")

    result = prepare_sankey_data(st.session_state.path_matrix_df)
    if result[0] is None:
        st.info("No path data available for Sankey diagram.")
    else:
        raw_labels, links, u_years, n_colors, legend_map = result

        if raw_labels:
            fig_s = go.Figure()

            fig_s.add_trace(go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=20, thickness=24,
                    label=[""] * len(raw_labels),
                    color=n_colors,
                    line=dict(color="white", width=0.2),
                    hovertemplate="",
                ),
                link=dict(
                    source=links['source'], target=links['target'], value=links['value'],
                    color="rgba(220, 220, 220, 0.25)",
                ),
            ))

            # Build legend with larger markers and fonts
            for code, full_name in sorted(legend_map.items()):
                color = n_colors[raw_labels.index(code)] if code in raw_labels else "rgba(180,180,180,0.8)"
                fig_s.add_trace(go.Scatter(
                    x=[None], y=[None], mode='markers',
                    marker=dict(size=28, color=color, line=dict(width=0.8, color='#888')),
                    legendgroup=code, showlegend=True, name=to_abbrev(full_name),
                ))

            # --- Year column labels at bottom ---
            num_y = len(u_years)
            annos = [dict(
                x=i / (num_y - 1) if num_y > 1 else 0.5,
                y=-0.08, xref="paper", yref="paper",
                text=f"<b>{y}</b>", showarrow=False, font=dict(size=16, family="Arial", color="#555")
            ) for i, y in enumerate(u_years)]

            fig_s.update_layout(
                height=540,
                margin=dict(t=40, b=110, l=40, r=40),
                annotations=annos,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(family="Arial"),
                legend=dict(
                    title=dict(text="Factor Legend", font=dict(size=22)),
                    orientation="v", x=1.01, y=1.0, xanchor="left",
                    font=dict(size=16),
                    borderwidth=0,
                ),
            )
            st.plotly_chart(fig_s, use_container_width=True,
                            config={"toImageButtonOptions": {"format": "png", "filename": "Sankey_Paths", "scale": 3}})



# ---- Path Analysis: Maps + Bar Chart (paper layout) ----
if st.session_state.path_matrix_df is not None:
    st.divider()
    st.header("Pathway analysis")
    
    # FACTOR_ABBREV / to_abbrev are defined at module level above (shared with the Sankey legend)
    
    path_matrix = st.session_state.path_matrix_df
    actual_id_col = 'ID' if 'ID' in path_matrix.columns else path_matrix.columns[0]
    path_cols = [c for c in path_matrix.columns if c.startswith('Path_')]
    eff_cols = [c for c in path_matrix.columns if c.startswith('DeltaEff_')]
    
    all_years = set()
    for col in path_cols:
        period = col.replace('Path_', '')
        parts = period.split('-')
        if len(parts) == 2:
            all_years.add(int(parts[0]))
            all_years.add(int(parts[1]))
    all_years = sorted(all_years)
    
    if not path_cols:
        st.info("No path data available.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            sel_start_year = st.selectbox("Start year", all_years, index=0, key="pa_start_year")
        with col_b:
            sel_end_year = st.selectbox("End year", all_years, index=len(all_years)-1, key="pa_end_year")
        
        sel_periods = [col.replace('Path_', '') for col in path_cols 
                       if sel_start_year <= int(col.split('_')[1].split('-')[0]) <= sel_end_year]
        
        if not sel_periods:
            st.warning(f"No data for {sel_start_year}-{sel_end_year}")
        else:
            period_data = []
            for period in sel_periods:
                path_col = f"Path_{period}"
                eff_col = f"DeltaEff_{period}"
                for idx, row in path_matrix.iterrows():
                    path_val = row.get(path_col)
                    eff_val = row.get(eff_col)
                    if pd.notna(path_val) and str(path_val) != 'None -> None':
                        parts = str(path_val).split(' -> ')
                        if len(parts) == 2 and parts[0] != parts[1]:
                            start_factor = re.sub(r'^\d{2}-\d{2}', '', parts[0]).strip()
                            end_factor = re.sub(r'^\d{2}-\d{2}', '', parts[1]).strip()
                            if start_factor and end_factor and start_factor != 'Non-Sig' and end_factor != 'Non-Sig':
                                period_data.append({
                                    'ID': row[actual_id_col],
                                    'Period': period,
                                    'StartFactor': start_factor,
                                    'EndFactor': end_factor,
                                    'DeltaEff': eff_val if pd.notna(eff_val) else 0
                                })
            
            if not period_data:
                st.info("No valid paths in selected period.")
            else:
                period_df = pd.DataFrame(period_data)
                
                path_stats = period_df.groupby(['StartFactor', 'EndFactor']).agg(
                    Count=('ID', 'count'),
                    AvgDeltaEff=('DeltaEff', 'mean'),
                    Regions=('ID', lambda x: ', '.join(x.unique()[:5]))
                ).reset_index()
                
                path_stats = path_stats[path_stats['Count'] >= freq_thresh].sort_values('Count', ascending=False)
                
                path_stats['PathLabel'] = path_stats['StartFactor'] + ' → ' + path_stats['EndFactor']
                path_stats['PathAbbrev'] = path_stats.apply(
                    lambda r: f"{to_abbrev(r['StartFactor'])}→{to_abbrev(r['EndFactor'])}", axis=1)
                
                all_paths = path_stats.copy()
                all_path_labels = all_paths['PathLabel'].tolist()
                
                neg_colors = ["#e6b0aa", "#d7bde2", "#f0c27f", "#d4ac6e"]
                pos_colors = ["#85c1e9", "#82e0aa", "#76d7c4", "#aed6f1", "#a9dfbf", "#d5f5e3", "#e8f8f5", "#d6eaf8", "#ebf5fb", "#f0f9e8"]
                
                all_unique_paths_for_color = sorted(list(set(
                    all_paths['PathLabel'].tolist()
                )))
                
                path_color_map = {}
                neg_idx = 0
                pos_idx = 0
                for p in all_unique_paths_for_color:
                    avg_eff = all_paths[all_paths['PathLabel'] == p]['AvgDeltaEff'].values[0]
                    if avg_eff < 0:
                        path_color_map[p] = neg_colors[neg_idx % len(neg_colors)]
                        neg_idx += 1
                    else:
                        path_color_map[p] = pos_colors[pos_idx % len(pos_colors)]
                        pos_idx += 1
                
                region_current_paths = []
                for region in period_df['ID'].unique():
                    region_data = period_df[period_df['ID'] == region]
                    path_counts = region_data[['StartFactor', 'EndFactor']].apply(
                        lambda x: f"{x['StartFactor']} → {x['EndFactor']}", axis=1).value_counts()
                    if not path_counts.empty:
                        dominant_path = path_counts.index[0]
                        dominant_count = path_counts.iloc[0]
                        avg_delta = region_data[
                            region_data.apply(lambda x: f"{x['StartFactor']} → {x['EndFactor']}", axis=1) == dominant_path
                        ]['DeltaEff'].mean()
                        region_current_paths.append({
                            'Region': region,
                            'DominantPath': dominant_path,
                            'DominantPathAbbrev': f"{to_abbrev(dominant_path.split(' → ')[0])}→{to_abbrev(dominant_path.split(' → ')[1])}",
                            'Count': dominant_count,
                            'AvgDeltaEff': avg_delta,
                            'EndFactor': dominant_path.split(' → ')[1]
                        })
                
                region_current_df = pd.DataFrame(region_current_paths)
                
                all_path_stats_full = all_paths[['StartFactor', 'EndFactor', 'Count', 'AvgDeltaEff', 'PathLabel']].copy()
                
                last_period = sel_periods[-1] if sel_periods else None
                last_period_data = []
                last_all_regions = []
                if last_period:
                    last_path_col = f"Path_{last_period}"
                    last_eff_col = f"DeltaEff_{last_period}"
                    for idx, row in path_matrix.iterrows():
                        path_val = row.get(last_path_col)
                        eff_val = row.get(last_eff_col)
                        sf, ef = None, None
                        if pd.notna(path_val) and str(path_val) != 'None -> None':
                            parts = str(path_val).split(' -> ')
                            if len(parts) == 2:
                                sf_raw = re.sub(r'^\d{2}-\d{2}', '', parts[0]).strip()
                                ef_raw = re.sub(r'^\d{2}-\d{2}', '', parts[1]).strip()
                                if sf_raw and ef_raw:
                                    sf, ef = sf_raw, ef_raw
                        last_all_regions.append({
                            'Region': row[actual_id_col],
                            'StartFactor': sf, 'EndFactor': ef,
                            'DeltaEff': eff_val if pd.notna(eff_val) else 0
                        })
                        if sf and ef and sf != ef and sf != 'Non-Sig' and ef != 'Non-Sig':
                            last_period_data.append({
                                'ID': row[actual_id_col],
                                'StartFactor': sf, 'EndFactor': ef,
                                'DeltaEff': eff_val if pd.notna(eff_val) else 0
                            })
                
                last_opt_df = pd.DataFrame()
                if last_all_regions:
                    last_opt_df = pd.DataFrame(last_period_data) if last_period_data else pd.DataFrame(columns=['ID','StartFactor','EndFactor','DeltaEff'])
                    if not last_opt_df.empty:
                        last_opt_df['PathLabel'] = last_opt_df['StartFactor'] + ' → ' + last_opt_df['EndFactor']
                    
                    last_region_dominant_opt = []
                    if not last_opt_df.empty:
                        for region in last_opt_df['ID'].unique():
                            rdf = last_opt_df[last_opt_df['ID'] == region]
                            pc = rdf['PathLabel'].value_counts()
                            if not pc.empty:
                                dp = pc.index[0]
                                ad = rdf[rdf['PathLabel'] == dp]['DeltaEff'].mean()
                                sf_end = dp.split(' → ')[1]
                                sf_start = dp.split(' → ')[0]
                                last_region_dominant_opt.append({
                                    'Region': region, 'DominantPath': dp, 
                                    'AvgDeltaEff': ad, 'EndFactor': sf_end, 'StartFactor': sf_start
                                })
                    
                    suggestion_list = []
                    for r in last_region_dominant_opt:
                        suggestion_list.append({
                            'Region': r['Region'], 'EndFactor': r['EndFactor'],
                            'StartFactor': r['StartFactor'], 'DominantPath': r['DominantPath']
                        })
                    
                    existing_regions = set([r['Region'] for r in last_region_dominant_opt])
                    for r in last_all_regions:
                        if r['Region'] in existing_regions:
                            continue
                        existing_regions.add(r['Region'])
                        if pd.notna(r['EndFactor']) and r['EndFactor']:
                            end_f = str(r['EndFactor'])
                            start_f = str(r['StartFactor']) if pd.notna(r['StartFactor']) else "—"
                            dom_path = f"{start_f} → {end_f}"
                        else:
                            end_f = "—"
                            start_f = "—"
                            dom_path = "N/S"
                        suggestion_list.append({
                            'Region': r['Region'], 'EndFactor': end_f,
                            'StartFactor': start_f, 'DominantPath': dom_path
                        })
                    
                    improvement_suggestions = []
                    for current_row in suggestion_list:
                        current_end = str(current_row['EndFactor']).strip()
                        current_path_label = str(current_row['DominantPath']).strip()
                        current_eff = 0
                        
                        parts_cp = current_path_label.split(' → ')
                        if len(parts_cp) == 2:
                            cur_abbrev = f"{to_abbrev(parts_cp[0])}→{to_abbrev(parts_cp[1])}"
                        else:
                            cur_abbrev = current_path_label
                        
                        next_paths = all_path_stats_full[
                            (all_path_stats_full['StartFactor'].str.strip() == current_end)
                        ]
                        
                        current_in_stats = all_path_stats_full[
                            all_path_stats_full['PathLabel'].str.strip() == current_path_label
                        ]
                        if not current_in_stats.empty:
                            next_paths = pd.concat([next_paths, current_in_stats]).drop_duplicates(subset=['PathLabel'])
                        
                        if not next_paths.empty:
                            best_next = next_paths.sort_values('AvgDeltaEff', ascending=False).iloc[0]
                            if str(best_next['PathLabel']).strip() == current_path_label:
                                sug_path = current_path_label
                                sug_abbrev = cur_abbrev
                            else:
                                sug_path = f"{current_end} → {best_next['EndFactor']}"
                                sug_abbrev = f"{to_abbrev(current_end)}→{to_abbrev(best_next['EndFactor'])}"
                            improvement_suggestions.append({
                                'Region': current_row['Region'],
                                'CurrentPath': current_row['DominantPath'],
                                'CurrentPathAbbrev': cur_abbrev,
                                'CurrentEndFactor': current_end,
                                'SuggestedNextFactor': best_next['EndFactor'],
                                'SuggestedPath': sug_path,
                                'SuggestedPathAbbrev': sug_abbrev,
                                'ExpectedDeltaEff': best_next['AvgDeltaEff']
                            })
                        else:
                            improvement_suggestions.append({
                                'Region': current_row['Region'],
                                'CurrentPath': current_row['DominantPath'],
                                'CurrentPathAbbrev': cur_abbrev,
                                'CurrentEndFactor': current_end,
                                'SuggestedNextFactor': 'N/A',
                                'SuggestedPath': 'Not significant',
                                'SuggestedPathAbbrev': 'Not significant',
                                'ExpectedDeltaEff': 0
                            })
                else:
                    improvement_suggestions = []
                
                improvement_df = pd.DataFrame(improvement_suggestions)
                
                last_period_for_color = sel_periods[-1] if sel_periods else None
                last_period_cee = {}
                if last_period_for_color and not last_opt_df.empty:
                    for pl in all_unique_paths_for_color:
                        parts = pl.split(' → ')
                        sf, ef = parts[0], parts[1]
                        match = last_opt_df[(last_opt_df['StartFactor'] == sf) & (last_opt_df['EndFactor'] == ef)]
                        if not match.empty:
                            last_period_cee[pl] = match['DeltaEff'].mean()
                        else:
                            last_period_cee[pl] = all_paths[all_paths['PathLabel'] == pl]['AvgDeltaEff'].values[0]
                else:
                    for pl in all_unique_paths_for_color:
                        last_period_cee[pl] = all_paths[all_paths['PathLabel'] == pl]['AvgDeltaEff'].values[0]
                
                path_color_map = {}
                neg_idx = 0
                pos_idx = 0
                for p in all_unique_paths_for_color:
                    cee_val = last_period_cee.get(p, all_paths[all_paths['PathLabel'] == p]['AvgDeltaEff'].values[0])
                    if cee_val < 0:
                        path_color_map[p] = neg_colors[neg_idx % len(neg_colors)]
                        neg_idx += 1
                    else:
                        path_color_map[p] = pos_colors[pos_idx % len(pos_colors)]
                        pos_idx += 1
                
                try:
                    import geopandas as gpd
                    from shapely.validation import make_valid
                    
                    gdf = gpd.read_file(SHP_PATH)
                    china_main = gdf[(gdf['admin'] == 'China') & (gdf['name'] != 'Paracel Islands')].copy()
                    china_main['name_cn'] = china_main['name_zh']
                    
                    for admin_name, cn_name in [('Taiwan', '台湾省'), ('Hong Kong S.A.R.', '香港特别行政区'), ('Macau S.A.R', '澳门特别行政区')]:
                        sub = gdf[gdf['admin'] == admin_name].copy()
                        if len(sub) > 0:
                            geom = sub.dissolve().geometry.iloc[0]
                            row_gdf = gpd.GeoDataFrame({'name_cn': [cn_name], 'geometry': [geom]}, crs=gdf.crs)
                            china_main = pd.concat([china_main, row_gdf], ignore_index=True)
                    
                    china_main["geometry"] = china_main["geometry"].apply(make_valid)
                    china_all = china_main.dissolve(by="name_cn").reset_index()
                    china_all["geometry"] = china_all["geometry"].simplify(tolerance=0.02, preserve_topology=True)
                    geo_json_full = json.loads(china_all.to_json())
                    
                    geo_config = dict(
                        visible=True, fitbounds=False,
                        projection_type="natural earth",
                        projection_scale=1.15, center={"lon": 104, "lat": 37},
                        lataxis_range=[17, 54], lonaxis_range=[73, 135], resolution=50,
                        showcoastlines=True, coastlinecolor="rgba(100,100,100,0.4)", coastlinewidth=0.4,
                        showland=False, showocean=True, oceancolor="rgba(245,248,250,1)",
                        showlakes=True, lakecolor="rgba(245,248,250,1)",
                        bgcolor="rgba(245,248,250,1)"
                    )
                    base_layout = dict(height=420, margin=dict(r=5, t=5, l=5, b=5), paper_bgcolor="white")
                    
                    def make_base_fig():
                        fig = go.Figure()
                        fig.add_trace(go.Choropleth(
                            geojson=geo_json_full,
                            locations=[f["properties"]["name_cn"] for f in geo_json_full["features"]],
                            z=[0] * len(geo_json_full["features"]),
                            featureidkey="properties.name_cn",
                            colorscale=[[0, "#ffffff"], [1, "#ffffff"]],
                            zmin=0, zmax=1,
                            showscale=False, marker_line=dict(color="#555555", width=0.5), hoverinfo="skip",
                        ))
                        return fig
                    
                    all_path_labels_list = all_paths['PathLabel'].tolist()
                    all_abbrev_list = all_paths['PathAbbrev'].tolist()
                    
                    LABEL_POSITIONS = {
                        "Heilongjiang": (127.5, 48.0), "Jilin": (126.0, 43.7),
                        "Liaoning": (123.0, 41.8), "Inner Mongolia": (114.0, 44.0),
                        "InnerMongolia": (114.0, 44.0),
                        "Xinjiang": (85.0, 41.5), "Tibet": (90.0, 32.0),
                        "Xizang": (90.0, 32.0),
                        "Qinghai": (95.0, 36.0), "Gansu": (103.5, 38.0),
                        "Sichuan": (103.5, 30.5), "Yunnan": (102.0, 25.5),
                        "Guizhou": (107.0, 26.8), "Guangxi": (108.0, 23.5),
                        "Hunan": (111.7, 27.8), "Hubei": (113.0, 31.0),
                        "Jiangxi": (116.0, 27.5), "Anhui": (117.0, 32.0),
                        "Zhejiang": (120.5, 29.2), "Fujian": (118.0, 26.0),
                        "Guangdong": (113.5, 23.5), "Hainan": (110.0, 19.0),
                        "Henan": (113.5, 33.8), "Shandong": (118.5, 36.3),
                        "Jiangsu": (119.8, 33.0), "Shanghai": (121.45, 31.2),
                        "Beijing": (116.3, 39.95), "Tianjin": (117.3, 39.2),
                        "Hebei": (116.0, 38.5), "Shanxi": (112.3, 37.5),
                        "Shaanxi": (108.8, 35.8), "Ningxia": (106.0, 37.2),
                        "Chongqing": (106.8, 30.0),
                    }
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("<p style='font-size:16px;font-weight:bold;color:#000000;margin-bottom:5px;'>(a) Pathway distribution</p>", unsafe_allow_html=True)
                        region_current_df_valid = region_current_df.dropna(subset=['DominantPath'])
                        fig_map_a = make_base_fig()
                        for path_name in all_path_labels_list:
                            mask = region_current_df_valid['DominantPath'] == path_name
                            if mask.any():
                                path_regions = region_current_df_valid[mask]['Region'].tolist()
                                path_mapped = [PROVINCE_MAP.get(r, r) for r in path_regions]
                                hover_texts = [f"{r}<br>Dominant: {path_name}" for r in path_regions]
                                fig_map_a.add_trace(go.Choropleth(
                                    geojson=geo_json_full, locations=path_mapped,
                                    z=[1] * len(path_mapped), featureidkey="properties.name_cn",
                                    colorscale=[[0, path_color_map[path_name]], [1, path_color_map[path_name]]],
                                    showscale=False, marker_line=dict(color="#888888", width=0.3),
                                    name=path_name, hovertext=hover_texts, hoverinfo="text", showlegend=False,
                                ))
                        fig_map_a.update_geos(**geo_config)
                        fig_map_a.update_layout(**base_layout, showlegend=False)
                        st.plotly_chart(fig_map_a, use_container_width=True,
                                        config={"toImageButtonOptions": {"format": "png", "filename": f"MapA_{sel_start_year}-{sel_end_year}", "scale": 3}})
                    
                    with col_b:
                        st.markdown("<p style='font-size:16px;font-weight:bold;color:#000000;margin-bottom:5px;'>(b) Latest pathway distribution</p>", unsafe_allow_html=True)
                        if last_period:
                            last_path_col = f"Path_{last_period}"
                            last_period_all = []
                            for idx, row in path_matrix.iterrows():
                                path_val = row.get(last_path_col)
                                sf, ef = None, None
                                if pd.notna(path_val) and str(path_val) != 'None -> None':
                                    parts = str(path_val).split(' -> ')
                                    if len(parts) == 2:
                                        sf = re.sub(r'^\d{2}-\d{2}', '', parts[0]).strip()
                                        ef = re.sub(r'^\d{2}-\d{2}', '', parts[1]).strip()
                                last_period_all.append({
                                    'Region': row[actual_id_col],
                                    'StartFactor': sf, 'EndFactor': ef
                                })
                            last_all_df = pd.DataFrame(last_period_all)
                            
                            last_region_dominant_all = []
                            for region in last_all_df.iloc[:, 0].unique():
                                rdf = last_all_df[last_all_df.iloc[:, 0] == region]
                                ef_vals = rdf['EndFactor'].dropna()
                                ef_val = ef_vals.iloc[0] if not ef_vals.empty else None
                                sf_vals = rdf['StartFactor'].dropna()
                                sf_val = sf_vals.iloc[0] if not sf_vals.empty else None
                                last_region_dominant_all.append({'Region': region, 'StartFactor': sf_val, 'EndFactor': ef_val})
                            
                            last_region_df_all = pd.DataFrame(last_region_dominant_all)
                            
                            fig_map_b = make_base_fig()
                            
                            regions_with_path = last_region_df_all[last_region_df_all['StartFactor'].notna() & last_region_df_all['EndFactor'].notna()]
                            if not regions_with_path.empty:
                                regions_with_path = regions_with_path.copy()
                                regions_with_path['PathLabel'] = regions_with_path['StartFactor'] + ' → ' + regions_with_path['EndFactor']
                                for path_name in regions_with_path['PathLabel'].unique():
                                    mask = regions_with_path['PathLabel'] == path_name
                                    if mask.any():
                                        pr = regions_with_path[mask]['Region'].tolist()
                                        pm = [PROVINCE_MAP.get(r, r) for r in pr]
                                        ht = [f"{r}<br>Dominant: {path_name}" for r in pr]
                                        fig_map_b.add_trace(go.Choropleth(
                                            geojson=geo_json_full, locations=pm,
                                            z=[1]*len(pm), featureidkey="properties.name_cn",
                                            colorscale=[[0, path_color_map.get(path_name, "#ffffff")],[1, path_color_map.get(path_name, "#ffffff")]],
                                            showscale=False, marker_line=dict(color="#888888", width=0.3),
                                            name=path_name, hovertext=ht, hoverinfo="text", showlegend=False,
                                        ))
                            
                            label_data = []
                            for _, row in last_region_df_all.iterrows():
                                region_cn = PROVINCE_MAP.get(row['Region'], row['Region'])
                                pos = LABEL_POSITIONS.get(row['Region'])
                                if pos:
                                    if pd.notna(row['EndFactor']):
                                        ef_abbrev = to_abbrev(row['EndFactor'])
                                    else:
                                        ef_abbrev = "N/S"
                                    label_data.append({"lon": pos[0], "lat": pos[1], "label": ef_abbrev})
                            
                            if label_data:
                                df_labels_b = pd.DataFrame(label_data)
                                fig_map_b.add_trace(go.Scattergeo(
                                    lon=df_labels_b["lon"], lat=df_labels_b["lat"],
                                    text=df_labels_b["label"], mode="text",
                                    textfont=dict(size=10, family="Arial", color="#000000"),
                                    textposition="middle center",
                                    hoverinfo="skip", showlegend=False,
                                ))
                            
                            fig_map_b.update_geos(**geo_config)
                            fig_map_b.update_layout(**base_layout, showlegend=False)
                            st.plotly_chart(fig_map_b, use_container_width=True,
                                            config={"toImageButtonOptions": {"format": "png", "filename": f"MapB_{last_period}", "scale": 3}})
                    
                    bottom_col1, bottom_col2 = st.columns(2)
                    
                    with bottom_col1:
                        st.markdown("<p style='font-size:16px;font-weight:bold;color:#000000;margin-bottom:5px;'>(c) Pathway details</p>", unsafe_allow_html=True)
                        if last_period_data:
                            last_detail_df = pd.DataFrame(last_period_data)
                            last_detail_df['PathLabel'] = last_detail_df['StartFactor'] + ' → ' + last_detail_df['EndFactor']
                            last_detail_df['PathAbbrev'] = last_detail_df.apply(lambda r: f"{to_abbrev(r['StartFactor'])}→{to_abbrev(r['EndFactor'])}", axis=1)
                            last_detail_df = last_detail_df[last_detail_df['PathLabel'].isin(path_color_map.keys())]
                            if not last_detail_df.empty:
                                last_detail_stats = last_detail_df.groupby(['PathLabel', 'PathAbbrev', 'ID']).agg(
                                    AvgDeltaEff=('DeltaEff', 'mean')
                                ).reset_index()
                                last_detail_stats.columns = ['PathLabel', 'PathAbbrev', 'Region', 'AvgDeltaEff']
                                last_path_order = last_detail_df['PathLabel'].unique().tolist()
                                detail_figs = []
                                def format_val(v):
                                    if v == 0:
                                        return "≈0.00"
                                    elif 0 < v < 0.01:
                                        return "<0.01"
                                    elif -0.01 < v < 0:
                                        return ">-0.01"
                                    else:
                                        return f"{v:.2f}"
                                for pl in last_path_order:
                                    pdata = last_detail_stats[last_detail_stats['PathLabel'] == pl]
                                    if pdata.empty:
                                        continue
                                    pdata = pdata.sort_values('AvgDeltaEff', ascending=False).head(5)
                                    abbrev = pdata['PathAbbrev'].iloc[0]
                                    avg_eff = pdata['AvgDeltaEff'].mean()
                                    bar_color = path_color_map.get(pl, "#cccccc")
                                    fig_detail = go.Figure()
                                    fig_detail.add_trace(go.Bar(
                                        x=pdata['Region'].tolist(),
                                        y=pdata['AvgDeltaEff'].tolist(),
                                        orientation='v',
                                        marker=dict(color=bar_color, line=dict(color='#888888', width=0.5)),
                                        text=[format_val(v) for v in pdata['AvgDeltaEff'].tolist()],
                                        textposition='outside',
                                        textfont=dict(size=12, color='#333333'),
                                        showlegend=False,
                                        cliponaxis=False,
                                    ))
                                    fig_detail.add_hline(y=0, line_dash="dash", line_color="#999999", line_width=1)
                                    fig_detail.update_layout(
                                        height=350, margin=dict(l=40, r=15, t=40, b=30),
                                        yaxis=dict(title="ΔCEE", zeroline=True, zerolinecolor="#cccccc", titlefont=dict(size=11)),
                                        xaxis=dict(title="", tickfont=dict(size=11, color='#333333')),
                                        paper_bgcolor="white", plot_bgcolor="white",
                                        font=dict(family="Arial", size=9),
                                        title=dict(text=f"{abbrev}", font=dict(size=12, family="Arial"), x=0.5),
                                    )
                                    detail_figs.append((fig_detail, abbrev))
                                for i in range(0, len(detail_figs), 2):
                                    row_cols = st.columns(2)
                                    for j in range(2):
                                        if i + j < len(detail_figs):
                                            with row_cols[j]:
                                                st.plotly_chart(detail_figs[i+j][0], use_container_width=True,
                                                                config={"toImageButtonOptions": {"format": "png", "filename": f"Detail_{detail_figs[i+j][1]}", "scale": 3}})
                    
                    with bottom_col2:
                        st.markdown("<p style='font-size:16px;font-weight:bold;color:#000000;margin-bottom:5px;'>(d) Optimized pathway distribution</p>", unsafe_allow_html=True)
                        improvement_df_valid = improvement_df.dropna(subset=['Region'])
                        all_suggested_paths = improvement_df_valid[
                            (improvement_df_valid['SuggestedPath'] != 'No improvement') &
                            (improvement_df_valid['SuggestedPath'] != 'No suggestion') &
                            (improvement_df_valid['SuggestedPath'] != 'Not significant')
                        ]['SuggestedPath'].unique()
                        fig_map_opt = make_base_fig()
                        for sug_path in all_suggested_paths:
                            mask = improvement_df_valid['SuggestedPath'] == sug_path
                            if mask.any():
                                sr = improvement_df_valid[mask]['Region'].tolist()
                                sm = [PROVINCE_MAP.get(r, r) for r in sr]
                                ev = improvement_df_valid[mask]['ExpectedDeltaEff'].values
                                ht = [f"{r}<br>Suggested: {sug_path}<br>ΔCEE: {e:.2f}" for r, e in zip(sr, ev)]
                                fig_map_opt.add_trace(go.Choropleth(
                                    geojson=geo_json_full, locations=sm,
                                    z=[1]*len(sm), featureidkey="properties.name_cn",
                                    colorscale=[[0, path_color_map.get(sug_path, "#cccccc")],[1, path_color_map.get(sug_path, "#cccccc")]],
                                    showscale=False, marker_line=dict(color="#888888", width=0.3),
                                    name=sug_path, hovertext=ht, hoverinfo="text", showlegend=False,
                                ))
                        fig_map_opt.update_geos(**geo_config)
                        fig_map_opt.update_layout(**base_layout, showlegend=False)
                        st.plotly_chart(fig_map_opt, use_container_width=True,
                                        config={"toImageButtonOptions": {"format": "png", "filename": f"MapD_{sel_start_year}-{sel_end_year}", "scale": 3}})
                    
                    st.markdown("---")
                    legend_data = []
                    for idx, (_, row) in enumerate(all_paths.iterrows()):
                        p = row['PathLabel']
                        abbrev = row['PathAbbrev']
                        eff = row['AvgDeltaEff']
                        cnt = row['Count']
                        color = path_color_map[p]
                        legend_data.append((abbrev, p, eff, cnt, color))
                    
                    seen_sug = set()
                    for sp in sorted(all_suggested_paths):
                        if sp not in seen_sug and sp not in all_path_labels_list:
                            seen_sug.add(sp)
                            sp_abbrev = f"{to_abbrev(sp.split(' → ')[0])}→{to_abbrev(sp.split(' → ')[1])}"
                            sp_eff = improvement_df_valid[improvement_df_valid['SuggestedPath'] == sp]['ExpectedDeltaEff'].mean()
                            sp_cnt = len(improvement_df_valid[improvement_df_valid['SuggestedPath'] == sp])
                            sp_color = path_color_map.get(sp, "#cccccc")
                            legend_data.append((sp_abbrev, sp, sp_eff, sp_cnt, sp_color))
                    
                    legend_items = []
                    for abbrev, p, eff, cnt, color in legend_data:
                        legend_items.append(f"<span style='display:inline-flex;align-items:center;margin-right:50px;'><span style='display:inline-block;width:14px;height:14px;background:{color};border-radius:3px;margin-right:6px;border:1px solid #999;'></span><b>{abbrev}</b></span>")
                    legend_items.append("<span style='display:inline-flex;align-items:center;margin-right:50px;'><span style='display:inline-block;width:14px;height:14px;background:#ffffff;border-radius:3px;margin-right:6px;border:1px solid #999;'></span><b>Not applied or not significant</b></span>")
                    
                    mid = (len(legend_items) + 1) // 2
                    row1 = "".join(legend_items[:mid])
                    row2 = "".join(legend_items[mid:])
                    
                    legend_html = f"<div style='margin-top:10px;'><div>{row1}</div><div style='margin-top:10px;'>{row2}</div></div>"
                    st.markdown(legend_html, unsafe_allow_html=True)
                    st.caption(f"Abbreviations: IS=Industrial structure, UR=Urbanization rate, ERS=Environmental regulation strength, ES=Energy structure")
                    
                except FileNotFoundError:
                    st.warning("ne_10m_admin_1_states_provinces.shp not found. Please run download_geojson.py first.")
                except Exception as e:
                    st.error(f"Map display failed: {e}")
                    st.code(traceback.format_exc())


# ==========================================
# 4. AI Chat
# ==========================================
if st.session_state.sbm_res_df is not None:
    st.divider()
    st.header("AI Assistant Chat History")
    for msg in st.session_state.chat_messages:
        label = "User" if msg["role"] == "user" else "AI Assistant"
        style = "user-msg" if msg["role"] == "user" else "ai-msg"
        st.markdown(f'<div class="{style}"><b>{label}:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)

    if st.session_state.get("agent_trace"):
        with st.expander("Agent trace (plan / tools / verification)", expanded=False):
            st.json(st.session_state["agent_trace"])

    if st.session_state.get("need_ai_response"):
        user_q = st.session_state.chat_messages[-1]["content"]

        # ---- Agent mode: plan -> tools -> evidence -> deterministic verification -> repair ----
        if st.session_state.get("agent_mode") and _agent is not None and _verification is not None:
            try:
                allowed_regions, allowed_years = None, None
                try:
                    _sbm = st.session_state.sbm_res_df
                    _idc = st.session_state.get("last_id_col")
                    _tmc = st.session_state.get("last_tm_col")
                    if _sbm is not None and _idc and _tmc:
                        allowed_regions = sorted(_sbm[_idc].astype(str).unique().tolist())
                        allowed_years = [int(v) for v in pd.to_numeric(_sbm[_tmc], errors="coerce").dropna().unique()]
                except Exception:
                    pass
                with st.spinner("Agent planning..."):
                    ok_plan, plan = _agent.agent_plan(user_q, _call_llm, allowed_regions, allowed_years)
                if ok_plan:
                    with st.spinner("Executing task plan..."):
                        exec_res = _agent.execute_agent_plan(plan)
                        records = _agent.collect_records(exec_res)
                        if not records and _retrieval is not None:
                            records = _retrieval.retrieve_records(user_q)
                        ev_text = _retrieval.serialize(records) if (records and _retrieval is not None) else ""
                    if not ev_text:
                        ans = ("Insufficient evidence in the computed results to answer this question. "
                               "Please run the required analytical step or specify a province/year covered by the data.")
                        report = []
                    else:
                        with st.spinner("Generating answer..."):
                            ans = _agent.synthesize_answer(user_q, ev_text, _call_llm)
                        with st.spinner("Verifying numerical claims..."):
                            ans, report = _verification.verify_and_repair(user_q, ans, ev_text, records, _call_llm, max_rounds=1)
                    st.session_state.chat_messages.append({"role": "assistant", "content": ans})
                    st.session_state["agent_trace"] = {
                        "plan": plan,
                        "exec": {k: v.get("status") for k, v in exec_res.items()},
                        "verify": _verification.summarize(report) if report else {"total": 0},
                        "evidence_chars": len(ev_text),
                    }
                    st.session_state.need_ai_response = False
                    safe_rerun()
                    st.stop()
                else:
                    st.warning(f"Agent plan rejected: {plan}. Falling back to standard retrieval.")
            except Exception as e:
                st.warning(f"Agent mode failed: {e}. Falling back to standard retrieval.")

        # ---- V2 fine-grained retrieval (identical to the evaluation harness) ----
        evidence_v2 = ""
        if _retrieval is not None:
            # inject the CURRENT session results so retrieval matches the data on screen
            try:
                _path_raw = st.session_state.path_raw_df
                _store = _retrieval.store_from_session(
                    sbm=st.session_state.sbm_res_df,
                    sbm_id=st.session_state.get("last_id_col"),
                    sbm_time=st.session_state.get("last_tm_col"),
                    path_raw=_path_raw,
                    path_id=(_path_raw.columns[0] if _path_raw is not None and len(_path_raw.columns) else None),
                    path_matrix=st.session_state.path_matrix_df,
                    path_stats=st.session_state.path_res_df,
                )
                _retrieval.set_store(_store)
            except Exception:
                pass
            try:
                with st.spinner("Retrieving evidence (V2)..."):
                    evidence_v2 = _retrieval.retrieve_evidence(user_q)
            except Exception as e:
                st.warning(f"V2 retrieval failed; falling back to catalog routing. Details: {e}")
                evidence_v2 = ""

        if evidence_v2:
            selected = ["RetrievalV2"]
            with st.expander(f"Retrieved evidence (V2) — {len(evidence_v2):,} chars", expanded=False):
                st.code(evidence_v2[:3000] + ("..." if len(evidence_v2) > 3000 else ""))
            prompt = (f"[User Question]\n{user_q}\n\n"
                      f"[Retrieved evidence (V2: province/year filtered)]\n{evidence_v2}")
            system_content = ANALYSIS_SYSTEM_PROMPT + "\n\n" + OUTPUT_CONTRACT
        else:
            catalog = build_result_catalog()
            try:
                with st.spinner("Retrieving relevant sub-results..."):
                    selected = retrieve_subresults(user_q, catalog)
            except Exception as e:
                st.warning(f"Retrieval stage failed, using all sub-results. Details: {e}")
                selected = list(catalog.keys())
            if not selected:
                selected = list(catalog.keys())
            with st.expander(f"Retrieved sub-results ({len(selected)}): " + ", ".join(selected), expanded=False):
                for n in selected:
                    st.markdown(f"**{n}** — {catalog[n]['description']}  \n*Analysis strategy: {catalog[n]['strategy']}*")
            strategy_ctx = build_strategy_context(selected, catalog)
            prompt = (f"[User Question]\n{user_q}\n\n"
                      f"[Retrieved Sub-results with Data and Analysis Strategies]\n{strategy_ctx}")
            system_content = ANALYSIS_SYSTEM_PROMPT
            if SYNTHESIS_NAME in selected:
                system_content += "\n\n" + SYNTHESIS_PROMPT
            system_content += "\n\n" + OUTPUT_CONTRACT
        area, full_resp = st.empty(), ""
        payload = {
            "model": ENDPOINT_ID, "temperature": 0.1, "stream": True,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
        }
        try:
            r = requests.post(
                API_URL,
                headers={"Authorization": "Bearer " + INTERNAL_API_KEY, "Content-Type": "application/json"},
                json=payload, timeout=180, stream=True)

            if r.status_code != 200:
                st.error(f"API error (HTTP {r.status_code}): {r.text[:500]}")
                st.session_state.need_ai_response = False
            else:
                for line in r.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if not line.startswith("data: "): continue
                        data_str = line[6:]
                        if data_str == "[DONE]": break
                        try:
                            content = json.loads(data_str)["choices"][0]["delta"].get("content", "")
                            full_resp += content
                            area.markdown(f'<div class="ai-msg"><b>AI Assistant:</b><br>\n\n{full_resp}\n\n</div>',
                                          unsafe_allow_html=True)
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

                if not full_resp:
                    st.warning("AI returned an empty response. Check your API key and endpoint ID, or try a different question.")
                else:
                    st.session_state.chat_messages.append({"role": "assistant", "content": full_resp})

                st.session_state.need_ai_response = False
                safe_rerun()
        except requests.exceptions.Timeout:
            st.error("Request timed out (180s). The API may be slow. Please try again.")
            st.session_state.need_ai_response = False
        except Exception as e:
            st.error(f"Connection error: {str(e)}")
            st.session_state.need_ai_response = False
