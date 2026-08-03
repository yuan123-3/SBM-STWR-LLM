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
INTERNAL_API_KEY = "73ebca76-10c5-42ae-9fe2-0fb478cd3bdd"
ENDPOINT_ID = "ep-20260129204202-7z4hn"

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
# 2. Sidebar
# ==========================================
with st.sidebar:
    st.header("Configuration")

    sbm_rts = st.radio("Returns to Scale (RTS):", ["VRS", "CRS"], index=0,
                       help="VRS (Variable Returns to Scale): preferred in most academic papers. CRS: Constant Returns to Scale.")

    st.divider()

    p_thresh = st.number_input("P-value Significance Threshold:", 0.001, 0.5, 0.05, 0.01,
                               help="Only variables with p-values below this threshold enter Step 3 path evolution analysis.")

    freq_thresh = st.number_input("Path Frequency Threshold (%):", 0.0, 10.0, 1.0, 0.1,
                                  help="Minimum frequency for paths shown in the Sankey diagram.") / 100.0

    st.divider()

    st.subheader("AI Assistant")
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
st.title("SBM-STWR Intelligent Research Assistant")

c1, c2 = st.columns(2)
with c1: sbm_f = st.file_uploader("1. Upload SBM Data (xlsx/csv)", type=["xlsx", "csv"])
with c2: stwr_fs = st.file_uploader("2. Upload STWR Driver Data", type=["xlsx", "csv"], accept_multiple_files=True)

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
            ex_v = st.multiselect("Explanatory Variables (Drivers)", scols)

    st.subheader("Computation Console")
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
                update_ai_context()
                safe_rerun()
        if st.session_state.sbm_res_df is not None:
            st.success("SBM Complete")
            st.download_button("Download SBM Results", st.session_state.sbm_res_df.to_csv(index=False).encode('utf-8-sig'),
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
                update_ai_context()
                safe_rerun()
        if st.session_state.stwr_res_pkg is not None:
            st.success("STWR Complete")
            st.download_button("Download Regression Package", st.session_state.stwr_res_pkg, "STWR_Package.zip",
                               use_container_width=True)

    # ---- STWR Coefficient Map (ArcGIS journal style) ----
    if st.session_state.path_raw_df is not None:
        st.divider()
        st.header("STWR coefficients")

        df_m = st.session_state.path_raw_df.copy()
        target_id = locals().get('id_t', 'sid')
        df_m[target_id] = df_m[target_id].astype(str).str.strip().replace(PROVINCE_MAP)

        m_c1, _ = st.columns([1, 4])
        with m_c1:
            m_yr = st.selectbox("Select Map Year", sorted(df_m['stm'].unique()), key="map_yr_top")
            m_var = st.selectbox("Select Coefficient", [c for c in df_m.columns if c.startswith('Coef_')], key="map_var_top")

        try:
            import geopandas as gpd
            from shapely.validation import make_valid

            gdf = gpd.read_file('ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp')
            china = gdf[(gdf['admin'] == 'China') & (gdf['name'] != 'Paracel Islands')].copy()
            china['name_cn'] = china['name_zh']

            # Repair geometry, dissolve by name, simplify to reduce fragment triangles
            china["geometry"] = china["geometry"].apply(make_valid)
            china = china.dissolve(by="name_cn").reset_index()
            china["geometry"] = china["geometry"].simplify(tolerance=0.02, preserve_topology=True)
            geo_json = json.loads(china.to_json())

            cn_to_en = {v: k for k, v in PROVINCE_MAP.items()}

            df_plot = df_m[df_m["stm"] == m_yr].copy()
            var_display = to_english(m_var.replace("Coef_", "").replace("Intercept", "Intercept"))

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
            for feat in geo_json["features"]:
                name_cn = feat["properties"]["name_cn"]
                en_label = cn_to_en.get(name_cn)
                if en_label is None:
                    continue
                pos = LABEL_POSITIONS.get(en_label)
                if pos is not None:
                    label_data.append({"lon": pos[0], "lat": pos[1], "label": en_label})
            df_labels = pd.DataFrame(label_data)

            fig_m = go.Figure()

            fig_m.add_trace(go.Choropleth(
                geojson=geo_json,
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
                        font=dict(size=20, family="Arial", color="#111"),
                        side="right",
                    ),
                    thickness=30,
                    len=0.78,
                    x=0.87,
                    y=0.5,
                    tickfont=dict(size=16, family="Arial", color="#222"),
                    outlinewidth=0,
                    borderwidth=0,
                    ticklen=5,
                    tickwidth=1,
                    xpad=4,
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
                textfont=dict(size=12, family="Arial", color="#000000"),
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
                height=550,
                width=820,
                margin={"r": 5, "t": 50, "l": 5, "b": 10},
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(family="Arial", size=11),
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
        if st.button("Discover Evolution Paths", use_container_width=True,
                     disabled=st.session_state.path_raw_df is None):
            detail_df, stat_p = analyze_paths(
                st.session_state.path_raw_df, id_t, 'stm', 'Efficiency', ex_v, p_thresh, freq_thresh)
            st.session_state.path_matrix_df = detail_df
            st.session_state.path_res_df = stat_p
            update_ai_context()
        if st.session_state.path_res_df is not None:
            st.success("Evolution Complete")
            st.download_button("Download Evolution Matrix",
                               st.session_state.path_matrix_df.to_csv(index=False).encode('utf-8-sig'),
                               "Province_Path_Matrix.csv", use_container_width=True)
            st.download_button("Download Path Statistics",
                               st.session_state.path_res_df.to_csv(index=False).encode('utf-8-sig'),
                               "Path_Stats_Summary.csv", use_container_width=True)

# ---- Sankey: legend-based, clean nodes ----
if st.session_state.path_matrix_df is not None:
    st.divider()
    st.header("Sankey diagram")

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
                    legendgroup=code, showlegend=True, name=f"{code}: {to_english(full_name)}",
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

    if st.session_state.get("need_ai_response"):
        user_q = st.session_state.chat_messages[-1]["content"]
        dynamic_ctx = get_dynamic_data_context(user_q)
        update_ai_context()
        prompt = f"[System: Full Data Report]\n{st.session_state.ai_full_context}\n{dynamic_ctx}\n\n[User Question]\n{user_q}"
        area, full_resp = st.empty(), ""
        payload = {
            "model": ENDPOINT_ID, "stream": True,
            "messages": [
                {"role": "system", "content": "You are an academic research assistant. Answer questions based on the provided data and your knowledge. Use standard markdown. Leave blank lines before and after tables."},
                {"role": "user", "content": prompt}
            ]
        }
        try:
            r = requests.post(
                "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
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
