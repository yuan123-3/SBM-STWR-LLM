import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import time
import io
import traceback
import zipfile
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import re
import plotly.express as px

# Statistical model library
import statsmodels.api as sm

# STWR libraries
from stwr.gwr import GWR, STWR
from stwr.sel_bw import Sel_BW, Sel_Spt_BW


# --- Robust file reader ---
def robust_load_data(uploaded_file):
    if uploaded_file is None:
        return None, "No file uploaded"

    filename = uploaded_file.name
    df = None
    errors = []

    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        try:
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            return None, str(e)
    else:
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb18030', 'ansi', 'latin1']
        for enc in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc)
                if not df.empty and len(df.columns) > 0:
                    break
            except Exception as e:
                errors.append(f"{enc}: {e}")
                continue
        if df is None:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='gb18030', engine='python')
            except Exception as e:
                return None, f"Read failed. Details: {errors}"

    if df is not None:
        df.columns = df.columns.astype(str).str.strip()
        df = df.dropna(how='all')
        return df, None
    else:
        return None, "Unknown error"


# --- Safe slicer ---
def safe_slice(arr, start, end, expected_shape, fill_val=np.nan):
    """Safe slicing: prevents out-of-bounds or length mismatch crashes."""
    if isinstance(expected_shape, tuple):
        target_rows = expected_shape[0]
    else:
        target_rows = expected_shape

    if arr is None:
        if isinstance(expected_shape, tuple):
            return np.full(expected_shape, fill_val)
        else:
            return np.full(target_rows, fill_val)

    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    total_len = len(arr)
    if start >= total_len:
        if isinstance(expected_shape, tuple):
            return np.full(expected_shape, fill_val)
        else:
            return np.full(target_rows, fill_val)

    slice_res = arr[start:end]
    current_rows = len(slice_res)

    if current_rows < target_rows:
        if isinstance(expected_shape, tuple):
            cols = expected_shape[1]
            padded = np.full(expected_shape, fill_val)
            min_cols = min(cols, slice_res.shape[1] if slice_res.ndim > 1 else 1)
            if slice_res.ndim > 1:
                padded[:current_rows, :min_cols] = slice_res[:, :min_cols]
            else:
                padded[:current_rows, 0] = slice_res
            return padded
        else:
            padded = np.full(target_rows, fill_val)
            padded[:current_rows] = slice_res
            return padded

    return slice_res


# ==========================================
# 1. SBM Algorithm (Slacks-Based Measure)
# ==========================================
def solve_sbm_undesirable(X, Yg, Yb, rts='VRS'):
    n = X.shape[1]
    m = X.shape[0]
    s1 = Yg.shape[0]
    s2 = Yb.shape[0]
    scores = []

    def normalize_matrix(mat):
        min_val = np.min(mat, axis=1, keepdims=True)
        max_val = np.max(mat, axis=1, keepdims=True)
        denom = max_val - min_val
        denom[denom == 0] = 1
        return 0.1 + 0.9 * (mat - min_val) / denom

    X_norm = normalize_matrix(X)
    Yg_norm = normalize_matrix(Yg)
    Yb_norm = normalize_matrix(Yb)

    progress_bar = st.progress(0)
    status_text = st.empty()

    for k in range(n):
        if k % 10 == 0 or k == n - 1:
            progress_bar.progress(int((k + 1) / n * 100))
            status_text.text(f"SBM Progress: {k + 1}/{n}")

        tau = cp.Variable(nonneg=True)
        lam = cp.Variable(n, nonneg=True)
        s_minus = cp.Variable(m, nonneg=True)
        s_good = cp.Variable(s1, nonneg=True)
        s_bad = cp.Variable(s2, nonneg=True)

        x_k = X_norm[:, k]
        y_g_k = Yg_norm[:, k]
        y_b_k = Yb_norm[:, k]

        obj = cp.Minimize(tau - (1 / m) * cp.sum(cp.multiply(s_minus, 1 / (x_k + 1e-6))))

        cons = [
            tau + (1 / (s1 + s2)) * (cp.sum(cp.multiply(s_good, 1 / (y_g_k + 1e-6))) +
                                     cp.sum(cp.multiply(s_bad, 1 / (y_b_k + 1e-6)))) == 1,
            cp.multiply(tau, x_k) == X_norm @ lam + s_minus,
            cp.multiply(tau, y_g_k) == Yg_norm @ lam - s_good,
            cp.multiply(tau, y_b_k) == Yb_norm @ lam + s_bad
        ]
        if rts == 'VRS': cons.append(cp.sum(lam) == tau)

        prob = cp.Problem(obj, cons)
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
        except:
            try:
                prob.solve(solver=cp.SCS, verbose=False)
            except:
                pass

        val = prob.value
        if val is None:
            scores.append(np.nan)
        else:
            if val > 1.00001: val = 1.0
            if val < 0: val = 0.0
            scores.append(val)

    progress_bar.empty()
    status_text.empty()
    return np.array(scores)


# ==========================================
# 2. STWR Core Logic
# ==========================================
def get_AICc(model_result, y_data, X_data):
    n = len(y_data)
    k = X_data.shape[1]
    if n == 0 or (n - k - 1) <= 0: return np.inf
    return model_result.aic + (2 * k * (k + 1)) / (n - k - 1)


def manual_ols_summary(ols_result, var_names):
    try:
        summary = "--- OLS Manual Summary ---\n"
        summary += f"AIC: {ols_result.aic:.4f}\nBIC: {ols_result.bic:.4f}\nR2: {ols_result.rsquared:.4f}\n\nCoefficients:\n"
        params = ols_result.params
        tvalues = ols_result.tvalues
        pvalues = ols_result.pvalues

        if hasattr(params, 'values'): params = params.values
        if hasattr(tvalues, 'values'): tvalues = tvalues.values
        if hasattr(pvalues, 'values'): pvalues = pvalues.values

        for i, name in enumerate(var_names):
            if i < len(params):
                line = f"{name:<15}: {params[i]:.4f} (t: {tvalues[i]:.4f}, p: {pvalues[i]:.4f})"
                summary += line + "\n"
        return summary
    except:
        return "OLS Summary Error"


def run_stwr_full_process(df, id_col, time_col, lat_col, lon_col, y_col, x_cols):
    """
    Year-by-year independent OLS, GWR, STWR fitting with per-year bandwidth search.
    """
    logs = []
    def log(msg):
        logs.append(msg)

    zip_content_dict = {}
    path_analysis_data = []

    try:
        log("Preparing data: numeric conversion & sorting...")
        df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
        df = df.sort_values(by=[time_col, id_col]).copy()

        required_cols = [lat_col, lon_col, y_col] + list(x_cols)
        for c in required_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df_clean = df.dropna(subset=required_cols).copy()
        if df_clean.empty:
            log("Input data has all NaN on key columns. Aborting.")
            return None, logs, None

        log("Standardizing (y and X)...")
        for c in [y_col] + list(x_cols):
            if c not in df_clean.columns:
                continue
            stdv = df_clean[c].std()
            if pd.notna(stdv) and stdv > 1e-12:
                df_clean[c] = (df_clean[c] - df_clean[c].mean()) / stdv
            else:
                df_clean[c] = df_clean[c] - df_clean[c].mean()

        unique_times = sorted(df_clean[time_col].unique())
        if len(unique_times) == 0:
            log("No valid time points. Aborting.")
            return None, logs, None
        log(f"Time points: {unique_times}")

        AICc_out, R2_out, RSS_out, Sigma_out = [], [], [], []
        var_names = ['Intercept'] + list(x_cols)

        for t in unique_times:
            log(f"Processing year: {t}")
            d_t = df_clean[df_clean[time_col] == t].copy()
            if d_t.empty:
                log(f"Year {t} has no data, skipping")
                continue

            coords = d_t[[lon_col, lat_col]].to_numpy(float)
            y_v = d_t[[y_col]].to_numpy(float)
            X_v = d_t[list(x_cols)].to_numpy(float) if len(x_cols) > 0 else np.empty((len(d_t), 0))
            X_const = np.hstack([np.ones((len(d_t), 1)), X_v])
            n = len(d_t)
            fld = f"{t}"
            base = pd.DataFrame({'Ename': d_t[id_col].values, 'lon': coords[:, 0], 'lat': coords[:, 1]})

            # ----- OLS -----
            try:
                ols = sm.OLS(y_v, X_const).fit()
                ols_summary = manual_ols_summary(ols, var_names)
                aic_o = ols.aic
                r2_o = ols.rsquared
                rss_o = ols.ssr
                sig_o = np.sqrt(ols.scale)
                ols_params = np.tile(ols.params, (n, 1))
                ols_tvals = np.tile(ols.tvalues, (n, 1))
                ols_pvals = np.tile(ols.pvalues, (n, 1))
                ols_resid = np.array(ols.resid).flatten()
            except Exception as e:
                log(f"OLS failed for year {t}: {e}")
                ols_summary = "OLS Failed"
                aic_o = r2_o = rss_o = sig_o = np.nan
                ols_params = ols_tvals = ols_pvals = np.full((n, len(var_names)), np.nan)
                ols_resid = np.full(n, np.nan)

            # ----- GWR -----
            try:
                sel_g = Sel_BW(coords, y_v, X_const, spherical=True)
                bw_g = sel_g.search()
                gwr = GWR(coords, y_v, X_const, bw_g, spherical=True).fit()
                aic_g = getattr(gwr, 'aicc', np.nan)
                r2_g = getattr(gwr, 'R2', np.nan)
                rss_g = getattr(gwr, 'resid_ss', np.nan)
                sig_g = np.sqrt(getattr(gwr, 'sigma2', np.nan)) if hasattr(gwr, 'sigma2') else np.nan
                gwr_p = getattr(gwr, 'params', np.full((n, len(var_names)), np.nan))
                gwr_t = getattr(gwr, 'tvalues', np.full((n, len(var_names)), np.nan))
                gwr_pv = 2 * (1 - stats.t.cdf(np.abs(gwr_t), getattr(gwr, 'df_model', n)))
                g_mean = np.nanmean(gwr_p, axis=0) if np.size(gwr_p) else np.full(len(var_names), np.nan)
                g_std = np.nanstd(gwr_p, axis=0) if np.size(gwr_p) else np.full(len(var_names), np.nan)
                with np.errstate(all='ignore'):
                    gwr_z = (gwr_p - g_mean) / np.where(g_std == 0, 1, g_std)
                gwr_resid = getattr(gwr, 'resid_response', np.full((n,), np.nan))
                gwr_lr2 = getattr(gwr, 'localR2', np.full((n,), np.nan)).flatten()
            except Exception as e:
                log(f"GWR failed for year {t}: {e}")
                aic_g = r2_g = rss_g = sig_g = np.nan
                gwr_p = gwr_t = gwr_pv = gwr_z = np.full((n, len(var_names)), np.nan)
                gwr_resid = np.full(n, np.nan)
                gwr_lr2 = np.full(n, np.nan)

            # ----- STWR -----
            st_p = np.full((n, len(var_names)), np.nan)
            st_t = np.full((n, len(var_names)), np.nan)
            st_pv = np.full((n, len(var_names)), np.nan)
            st_z = np.full((n, len(var_names)), np.nan)
            st_resid = np.full(n, np.nan)
            st_localR2 = np.full(n, np.nan)
            aic_s = np.nan
            sig_s = np.nan
            rss_s = np.nan

            coords_list = [coords]
            y_list = [y_v]
            X_list = [X_const]
            dt_list = [0.0]

            tried_stwr = False
            try:
                sel_st = Sel_Spt_BW(coords_list, y_list, X_list, dt_list, spherical=True)
                found = sel_st.search()
                if isinstance(found, (list, tuple)):
                    optalpha = found[0] if len(found) >= 1 else 0.5
                    optsita = found[1] if len(found) >= 2 else 1.0
                    opt_bw_local = found[3] if len(found) >= 4 else (found[-1] if len(found) >= 1 else 50)
                else:
                    optalpha, optsita, opt_bw_local = 0.5, 1.0, 50
                model_st = STWR(coords_list, y_list, X_list, dt_list, optsita, opt_bw_local, alpha=optalpha,
                                spherical=True, recorded=1)
                res_st = model_st.fit()
                tried_stwr = True
            except Exception as e_st:
                try:
                    sel_sp = Sel_BW(coords, y_v, X_const, spherical=True)
                    bw_sp = sel_sp.search()
                    optalpha, optsita, opt_bw_local = 0.5, 1.0, bw_sp
                    model_st = STWR(coords_list, y_list, X_list, dt_list, optsita, opt_bw_local, alpha=optalpha,
                                    spherical=True, recorded=1)
                    res_st = model_st.fit()
                    tried_stwr = True
                    log(f"STWR (spatial fallback) used for year {t} due to: {e_st}")
                except Exception as e2:
                    log(f"STWR failed for year {t}: {e_st}; fallback also failed: {e2}")
                    tried_stwr = False

            if tried_stwr:
                total_params = getattr(res_st, 'params', None)
                total_tvals = getattr(res_st, 'tvalues', None)
                total_localR2 = getattr(res_st, 'localR2', None)
                total_resid = getattr(res_st, 'resid_response', None)
                try:
                    st_p = np.array(total_params) if total_params is not None else st_p
                except:
                    st_p = st_p
                try:
                    st_t = np.array(total_tvals) if total_tvals is not None else st_t
                except:
                    st_t = st_t
                try:
                    st_localR2 = np.ravel(total_localR2) if total_localR2 is not None else st_localR2
                except:
                    st_localR2 = st_localR2
                try:
                    st_resid = np.ravel(total_resid) if total_resid is not None else st_resid
                except:
                    st_resid = st_resid
                try:
                    st_pv = 2 * (1 - stats.t.cdf(np.abs(st_t), getattr(res_st, 'df_model', max(n - len(var_names), 1))))
                except:
                    st_pv = st_pv
                try:
                    mean_p = np.nanmean(st_p, axis=0)
                    std_p = np.nanstd(st_p, axis=0)
                    with np.errstate(all='ignore'):
                        st_z = (st_p - mean_p) / np.where(std_p == 0, 1, std_p)
                except:
                    st_z = st_z
                aic_s = getattr(res_st, 'aicc', np.nan)
                sig_s = np.sqrt(getattr(res_st, 'sigma2', np.nan)) if hasattr(res_st, 'sigma2') else np.nan
                try:
                    if np.size(st_resid) and not np.all(np.isnan(st_resid)):
                        rss_s = np.sum(np.ravel(st_resid) ** 2)
                    else:
                        if np.size(st_p):
                            mean_coef = np.nanmean(st_p, axis=0)
                            y_pred = X_const @ mean_coef.reshape(-1, 1)
                            rss_s = np.sum((y_v.flatten() - y_pred.flatten()) ** 2)
                        else:
                            rss_s = np.nan
                except:
                    rss_s = np.nan
            else:
                st_p = np.full((n, len(var_names)), np.nan)
                st_t = np.full((n, len(var_names)), np.nan)
                st_pv = np.full((n, len(var_names)), np.nan)
                st_z = np.full((n, len(var_names)), np.nan)
                st_resid = np.full(n, np.nan)
                st_localR2 = np.full(n, np.nan)
                aic_s = np.nan
                sig_s = np.nan
                rss_s = np.nan

            y_true = y_v.flatten()
            tss_t = np.sum((y_true - np.mean(y_true)) ** 2)
            try:
                stwr_r2_yearly = 1 - (rss_s / tss_t) if (pd.notna(tss_t) and tss_t != 0) else np.nan
            except:
                stwr_r2_yearly = np.nan

            # Write yearly outputs
            zip_content_dict[f"Stage2_STWR/{fld}/stwr_params.csv"] = pd.concat(
                [base, pd.DataFrame(st_p, columns=var_names)], axis=1)
            zip_content_dict[f"Stage2_STWR/{fld}/stwr_t-Values.csv"] = pd.concat(
                [base, pd.DataFrame(st_t, columns=var_names)], axis=1)
            zip_content_dict[f"Stage2_STWR/{fld}/stwr_p-Values.csv"] = pd.concat(
                [base, pd.DataFrame(st_pv, columns=var_names)], axis=1)
            zip_content_dict[f"Stage2_STWR/{fld}/stwr_z-Scores.csv"] = pd.concat(
                [base, pd.DataFrame(st_z, columns=var_names)], axis=1)
            zip_content_dict[f"Stage2_STWR/{fld}/stwr_residuals.csv"] = pd.concat(
                [base, pd.DataFrame({'resid': np.ravel(st_resid), 'localR2': np.ravel(st_localR2)})], axis=1)
            zip_content_dict[f"Stage2_STWR/{fld}/stwr_R2.csv"] = pd.concat(
                [base, pd.DataFrame({'R2': np.ravel(st_localR2)})], axis=1)
            zip_content_dict[f"Stage2_STWR/{fld}/gwr_params.csv"] = pd.concat(
                [base, pd.DataFrame(gwr_p, columns=var_names)], axis=1)
            zip_content_dict[f"Stage2_STWR/{fld}/gwr_t-Values.csv"] = pd.concat(
                [base, pd.DataFrame(gwr_t, columns=var_names)], axis=1)
            zip_content_dict[f"Stage2_STWR/{fld}/gwr_p-Values.csv"] = pd.concat(
                [base, pd.DataFrame(gwr_pv, columns=var_names)], axis=1)
            zip_content_dict[f"Stage2_STWR/{fld}/gwr_z-Scores.csv"] = pd.concat(
                [base, pd.DataFrame(gwr_z, columns=var_names)], axis=1)
            zip_content_dict[f"Stage2_STWR/{fld}/gwr_residuals.csv"] = pd.concat(
                [base, pd.DataFrame({'resid': np.ravel(gwr_resid), 'localR2': np.ravel(gwr_lr2)})], axis=1)
            zip_content_dict[f"Stage2_STWR/{fld}/gwr_R2.csv"] = pd.concat(
                [base, pd.DataFrame({'R2': np.ravel(gwr_lr2)})], axis=1)
            zip_content_dict[f"Stage2_STWR/{fld}/ols_params.csv"] = pd.concat(
                [base, pd.DataFrame(ols_params, columns=var_names)], axis=1)
            zip_content_dict[f"Stage2_STWR/{fld}/ols_t-Values.csv"] = pd.concat(
                [base, pd.DataFrame(ols_tvals, columns=var_names)], axis=1)
            zip_content_dict[f"Stage2_STWR/{fld}/ols_p-Values.csv"] = pd.concat(
                [base, pd.DataFrame(ols_pvals, columns=var_names)], axis=1)
            zip_content_dict[f"Stage2_STWR/{fld}/ols_residuals.csv"] = pd.concat(
                [base, pd.DataFrame({'resid': np.ravel(ols_resid)})], axis=1)

            sum_str = f"=== YEARLY MODEL SUMMARY ({t}) ===\n\n[OLS]\n{ols_summary}\n\n[GWR]\n"
            try:
                sum_str += f"AICc={aic_g}, R2={r2_g}\n"
            except:
                sum_str += "GWR summary unavailable\n"
            sum_str += "\n\n[STWR]\n"
            try:
                sum_str += f"AICc={aic_s}\n"
            except:
                sum_str += "STWR summary unavailable\n"
            zip_content_dict[f"Stage2_STWR/{fld}/summary.txt"] = sum_str

            AICc_out.append([aic_o, aic_g, aic_s, t])
            R2_out.append([r2_o, r2_g, stwr_r2_yearly, t])
            RSS_out.append([rss_o, rss_g, rss_s, t])
            Sigma_out.append([sig_o, sig_g, sig_s, t])

            d_pt = pd.DataFrame({id_col: d_t[id_col].values, time_col: t, y_col: y_v.flatten()})
            for i, nm in enumerate(var_names):
                coef_col = st_p[:, i] if i < st_p.shape[1] else np.full(n, np.nan)
                pval_col = st_pv[:, i] if i < st_pv.shape[1] else np.full(n, np.nan)
                d_pt[f'Coef_{nm}'] = np.ravel(coef_col)
                d_pt[f'Pval_{nm}'] = np.ravel(pval_col)
            path_analysis_data.append(d_pt)

        zip_content_dict["Stage2_STWR/AICc.csv"] = pd.DataFrame(AICc_out, columns=['ols', 'gwr', 'stwr', 'date'])
        zip_content_dict["Stage2_STWR/R2.csv"] = pd.DataFrame(R2_out, columns=['ols', 'gwr', 'stwr', 'date'])
        zip_content_dict["Stage2_STWR/RSS.csv"] = pd.DataFrame(RSS_out, columns=['ols', 'gwr', 'stwr', 'date'])
        zip_content_dict["Stage2_STWR/Sigma.csv"] = pd.DataFrame(Sigma_out, columns=['ols', 'gwr', 'stwr', 'date'])

        full_path_df = pd.concat(path_analysis_data, ignore_index=True) if path_analysis_data else pd.DataFrame()
        log("STWR yearly computation complete.")
        return zip_content_dict, logs, full_path_df

    except Exception as e:
        log(f"STWR execution error: {e}")
        log(traceback.format_exc())
        raise e


# ==========================================
# 3. Path Evolution Analysis
# ==========================================
def analyze_paths(stwr_data, id_col, time_col, y_col, x_cols, p_threshold, freq_threshold):
    if stwr_data.empty: return pd.DataFrame(), pd.DataFrame()

    cols_chk = [c for c in x_cols if c != 'Intercept']
    temp_status_list = []

    for uid, grp in stwr_data.groupby(id_col):
        grp = grp.sort_values(by=time_col)
        for idx, row in grp.iterrows():
            curr_yr, curr_y = row[time_col], row[y_col]
            bst, max_abs, dr = "None", -1, ""
            for v in cols_chk:
                c_val = row.get(f'Coef_{v}', 0)
                p_val = row.get(f'Pval_{v}', 1)
                if pd.isna(c_val) or pd.isna(p_val): continue
                if p_val < p_threshold:
                    if abs(c_val) > max_abs:
                        max_abs = abs(c_val)
                        bst = v
                        dr = "+" if c_val > 0 else "-"
            status = f"{bst}{dr}" if bst != "None" else "Non-Sig"
            temp_status_list.append({'ID': uid, 'Year': int(curr_yr), 'Factor': status, 'Y': curr_y})

    df_status = pd.DataFrame(temp_status_list)
    if df_status.empty: return pd.DataFrame(), pd.DataFrame()

    all_years = sorted(df_status['Year'].unique())
    all_ids = df_status['ID'].unique()
    matrix_rows = []
    long_format_for_stats = []

    for uid in all_ids:
        city_data = df_status[df_status['ID'] == uid].set_index('Year')
        row_dict = {'ID': uid}
        for i in range(len(all_years) - 1):
            t_start = all_years[i]
            t_end = all_years[i + 1]
            col_name_path = f"Path_{t_start}-{t_end}"
            col_name_eff = f"DeltaEff_{t_start}-{t_end}"
            if t_start in city_data.index and t_end in city_data.index:
                start_fac = city_data.loc[t_start, 'Factor']
                end_fac = city_data.loc[t_end, 'Factor']
                y_start = city_data.loc[t_start, 'Y']
                y_end = city_data.loc[t_end, 'Y']
                path_str = f"{start_fac} -> {end_fac}"
                delta_y = y_end - y_start
                row_dict[col_name_path] = path_str
                row_dict[col_name_eff] = round(delta_y, 4)
                long_format_for_stats.append({'ID': uid, 'Path': path_str, 'Delta_Y': delta_y})
            else:
                row_dict[col_name_path] = None
                row_dict[col_name_eff] = None
        matrix_rows.append(row_dict)

    df_wide_result = pd.DataFrame(matrix_rows)
    df_long = pd.DataFrame(long_format_for_stats)
    if not df_long.empty:
        stats = df_long.groupby('Path').agg(
            Freq=('ID', 'count'), Avg_Eff=('Delta_Y', 'mean')).reset_index()
        th = max(1, int(freq_threshold))
        stats = stats[stats['Freq'] >= th].sort_values('Avg_Eff', ascending=False)
    else:
        stats = pd.DataFrame()

    return df_wide_result, stats


def analyze_path_geography(path_matrix_df, id_col=None):
    """
    Analyze the geographic distribution of evolution paths.
    
    Returns:
        1. path_by_region: DataFrame with each region's historical paths and effects
        2. path_frequency: DataFrame with path frequency statistics across all regions
        3. time_path_stats: DataFrame with path statistics by time period
        4. region_query_result: Function to query specific region's history
    """
    if path_matrix_df is None or (hasattr(path_matrix_df, 'empty') and path_matrix_df.empty):
        return None, None, None, None
    
    if not isinstance(path_matrix_df, pd.DataFrame):
        return None, None, None, None
    
    # Detect the actual ID column: prefer 'ID', then id_col parameter, then first non-Path/non-DeltaEff column
    actual_id_col = None
    if 'ID' in path_matrix_df.columns:
        actual_id_col = 'ID'
    elif id_col and id_col in path_matrix_df.columns:
        actual_id_col = id_col
    else:
        for c in path_matrix_df.columns:
            if not c.startswith('Path_') and not c.startswith('DeltaEff_'):
                actual_id_col = c
                break
    
    if actual_id_col is None:
        return None, None, None, None
    
    path_cols = [c for c in path_matrix_df.columns if c.startswith('Path_')]
    eff_cols = [c for c in path_matrix_df.columns if c.startswith('DeltaEff_')]
    
    if not path_cols:
        return None, None, None, None
    
    # 1. Analyze each region's historical paths
    region_records = []
    for idx, row in path_matrix_df.iterrows():
        try:
            region = row[actual_id_col]
        except (KeyError, TypeError):
            continue
        for i, path_col in enumerate(path_cols):
            if path_col in path_matrix_df.columns:
                path_val = row.get(path_col, None)
                eff_val = row.get(eff_cols[i], None) if i < len(eff_cols) else None
                time_period = path_col.replace('Path_', '')
                if pd.notna(path_val) and str(path_val) != 'None -> None':
                    region_records.append({
                        'Region': region,
                        'TimePeriod': time_period,
                        'Path': path_val,
                        'DeltaEfficiency': eff_val
                    })
    
    path_by_region = pd.DataFrame(region_records)
    
    # 2. Calculate path frequency across all regions
    if not path_by_region.empty:
        path_frequency = path_by_region.groupby('Path').agg(
            RegionCount=('Region', 'nunique'),
            TotalOccurrences=('Region', 'count'),
            AvgEfficiency=('DeltaEfficiency', 'mean'),
            MinEfficiency=('DeltaEfficiency', 'min'),
            MaxEfficiency=('DeltaEfficiency', 'max')
        ).reset_index().sort_values('TotalOccurrences', ascending=False)
    else:
        path_frequency = pd.DataFrame()
    
    # 3. Calculate path statistics by time period
    if not path_by_region.empty:
        time_path_stats = path_by_region.groupby(['TimePeriod', 'Path']).agg(
            RegionCount=('Region', 'nunique'),
            TotalOccurrences=('Region', 'count'),
            AvgEfficiency=('DeltaEfficiency', 'mean'),
            Regions=('Region', lambda x: ', '.join(x.unique()[:5]))  # Top 5 regions
        ).reset_index().sort_values(['TimePeriod', 'TotalOccurrences'], ascending=[True, False])
    else:
        time_path_stats = pd.DataFrame()
    
    # 4. Function to query specific region's history
    def query_region(region_name):
        if path_by_region.empty:
            return None
        region_data = path_by_region[path_by_region['Region'] == region_name]
        if region_data.empty:
            return None
        
        # Get most frequent path for this region
        path_counts = region_data['Path'].value_counts()
        most_frequent_path = path_counts.index[0] if not path_counts.empty else None
        most_frequent_count = path_counts.iloc[0] if not path_counts.empty else 0
        
        # Get average efficiency for most frequent path
        if most_frequent_path:
            path_eff = region_data[region_data['Path'] == most_frequent_path]['DeltaEfficiency']
            avg_eff = path_eff.mean() if not path_eff.empty else 0
        else:
            avg_eff = 0
        
        return {
            'region': region_name,
            'total_records': len(region_data),
            'most_frequent_path': most_frequent_path,
            'most_frequent_count': most_frequent_count,
            'avg_efficiency': avg_eff,
            'all_paths': region_data[['TimePeriod', 'Path', 'DeltaEfficiency']].to_dict('records')
        }
    
    return path_by_region, path_frequency, time_path_stats, query_region


def prepare_path_geography_data(path_matrix_df, id_col=None, target_year=None, target_region=None):
    """
    Prepare data for path geography visualization.
    
    Args:
        path_matrix_df: Path matrix DataFrame
        id_col: ID column name (auto-detects if None)
        target_year: Specific time period to analyze (e.g., '2015-2018')
        target_region: Specific region to analyze (e.g., 'Fujian')
    
    Returns:
        Dictionary with visualization data
    """
    if path_matrix_df is None or path_matrix_df.empty:
        return None
    
    # Auto-detect ID column
    if 'ID' in path_matrix_df.columns:
        actual_id_col = 'ID'
    elif id_col and id_col in path_matrix_df.columns:
        actual_id_col = id_col
    else:
        actual_id_col = path_matrix_df.columns[0]
    
    path_cols = [c for c in path_matrix_df.columns if c.startswith('Path_')]
    eff_cols = [c for c in path_matrix_df.columns if c.startswith('DeltaEff_')]
    
    result = {
        'all_regions': [],
        'path_summary': {},
        'time_period_data': {},
        'region_query': {}
    }
    
    # Process all regions
    for _, row in path_matrix_df.iterrows():
        region = row[actual_id_col]
        region_info = {'region': region, 'paths': []}
        
        for i, path_col in enumerate(path_cols):
            path_val = row[path_col]
            eff_val = row[eff_cols[i]] if i < len(eff_cols) else None
            time_period = path_col.replace('Path_', '')
            
            if pd.notna(path_val) and path_val != 'None -> None':
                region_info['paths'].append({
                    'time_period': time_period,
                    'path': path_val,
                    'delta_efficiency': eff_val
                })
        
        result['all_regions'].append(region_info)
    
    # Calculate path summary
    all_paths = []
    for region_info in result['all_regions']:
        for path_info in region_info['paths']:
            all_paths.append(path_info['path'])
    
    if all_paths:
        path_counts = pd.Series(all_paths).value_counts()
        result['path_summary'] = {
            'paths': path_counts.index.tolist(),
            'counts': path_counts.values.tolist(),
            'percentages': (path_counts.values / len(all_paths) * 100).tolist()
        }
    
    # Process specific time period
    if target_year:
        time_data = {'regions': [], 'paths': [], 'effects': []}
        for region_info in result['all_regions']:
            for path_info in region_info['paths']:
                if path_info['time_period'] == target_year:
                    time_data['regions'].append(region_info['region'])
                    time_data['paths'].append(path_info['path'])
                    time_data['effects'].append(path_info['delta_efficiency'])
        result['time_period_data'][target_year] = time_data
    
    # Process specific region
    if target_region:
        for region_info in result['all_regions']:
            if region_info['region'] == target_region:
                result['region_query'][target_region] = region_info['paths']
                break
    
    return result


def prepare_sankey_data(path_matrix_df):
    """
    Prepare Sankey diagram data. Nodes use short codes; a separate legend
    maps codes to factor names for clean journal-style rendering.
    """
    if path_matrix_df is None or path_matrix_df.empty:
        return None, None, None, None, None

    path_cols = [c for c in path_matrix_df.columns if c.startswith('Path_')]
    links_data, nodes_map, node_labels, node_colors = [], {}, [], []

    # --- Collect unique factors ---
    unique_factors_set = set()
    for col in path_cols:
        for p in path_matrix_df[col].dropna().unique():
            if " -> " in str(p):
                parts = [re.sub(r'^\d{2}-\d{2}', '', i).strip() for i in str(p).split(" -> ")]
                unique_factors_set.update(parts)

    sorted_factors = sorted(list(unique_factors_set))

    # --- Macaron palette ---
    macaron_colors = [
        "rgba(168, 216, 234, 0.8)", "rgba(170, 150, 218, 0.8)",
        "rgba(252, 186, 211, 0.8)", "rgba(181, 234, 215, 0.8)",
        "rgba(255, 218, 193, 0.8)", "rgba(226, 240, 203, 0.8)",
        "rgba(213, 196, 240, 0.8)", "rgba(249, 197, 209, 0.8)",
        "rgba(199, 206, 234, 0.8)", "rgba(230, 230, 250, 0.8)",
        "rgba(255, 218, 185, 0.8)", "rgba(152, 251, 152, 0.8)",
    ]
    factor_color_map = {f: macaron_colors[i % len(macaron_colors)] for i, f in enumerate(sorted_factors)}
    # Short codes for nodes (A, B, C, ...)
    from string import ascii_lowercase
    factor_code_map = {}
    legend_map = {}
    for i, f in enumerate(sorted_factors):
        code = f"F{i+1}"
        factor_code_map[f] = code
        legend_map[code] = f

    # --- Extract years ---
    years_set = set()
    for col in path_cols:
        years_set.update(col.replace("Path_", "").split("-"))
    unique_years = sorted(list(years_set))

    # --- Build nodes with short codes ---
    for col in path_cols:
        y_start, y_end = col.replace("Path_", "").split("-")
        counts = path_matrix_df[col].value_counts()
        for path_str, count in counts.items():
            if " -> " not in str(path_str): continue
            clean_p = [re.sub(r'^\d{2}-\d{2}', '', i).strip() for i in str(path_str).split(" -> ")]
            for yr, fac in [(y_start, clean_p[0]), (y_end, clean_p[1])]:
                key = f"{yr}_{fac}"
                if key not in nodes_map:
                    nodes_map[key] = len(node_labels)
                    node_labels.append(factor_code_map.get(fac, fac))
                    node_colors.append(factor_color_map.get(fac, "rgba(200, 200, 200, 0.8)"))
            links_data.append({
                'source': nodes_map[f"{y_start}_{clean_p[0]}"],
                'target': nodes_map[f"{y_end}_{clean_p[1]}"],
                'value': count,
                'label': f"{y_start}-{y_end}: {clean_p[0]} -> {clean_p[1]}"
            })

    return node_labels, pd.DataFrame(links_data), unique_years, node_colors, legend_map


# ==========================================
# 5. Result metadata & validation (read-only helpers for the agent layer)
# ==========================================
def build_result_metadata(model, region_col, time_col, result_df, parameters=None, extra=None):
    """Describe an existing result so the agent can decide whether it is usable."""
    if result_df is None or len(result_df) == 0:
        return {"model": model, "status": "missing", "row_count": 0}
    regions = []
    if region_col in result_df.columns:
        regions = sorted(result_df[region_col].dropna().astype(str).unique().tolist())
    years = []
    if time_col in result_df.columns:
        try:
            years = [int(v) for v in pd.to_numeric(result_df[time_col], errors="coerce").dropna().unique()]
        except Exception:
            years = []
    meta = {
        "model": model,
        "status": "success",
        "region_column": region_col,
        "time_column": time_col,
        "row_count": int(len(result_df)),
        "regions": regions,
        "year_range": [min(years), max(years)] if years else [],
        "parameters": parameters or {},
    }
    if extra:
        meta.update(extra)
    return meta


def validate_result(df, required_columns):
    """Return (ok, message). Used before feeding results into the RAG layer."""
    if df is None or getattr(df, "empty", True):
        return False, "Empty result"
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}"
    try:
        if df[list(required_columns)].isna().any().any():
            return False, "Missing values detected in required columns"
    except Exception:
        pass
    return True, "Valid"
