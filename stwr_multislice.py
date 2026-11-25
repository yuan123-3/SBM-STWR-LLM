"""Correct multi-slice STWR implementation (separate module; does NOT modify core.py).

Why this module
---------------
core.py::run_stwr_full_process builds STWR with a SINGLE time slice
(coords_list=[coords], tick_nums=1), so the temporal kernel never acts and
"STWR" collapses to yearly GWR. This module accumulates MULTIPLE yearly slices
per target year, so temporal weighting is actually in effect.

Key API facts (verified against stwr 2.1.0):
- STWR(coords_list, y_list, X_list, tick_times_intervel, sita, gwr_bw0,
       tick_nums=k, alpha=a, spherical=..., recorded=1)
  * slices must be ordered oldest -> newest (latest = last element)
  * tick_nums = number of accumulated slices
  * tick_times_intervel = intervals between adjacent slices (length >= tick_nums)
  * fit() returns params/tvalues/localR2/resid for the LATEST slice only (n_last x k)
- Sel_Spt_BW(...).search() returns (alpha, sita, bticks, bw0)

Usage
-----
    from stwr_multislice import run_stwr_multislice
    res = run_stwr_multislice(df, id_col, time_col, lat_col, lon_col,
                              y_col, x_cols, history="all", search=True)
    res["yearly"][2022]      # target-year STWR params/t/p/z/localR2/resid
    res["diagnostics"]["AICc"]  # ols/gwr/stwr comparison
    res["path_df"]           # consolidated Coef_/Pval_ (same layout as core)
"""
import io
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from stwr.gwr import GWR, STWR
from stwr.sel_bw import Sel_BW, Sel_Spt_BW


# ------------------------------------------------------------------ helpers
def _standardize(df, cols):
    df = df.copy()
    for c in cols:
        stdv = df[c].std()
        if pd.notna(stdv) and stdv > 1e-12:
            df[c] = (df[c] - df[c].mean()) / stdv
        else:
            df[c] = df[c] - df[c].mean()
    return df


def _build_slices(df, time_col, id_col, lat_col, lon_col, y_col, x_cols, years_used):
    """Return oldest->newest lists of coords/y/X for the given years."""
    coords_list, y_list, X_list = [], [], []
    for t in years_used:
        d = df[df[time_col] == t]
        if d.empty:
            continue
        coords_list.append(d[[lon_col, lat_col]].to_numpy(float))
        y_list.append(d[[y_col]].to_numpy(float))
        X_list.append(np.hstack([np.ones((len(d), 1)),
                                 d[list(x_cols)].to_numpy(float) if x_cols else np.empty((len(d), 0))]))
    return coords_list, y_list, X_list


def _fit_ols(y_v, X_const):
    ols = sm.OLS(y_v, X_const).fit()
    return ols


def _fit_gwr(coords, y_v, X_const, spherical):
    sel = Sel_BW(coords, y_v, X_const, spherical=spherical)
    bw = sel.search()
    gwr = GWR(coords, y_v, X_const, bw, spherical=spherical).fit()
    return gwr, bw


# ------------------------------------------------------------------ core multi-slice
def run_stwr_multislice(df, id_col, time_col, lat_col, lon_col, y_col, x_cols,
                        history="all", standardize=True, spherical=True,
                        search=True, sita_fixed=0.5, bw0_fixed=50,
                        with_ols_gwr=True, log=None):
    """Fit STWR with accumulated yearly slices.

    history: "all" (use every year up to t) or an int (rolling window length).
    Returns a dict with:
      yearly[year] -> DataFrame(index aligned to that year's rows)
                      columns: id, lon, lat, Intercept, Coef_<var>..., Pval_<var>...
      diagnostics  -> {"AICc","R2","RSS","Sigma"} each a DataFrame [ols,gwr,stwr,date]
      path_df      -> consolidated [id, time, y, Coef_*, Pval_*]
      logs         -> list of strings
    """
    logs = []

    def _log(msg):
        logs.append(msg)
        if log:
            log(msg)

    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    for c in [lat_col, lon_col, y_col] + list(x_cols):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    need = [time_col, id_col, lat_col, lon_col, y_col] + list(x_cols)
    df = df.dropna(subset=[c for c in need if c in df.columns]).copy()
    df = df.sort_values([time_col, id_col])

    if standardize:
        df = _standardize(df, [y_col] + list(x_cols))

    times = sorted(df[time_col].unique())
    var_names = ["Intercept"] + list(x_cols)
    yearly, diag = {}, {"AICc": [], "R2": [], "RSS": [], "Sigma": []}
    path_rows = []

    for idx_t, t in enumerate(times):
        if history == "all":
            years_used = times[: idx_t + 1]
        else:
            years_used = times[max(0, idx_t + 1 - int(history)): idx_t + 1]

        coords_list, y_list, X_list = _build_slices(
            df, time_col, id_col, lat_col, lon_col, y_col, x_cols, years_used)
        if not coords_list:
            continue
        k = len(coords_list)
        d_t = df[df[time_col] == t]
        n_last = len(d_t)
        coords = coords_list[-1]
        y_v = y_list[-1]
        X_const = X_list[-1]

        # intervals between adjacent slices (annual data -> 1.0); length k
        ticks = [float(years_used[i + 1] - years_used[i]) for i in range(k - 1)]
        tick_times_intervel = (ticks + [1.0]) if ticks else [1.0]
        if len(tick_times_intervel) < k:
            tick_times_intervel = [1.0] * k

        # ---- STWR (multi-slice) ----
        aic_s = sig_s = rss_s = np.nan
        st_p = np.full((n_last, len(var_names)), np.nan)
        st_t = np.full((n_last, len(var_names)), np.nan)
        st_pv = np.full((n_last, len(var_names)), np.nan)
        st_z = np.full((n_last, len(var_names)), np.nan)
        st_resid = np.full(n_last, np.nan)
        st_r2 = np.full(n_last, np.nan)
        sita_used = bw0_used = alpha_used = np.nan
        try:
            if search and k > 1:
                try:
                    sel = Sel_Spt_BW(coords_list, y_list, X_list, tick_times_intervel,
                                     spherical=spherical)
                    alpha_used, sita_used, _bticks, bw0_used = sel.search()
                except Exception as e_sel:
                    _log(f"[{t}] bandwidth search failed ({e_sel}); using fixed sita/bw")
                    alpha_used, sita_used, bw0_used = 0.5, sita_fixed, bw0_fixed
            else:
                alpha_used, sita_used, bw0_used = 0.5, sita_fixed, bw0_fixed
            model = STWR(coords_list, y_list, X_list, tick_times_intervel,
                         sita_used, bw0_used, tick_nums=k, alpha=alpha_used,
                         spherical=spherical, recorded=1)
            res = model.fit()
            st_p = np.asarray(res.params)[:n_last]
            st_t = np.asarray(getattr(res, "tvalues", st_p))[:n_last]
            st_r2 = np.ravel(getattr(res, "localR2", st_r2))[:n_last]
            st_resid = np.ravel(getattr(res, "resid_response", st_resid))[:n_last]
            st_pv = 2 * (1 - stats.t.cdf(np.abs(st_t),
                                         getattr(res, "df_model", max(n_last - len(var_names), 1))))
            with np.errstate(all="ignore"):
                mu = np.nanmean(st_p, axis=0)
                sd = np.nanstd(st_p, axis=0)
                st_z = (st_p - mu) / np.where(sd == 0, 1, sd)
            aic_s = getattr(res, "aicc", np.nan)
            sig_s = np.sqrt(getattr(res, "sigma2", np.nan)) if hasattr(res, "sigma2") else np.nan
            rss_s = float(np.nansum(st_resid ** 2)) if np.size(st_resid) else np.nan
            _log(f"[{t}] STWR slices={k} alpha={alpha_used:.3f} sita={sita_used:.3f} bw0={bw0_used:.2f} R2~{np.nanmean(st_r2):.4f}")
        except Exception as e:
            _log(f"[{t}] STWR failed: {e}")
            if k == 1:
                raise  # single-slice at the first year is GWR-like and should work

        base = pd.DataFrame({"id": d_t[id_col].values, "lon": coords[:, 0], "lat": coords[:, 1]})
        params_df = pd.concat([base, pd.DataFrame(st_p, columns=var_names)], axis=1)
        yearly[t] = pd.concat([params_df,
                               pd.DataFrame(st_pv, columns=["Pval_" + v for v in var_names])], axis=1)

        d_pt = pd.DataFrame({id_col: d_t[id_col].values, time_col: t, y_col: y_v.flatten()})
        for i, nm in enumerate(var_names):
            d_pt["Coef_" + nm] = st_p[:, i]
            d_pt["Pval_" + nm] = st_pv[:, i]
        path_rows.append(d_pt)

        # ---- OLS / GWR for comparison ----
        aic_o = r2_o = rss_o = sig_o = np.nan
        aic_g = r2_g = rss_g = sig_g = np.nan
        if with_ols_gwr:
            try:
                ols = _fit_ols(y_v, X_const)
                aic_o, r2_o, rss_o, sig_o = ols.aic, ols.rsquared, ols.ssr, np.sqrt(ols.scale)
            except Exception as e:
                _log(f"[{t}] OLS failed: {e}")
            try:
                gwr, _bw = _fit_gwr(coords, y_v, X_const, spherical)
                aic_g = getattr(gwr, "aicc", np.nan)
                r2_g = getattr(gwr, "R2", np.nan)
                rss_g = getattr(gwr, "resid_ss", np.nan)
                sig_g = np.sqrt(getattr(gwr, "sigma2", np.nan)) if hasattr(gwr, "sigma2") else np.nan
            except Exception as e:
                _log(f"[{t}] GWR failed: {e}")

        diag["AICc"].append([aic_o, aic_g, aic_s, t])
        diag["R2"].append([r2_o, r2_g, (np.nanmean(st_r2) if np.size(st_r2) else np.nan), t])
        diag["RSS"].append([rss_o, rss_g, rss_s, t])
        diag["Sigma"].append([sig_o, sig_g, sig_s, t])

    diagnostics = {k: pd.DataFrame(v, columns=["ols", "gwr", "stwr", "date"]) for k, v in diag.items()}
    path_df = pd.concat(path_rows, ignore_index=True) if path_rows else pd.DataFrame()
    return {"yearly": yearly, "diagnostics": diagnostics, "path_df": path_df, "logs": logs}


def export_stage2_zip(res):
    """Build a dict compatible with core.run_stwr_full_process's zip layout."""
    z = {}
    for t, dfp in res["yearly"].items():
        z[f"Stage2_STWR/{t}/stwr_params.csv"] = dfp.filter(regex=r"^(?!Pval_)")
        z[f"Stage2_STWR/{t}/stwr_p-Values.csv"] = dfp.filter(regex=r"^(id|lon|lat|Pval_)")
    for name, dfx in res["diagnostics"].items():
        z[f"Stage2_STWR/{name}.csv"] = dfx
    return z


# ------------------------------------------------------------------ demo / self-test
if __name__ == "__main__":
    rng = np.random.RandomState(0)
    rows = []
    provs = [f"P{i}" for i in range(12)]
    base = rng.uniform(0, 10, (len(provs), 2))
    for yi, y in enumerate([2018, 2019, 2020, 2021]):
        for i, p in enumerate(provs):
            x1 = rng.normal()
            x2 = rng.normal()
            # time-varying effect: slope of x1 changes sign over years
            slope = 2.0 if y <= 2019 else -2.0
            eff = 0.5 + slope * x1 + 1.0 * x2 + 0.3 * yi + rng.normal(scale=0.1)
            rows.append(dict(prov=p, year=y, lon=base[i, 0] + 0.1 * yi, lat=base[i, 1],
                             x1=x1, x2=x2, eff=eff))
    df = pd.DataFrame(rows)
    res = run_stwr_multislice(df, "prov", "year", "lat", "lon", "eff", ["x1", "x2"],
                              history="all", search=False, sita_fixed=0.5, bw0_fixed=8,
                              standardize=True, log=print)
    print("\nYears fitted:", sorted(res["yearly"].keys()))
    print("path_df shape:", res["path_df"].shape)
    print("AICc table:\n", res["diagnostics"]["AICc"].to_string(index=False))
    for y in [2020, 2021]:
        d = res["yearly"][y]
        print(f"\n{y} STWR x1 coefficient range:", round(d["x1"].min(), 3), "to", round(d["x1"].max(), 3),
              "| Pval_x1 mean:", round(d["Pval_x1"].mean(), 4))
