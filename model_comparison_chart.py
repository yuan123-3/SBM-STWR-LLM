import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ══════════════════════════════════════════════════════════
#  Data from 模型结果.xls — OLS vs GWR vs STWR 2004–2022
# ══════════════════════════════════════════════════════════
years = list(range(2004, 2023))

r2_ols  = [0.25, 0.10, 0.25, 0.24, 0.33, 0.33, 0.35,
           0.31, 0.13, 0.15, 0.25, 0.27, 0.20, 0.33, 0.34, 0.40, 0.50, 0.27, 0.34]
r2_gwr  = [0.38, 0.27, 0.54, 0.51, 0.59, 0.53, 0.58,
           0.46, 0.35, 0.36, 0.48, 0.49, 0.41, 0.52, 0.48, 0.52, 0.64, 0.51, 0.55]
r2_stwr = [0.38, 0.27, 0.62, 0.54, 0.68, 0.61, 0.72,
           0.46, 0.35, 0.74, 0.48, 0.49, 0.41, 0.52, 0.48, 0.52, 0.93, 0.62, 0.81]

rss_ols  = [18.97, 20.98, 14.82, 17.32, 13.57, 12.62, 16.83,
            25.95, 29.20, 29.03, 22.90, 17.46, 27.81, 18.61, 18.11, 18.65, 13.58, 25.84, 22.89]
rss_gwr  = [15.70, 17.06,  9.20, 11.31,  8.38,  8.90, 10.90,
            20.33, 21.91, 21.84, 15.93, 12.20, 20.46, 13.29, 14.24, 14.89,  9.88, 17.38, 15.66]
rss_stwr = [15.70, 17.06,  7.47, 10.50,  6.49,  7.27,  7.30,
            20.33, 21.91,  9.06, 15.93, 12.20, 20.46, 13.29, 14.24, 14.89,  1.99, 13.33,  6.70]

sigma_ols  = [0.85, 0.90, 0.75, 0.82, 0.72, 0.70, 0.80,
              1.00, 1.06, 1.06, 0.94, 0.82, 1.03, 0.85, 0.83, 0.85, 0.72, 1.00, 0.94]
sigma_gwr  = [0.82, 0.85, 0.63, 0.69, 0.59, 0.61, 0.68,
              0.93, 0.97, 0.96, 0.82, 0.72, 0.93, 0.75, 0.78, 0.80, 0.65, 0.86, 0.82]
sigma_stwr = [0.82, 0.85, 0.58, 0.67, 0.54, 0.57, 0.59,
              0.93, 0.97, 0.73, 0.82, 0.72, 0.93, 0.75, 0.78, 0.80, 0.41, 0.79, 0.66]

# ── Pale macaron palette ──
C_OLS  = "#fad4de"   # pale rose
C_GWR  = "#fae0c3"   # pale orange
C_STWR = "#d2ebcc"   # pale sage
EDGE   = "#333333"
LW     = 0.9

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.edgecolor": "#555",
    "xtick.color": "#444",
    "ytick.color": "#444",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

x = np.arange(len(years))
width = 0.25

fig, axes = plt.subplots(3, 1, figsize=(16, 10.5), facecolor="white")

# ────────────────────── R² ──────────────────────
ax = axes[0]
b1 = ax.bar(x - width, r2_ols,  width, color=C_OLS,  edgecolor=EDGE, linewidth=LW, label="OLS",  zorder=2)
b2 = ax.bar(x,          r2_gwr,  width, color=C_GWR,  edgecolor=EDGE, linewidth=LW, label="GWR",  zorder=2)
b3 = ax.bar(x + width,  r2_stwr, width, color=C_STWR, edgecolor=EDGE, linewidth=LW, label="STWR", zorder=2)
ax.set_xticks(x[::2])
ax.set_xticklabels(years[::2], fontsize=11, rotation=0)
ax.set_ylabel("R²", fontsize=12, weight="bold", color="#222")
ax.set_title("R²", fontsize=14, weight="bold", color="#111", pad=8, loc="left")
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax.grid(axis="y", ls="--", alpha=0.3, color="#999", zorder=0)
ax.tick_params(labelsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ────────────────────── RSS ──────────────────────
ax = axes[1]
ax.bar(x - width, rss_ols,  width, color=C_OLS,  edgecolor=EDGE, linewidth=LW, label="OLS",  zorder=2)
ax.bar(x,          rss_gwr,  width, color=C_GWR,  edgecolor=EDGE, linewidth=LW, label="GWR",  zorder=2)
ax.bar(x + width,  rss_stwr, width, color=C_STWR, edgecolor=EDGE, linewidth=LW, label="STWR", zorder=2)
ax.set_xticks(x[::2])
ax.set_xticklabels(years[::2], fontsize=11, rotation=0)
ax.set_ylabel("RSS", fontsize=12, weight="bold", color="#222")
ax.set_title("RSS", fontsize=14, weight="bold", color="#111", pad=8, loc="left")
ax.set_ylim(0, 34)
ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
ax.grid(axis="y", ls="--", alpha=0.3, color="#999", zorder=0)
ax.tick_params(labelsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ────────────────────── Sigma ──────────────────────
ax = axes[2]
ax.bar(x - width, sigma_ols,  width, color=C_OLS,  edgecolor=EDGE, linewidth=LW, label="OLS",  zorder=2)
ax.bar(x,          sigma_gwr,  width, color=C_GWR,  edgecolor=EDGE, linewidth=LW, label="GWR",  zorder=2)
ax.bar(x + width,  sigma_stwr, width, color=C_STWR, edgecolor=EDGE, linewidth=LW, label="STWR", zorder=2)
ax.set_xticks(x[::2])
ax.set_xticklabels(years[::2], fontsize=11, rotation=0)
ax.set_ylabel("Sigma", fontsize=12, weight="bold", color="#222")
ax.set_title("Sigma", fontsize=14, weight="bold", color="#111", pad=8, loc="left")
ax.set_ylim(0, 1.20)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax.grid(axis="y", ls="--", alpha=0.3, color="#999", zorder=0)
ax.tick_params(labelsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Shared legend (horizontal at top) ──
fig.legend([b1, b2, b3], ["OLS", "GWR", "STWR"], loc="upper center",
           ncol=3, frameon=True, fontsize=14, fancybox=False,
           edgecolor="#ccc", facecolor="white",
           bbox_to_anchor=(0.5, 0.96))

fig.suptitle("Comparison of OLS, GWR and STWR models (2004–2022)",
             fontsize=17, weight="bold", color="#111", y=1.03)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.savefig("model_comparison.pdf", bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("Saved: model_comparison.png + model_comparison.pdf")
