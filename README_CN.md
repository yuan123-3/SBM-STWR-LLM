# SBM-STWR 全流程分析系统

基于 Streamlit 的学术科研工具，用于**能源效率评价**与**驱动因素时空演化分析**，将 SBM 效率测度、STWR 时空加权回归、AI 智能问答整合在一个 Web 应用中。

## 工作流程

```
┌─────────────────────────┐    ┌──────────────────────────────────┐    ┌─────────────────────────────┐
│   Step 1: SBM 效率评价    │    │   Step 2: STWR 时空回归          │    │   Step 3: 路径演化分析       │
│                          │ → │                                  │ → │                             │
│   投入 / 期望产出 /        │    │   逐年独立拟合                    │    │   逐年主导因子判定            │
│   非期望产出 → 效率分数    │    │   OLS → GWR → STWR              │    │   转移矩阵构建                │
│                          │    │   自动带宽搜索                   │    │   桑基图 + 地图可视化         │
└─────────────────────────┘    └──────────────────────────────────┘    └─────────────────────────────┘
```

## 项目结构

| 文件 | 说明 |
|---|---|
| `app.py` | Streamlit 主应用（UI 界面、状态管理、AI 对话、可视化） |
| `core.py` | 核心算法库（SBM 求解器、STWR 建模、路径分析、桑基图数据处理） |
| `All_SBM_Data_Merged.xlsx` | 示例 SBM 输入数据（省份 × 年份 × 投入产出） |
| `STWR_Merged_Input1.csv` | 示例 STWR 驱动因素数据（省份 × 年份 × 经纬度 × 解释变量） |
| `download_geojson.py` | 一次性脚本，下载 `china_provinces.geojson` 用于地图渲染 |

## 环境安装

```bash
pip install streamlit pandas numpy cvxpy scipy matplotlib seaborn plotly statsmodels requests openpyxl
```

`stwr` 库（时空加权回归）需在本地可用，如未安装请从课题组仓库获取。

## 快速启动

```bash
streamlit run app.py
```

### 首次使用

下载中国省份 GeoJSON 文件以启用地图可视化：

```bash
python download_geojson.py
```

## 使用指南

### 1. 配置参数（侧边栏）

| 参数 | 说明 | 默认值 |
|---|---|---|
| 规模报酬 (RTS) | VRS（可变）或 CRS（固定） | VRS |
| P 值阈值 | Step 3 路径分析的显著性水平 | 0.05 |
| 路径频率阈值 (%) | 桑基图显示的最低路径出现频率 | 1% |

### 2. 上传数据

- **SBM 数据**（xlsx/csv）：包含投入、期望产出、非期望产出，以及 ID 列和时间列
- **STWR 驱动因素数据**（xlsx/csv）：包含解释变量、经纬度、ID 列和时间列

### 3. 运行流程

| 步骤 | 按钮 | 功能 |
|---|---|---|
| Step 1 | **Run SBM Calculation** | 通过含非期望产出的 SBM 模型计算效率分数 |
| Step 2 | **Run STWR Regression** | 逐年独立拟合 OLS → GWR → STWR，输出系数、t 值、p 值、local R² |
| Step 3 | **Discover Evolution Paths** | 识别每个省份-年份的主导驱动因子，构建转移矩阵 |

### 4. 可视化

- **地图**：STWR 系数在各省份的空间分布（需要 `china_provinces.geojson`）
- **桑基图**：驱动因子跨年演化流向，采用马卡龙色系

### 5. AI 助手

完成计算后，在侧边栏输入问题。AI 助手可访问全部计算数据（SBM 效率、回归诊断指标、系数矩阵、路径分析结果），回答专业领域问题。

> ⚠️ 请将 `app.py` 中的 `INTERNAL_API_KEY` 和 `ENDPOINT_ID` 替换为你自己的 API 凭证。

## 数据格式

### SBM 输入（Excel 或 CSV）

列名可自定义（在界面中手动选择）：

| 列角色 | 示例 |
|---|---|
| ID | 省份/城市名称（如 `Beijing`、`上海`） |
| 时间 | 年份（如 `2004`、`2005`） |
| 投入 | 劳动力、资本、能源消耗等 |
| 期望产出 | 发电量、GDP 等 |
| 非期望产出 | CO₂ 排放、污染物等 |

### STWR 驱动因素输入（Excel 或 CSV）

| 列角色 | 示例 |
|---|---|
| ID | 省份/城市名称（与 SBM 的 ID 一致） |
| 时间 | 年份（与 SBM 时间一致） |
| 纬度 | 十进制 |
| 经度 | 十进制 |
| 解释变量 | 产业结构、人口、能源强度等 |

## 输出文件

Step 2 完成后可下载 ZIP 包，目录结构：

```
Stage2_STWR/
├── AICc.csv              # 模型对比: OLS vs GWR vs STWR
├── R2.csv                # R² 对比
├── RSS.csv               # 残差平方和对比
├── Sigma.csv             # Sigma 对比
└── {年份}/
    ├── stwr_params.csv      # STWR 系数
    ├── stwr_t-Values.csv    # t 值
    ├── stwr_p-Values.csv    # p 值
    ├── stwr_z-Scores.csv    # z 分数
    ├── stwr_residuals.csv   # 残差 + local R²
    ├── gwr_params.csv       # GWR 系数
    ├── ols_params.csv       # OLS 系数
    └── summary.txt          # 年度模型摘要
```

Step 3 完成后可下载省份演化矩阵和路径统计表。

## 省份名称映射

应用同时支持**中文**和**英文**省份 ID。英文名称会自动映射为 GeoJSON 地图所需的中文全称。支持的英文 ID：

`Beijing`、`Tianjin`、`Hebei`、`Shanxi`、`Inner Mongolia`、`InnerMongolia`、`Liaoning`、`Jilin`、`Heilongjiang`、`Shanghai`、`Jiangsu`、`Zhejiang`、`Anhui`、`Fujian`、`Jiangxi`、`Shandong`、`Henan`、`Hubei`、`Hunan`、`Guangdong`、`Guangxi`、`Hainan`、`Chongqing`、`Sichuan`、`Guizhou`、`Yunnan`、`Tibet`、`Xizang`、`Shaanxi`、`Gansu`、`Qinghai`、`Ningxia`、`Xinjiang`

## 算法说明

### SBM（Slacks-Based Measure）

- 含非期望产出的非径向、非导向 DEA 模型
- 将投入产出矩阵归一化至 [0.1, 1.0] 区间
- 求解器策略：CVXPY 调用 ECOS → 失败则回退至 SCS

### STWR（时空加权回归）

- 逐年独立拟合，自动带宽搜索
- 优先使用 `Sel_Spt_BW`（时空带宽）
- 回退方案：`Sel_BW`（纯空间带宽）+ 默认时间衰减
- 输出：参数、t 值、p 值、z 分数、残差、local R²

### 路径演化分析

- 逐年逐省份主导因子判定（p < 阈值且 |系数| 最大）
- 相邻年份转移矩阵（如 `Industrial+ → Energy-`）
- 频率统计 + 桑基图可视化


---

## v2 更新：RAG 检索 + Agent 调度 + 数值核验

新增模块
- `retrieval.py` — 省/年份精细检索、稳定证据 ID、JSON 证据序列化。
- `agent_layer.py` — LLM 规划（JSON 任务计划）+ 校验后的工具注册表与执行器。
- `verification.py` — 数值声明抽取、确定性核验、答案修正。
- `stwr_multislice.py` — 正确的多切片 STWR（时间核生效）。
- `prompts.py` — 检索/分析/综合/输出契约提示词。

模型与 API Key
问答使用 DeepSeek-Chat（OpenAI 兼容接口）。**密钥不写入仓库**：用环境变量或本地 `local_secrets.py`（已被 .gitignore 忽略）。

```bash
export DEEPSEEK_API_KEY=你的key
```

`local_secrets.py`：
```python
DEEPSEEK_API_KEY = "你的key"
```

运行
```bash
pip install -r requirements.txt
python download_geojson.py   # 下载地图所需 shapefile
streamlit run app.py
```

数据
- `结果文件/` — 预计算的 SBM/STWR/路径快照，检索层启动时读取。
- `data/` — 原始输入表。

Agent 模式：侧边栏勾选 "Agent-driven planning + numerical verification"。
