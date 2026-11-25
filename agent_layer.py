"""Agent-driven analytical task orchestration for the SBM-STWR workflow.

Principle:
  Agent decides WHAT to call; core.py performs the statistical computation;
  retrieval.py retrieves evidence; the LLM only plans / extracts / writes.

The LLM returns a fixed JSON task plan (never Python code). A deterministic
validator checks the plan (whitelist, unique ids, dependency existence, no cycles,
data bounds) before execution.
"""
import json

try:
    import streamlit as st
except Exception:  # allows offline unit tests
    st = None

import retrieval
import prompts

MAX_TASKS = 6

TOOL_WHITELIST = {
    "inspect_data", "get_sbm", "get_stwr", "get_paths",
    "run_sbm", "run_stwr", "run_paths", "retrieve_evidence", "synthesize",
}


def _ss():
    return st.session_state if st is not None else {}


def _ok(**kw):
    d = {"status": "success"}
    d.update(kw)
    return d


def _fail(status, message, **kw):
    d = {"status": status, "message": message}
    d.update(kw)
    return d


# ------------------------------------------------------------------ tools (read existing results)
def tool_inspect_data(args):
    ss = _ss()
    out = {}
    for key, model in (("sbm_res_df", "SBM"), ("path_raw_df", "STWR"), ("path_matrix_df", "Path")):
        df = ss.get(key)
        if df is None or getattr(df, "empty", True):
            out[model] = {"status": "missing"}
        else:
            out[model] = {"status": "success", "rows": int(len(df)), "columns": list(map(str, df.columns))[:12]}
    return _ok(available=out)


def _df_records(df, limit=None):
    d = df if limit is None else df.head(limit)
    return d.to_dict("records")


def tool_get_sbm(args):
    df = _ss().get("sbm_res_df")
    if df is None:
        return _fail("missing", "SBM has not been calculated.")
    return _ok(rows=int(len(df)), data=_df_records(df, args.get("limit")))


def tool_get_stwr(args):
    df = _ss().get("path_raw_df")
    if df is None:
        return _fail("missing", "STWR has not been calculated.")
    return _ok(rows=int(len(df)), data=_df_records(df, args.get("limit")))


def tool_get_paths(args):
    df = _ss().get("path_matrix_df")
    if df is None:
        return _fail("missing", "Path analysis has not been completed.")
    return _ok(rows=int(len(df)), data=_df_records(df, args.get("limit")))


# ------------------------------------------------------------------ tools (compute; reuse saved configs)
def tool_run_sbm(args):
    ss = _ss()
    cfg = ss.get("sbm_config")
    raw = ss.get("raw_sbm_df")
    if cfg is None or raw is None:
        return _fail("missing_config", "SBM configuration is not available; run Step 1 manually once.")
    import core
    d_c = raw[(raw[cfg["time"]] >= cfg["start"]) & (raw[cfg["time"]] <= cfg["end"])].copy().sort_values([cfg["time"], cfg["id"]])
    d_c["Efficiency"] = core.solve_sbm_undesirable(
        d_c[cfg["inputs"]].values.T, d_c[cfg["good"]].values.T, d_c[cfg["bad"]].values.T, cfg["rts"])
    ss["sbm_res_df"] = d_c
    ss["sbm_meta"] = core.build_result_metadata("SBM", cfg["id"], cfg["time"], d_c, cfg)
    return _ok(rows=int(len(d_c)))


def tool_run_stwr(args):
    ss = _ss()
    cfg = ss.get("stwr_config")
    sbm = ss.get("sbm_res_df")
    stwr_data = ss.get("stwr_data")
    if cfg is None or sbm is None or stwr_data is None:
        return _fail("missing_config", "STWR configuration/data is not ready; run Steps 1-2 manually once.")
    import core
    import pandas as pd
    sbm_sub = sbm[[cfg["sbm_id"], cfg["sbm_time"], "Efficiency"]].copy()
    sbm_sub.columns = ["sid", "stm", "Efficiency"]
    sc = stwr_data.copy()
    sc["sid"] = sc[cfg["id"]].astype(str).str.strip()
    sc["stm"] = sc[cfg["time"]].astype(int)
    dm = pd.merge(sc, sbm_sub, on=["sid", "stm"], how="inner")
    z, _, path_df = core.run_stwr_full_process(dm, cfg["id"], "stm", cfg["lat"], cfg["lon"], "Efficiency", cfg["x"])
    ss["path_raw_df"] = path_df
    ss["stwr_meta"] = core.build_result_metadata("STWR", cfg["id"], "stm", path_df, cfg)
    return _ok(rows=int(len(path_df)))


def tool_run_paths(args):
    ss = _ss()
    cfg = ss.get("path_config")
    path_raw = ss.get("path_raw_df")
    if cfg is None or path_raw is None:
        return _fail("missing_config", "Path configuration/STWR result is not ready.")
    import core
    detail, stats = core.analyze_paths(path_raw, cfg["id"], cfg["time"], cfg["y"], cfg["x"], cfg["p"], cfg["freq"])
    ss["path_matrix_df"] = detail
    ss["path_res_df"] = stats
    ss["path_meta"] = core.build_result_metadata("Path", "ID", cfg["time"], detail, cfg)
    return _ok(rows=int(len(detail)))


# ------------------------------------------------------------------ tools (retrieval / synthesis)
def tool_retrieve_evidence(args):
    region = args.get("region") or args.get("province")
    years = args.get("years") or []
    intent = args.get("intent") or "comprehensive"
    question = args.get("question") or args.get("goal") or ""
    records = retrieval.retrieve_task_records(
        {"region": region, "years": years, "intent": intent, "goal": question,
         "factors": args.get("factors") or []})
    if not records:
        return _fail("insufficient_evidence", "No evidence found for the requested scope.",
                     records=[], evidence="")
    evidence = retrieval.serialize(records)
    return _ok(records=records, evidence=evidence, n=len(records))


def tool_synthesize(args):
    return _ok(note="synthesis is performed by the answer-generation step")


TOOLS = {
    "inspect_data": tool_inspect_data,
    "get_sbm": tool_get_sbm,
    "get_stwr": tool_get_stwr,
    "get_paths": tool_get_paths,
    "run_sbm": tool_run_sbm,
    "run_stwr": tool_run_stwr,
    "run_paths": tool_run_paths,
    "retrieve_evidence": tool_retrieve_evidence,
    "synthesize": tool_synthesize,
}


def build_tool_registry():
    return dict(TOOLS)


# ------------------------------------------------------------------ plan validation
def _has_cycle(tasks):
    graph = {t["id"]: list(t.get("depends_on") or []) for t in tasks}
    state = {}

    def dfs(n, stack):
        if state.get(n) == 1:
            return True
        if state.get(n) == 2:
            return False
        state[n] = 1
        for m in graph.get(n, []):
            if dfs(m, stack + [m]):
                return True
        state[n] = 2
        return False

    return any(dfs(n, [n]) for n in graph)


def validate_agent_plan(plan, allowed_regions=None, allowed_years=None):
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except Exception as e:
            return False, f"Plan is not valid JSON: {e}"
    if not isinstance(plan, dict):
        return False, "Plan must be a JSON object."
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return False, "Plan must contain a non-empty 'tasks' list."
    if len(tasks) > MAX_TASKS:
        return False, f"Too many tasks ({len(tasks)} > {MAX_TASKS})."
    ids = []
    for t in tasks:
        if not isinstance(t, dict) or "id" not in t or "tool" not in t:
            return False, "Each task needs 'id' and 'tool'."
        if t["tool"] not in TOOL_WHITELIST:
            return False, f"Tool '{t['tool']}' is not allowed."
        ids.append(t["id"])
    if len(ids) != len(set(ids)):
        return False, "Task ids must be unique."
    idset = set(ids)
    for t in tasks:
        for d in (t.get("depends_on") or []):
            if d not in idset:
                return False, f"Task '{t['id']}' depends on unknown task '{d}'."
        if t["id"] in (t.get("depends_on") or []):
            return False, f"Task '{t['id']}' depends on itself."
    if _has_cycle(tasks):
        return False, "Dependency cycle detected."
    region = plan.get("region")
    if allowed_regions and region:
        if not any(str(region).lower() in str(r).lower() or str(r).lower() in str(region).lower()
                   for r in allowed_regions):
            return False, f"Region '{region}' is outside the available data."
    period = plan.get("period") or []
    if allowed_years and period:
        bad = [int(y) for y in period if int(y) not in allowed_years]
        if bad:
            return False, f"Years {bad} are outside the available data."
    return True, plan


def agent_plan(question, call_llm, allowed_regions=None, allowed_years=None):
    try:
        raw = call_llm(prompts.AGENT_PLANNER_PROMPT, f"User question:\n{question}")
    except Exception as e:
        return False, f"Planner call failed: {e}"
    return validate_agent_plan(raw, allowed_regions, allowed_years)


# ------------------------------------------------------------------ execution
def execute_agent_plan(plan, registry=None, max_tasks=MAX_TASKS, retries=0):
    registry = registry or build_tool_registry()
    results = {}
    tasks = plan.get("tasks", [])[:max_tasks]
    for task in tasks:
        tid = task.get("id")
        tool = task.get("tool")
        deps = task.get("depends_on") or []
        if not all(results.get(d, {}).get("status") == "success" for d in deps):
            results[tid] = _fail("blocked", f"Dependencies not satisfied: {deps}")
            continue
        if tool not in registry:
            results[tid] = _fail("invalid_tool", f"Unknown tool: {tool}")
            continue
        args = dict(task.get("args") or {})
        # propagate plan-level context if the task did not specify it
        args.setdefault("region", plan.get("region"))
        args.setdefault("years", plan.get("period") or [])
        args.setdefault("intent", plan.get("intent"))
        args.setdefault("question", plan.get("question") or "")
        last = None
        for attempt in range(retries + 1):
            try:
                results[tid] = registry[tool](args)
                last = None
                break
            except Exception as e:
                last = e
        if last is not None:
            results[tid] = _fail("failed", str(last))
    return results


def collect_records(exec_results):
    recs = []
    for r in exec_results.values():
        if isinstance(r, dict) and isinstance(r.get("records"), list):
            recs.extend(r["records"])
    # de-duplicate by stable evidence id
    seen, out = set(), []
    for r in recs:
        eid = retrieval.generate_evidence_id(r)
        if eid not in seen:
            seen.add(eid)
            out.append(r)
    return out


def synthesize_answer(question, evidence, call_llm):
    system = prompts.ANALYSIS_SYSTEM_PROMPT + "\n\n" + prompts.OUTPUT_CONTRACT
    user = f"[User Question]\n{question}\n\n[Retrieved Evidence]\n{evidence}"
    return call_llm(system, user)
