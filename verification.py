"""Evidence-grounded numerical verification.

Pipeline: answer -> claim extraction (LLM) -> deterministic verification (code) -> repair.

The LLM is used ONLY to extract claims; the supported/unsupported decision is made
here by deterministic comparison against the retrieved evidence records.
"""
import json
import re

import retrieval
import prompts


# ------------------------------------------------------------------ parsing
def _parse_json_array(text):
    if not text:
        return []
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    m = re.search(r"\[.*\]", t, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return arr
            if isinstance(arr, dict) and isinstance(arr.get("claims"), list):
                return arr["claims"]
        except Exception:
            pass
    # salvage individually-valid objects (handles truncated arrays)
    objs = []
    for mm in re.finditer(r"\{[^{}]*\}", t, re.DOTALL):
        try:
            o = json.loads(mm.group(0))
            if isinstance(o, dict):
                objs.append(o)
        except Exception:
            pass
    if objs:
        return objs
    # truncated array: keep from '[' to the last '}' and close it
    if "[" in t and "{" in t:
        frag = t[t.find("["):]
        frag = frag[: frag.rfind("}") + 1] + "]"
        try:
            arr = json.loads(frag)
            if isinstance(arr, list):
                return arr
        except Exception:
            pass
    return []


def _as_float(v):
    try:
        return float(v)
    except Exception:
        return None


# ------------------------------------------------------------------ claim extraction
def extract_claims(answer, evidence, call_llm):
    user = (f"ANSWER:\n{answer}\n\nEVIDENCE:\n{evidence}\n\n"
            f"Return the JSON array of numerical claims only.")
    try:
        raw = call_llm(prompts.NUMERICAL_VERIFICATION_PROMPT, user)
    except Exception:
        return []
    claims = _parse_json_array(raw)
    cleaned = []
    for c in claims:
        if not isinstance(c, dict):
            continue
        val = _as_float(c.get("value", c.get("numerical_value", c.get("number"))))
        if val is None:
            continue
        cleaned.append({
            "region": str(c.get("region", "") or ""),
            "year": str(c.get("year", "") or ""),
            "factor": str(c.get("factor", "") or ""),
            "value": val,
            "unit": str(c.get("unit", "") or ""),
            "evidence_ids": [str(x) for x in (c.get("evidence_ids") or [])],
            "claim_type": str(c.get("claim_type", "other") or "other"),
        })
    return cleaned


# ------------------------------------------------------------------ deterministic verification
def _close(a, b, abs_tol=0.01, rel_tol=0.03):
    if a is None or b is None:
        return False
    if abs(a - b) <= abs_tol:
        return True
    if abs(b) > 1e-9 and abs(a - b) / abs(b) <= rel_tol:
        return True
    return False


def _year_match(claim_year, rec_year):
    cy = re.findall(r"\d{4}", str(claim_year))
    ry = str(rec_year)
    if not cy:
        return True
    if any(y in ry for y in cy):      # claim year inside a period string
        return True
    return ry in cy


def _is_p_claim(factor):
    f = str(factor).lower()
    return ("p-value" in f) or ("p value" in f) or ("pvalue" in f) or ("p值" in f) or ("p 值" in f)


def _rec_value(record, is_p):
    try:
        return float(record.get("p") if is_p else record.get("value"))
    except Exception:
        return None


def _region_match(claim_region, rec_region):
    if not claim_region:
        return True
    cr = claim_region.lower()
    rr = str(rec_region).lower()
    return rr in cr or cr in rr


_ALIAS = {
    "carbon emission efficiency": "efficiency",
    "cee": "efficiency",
    "delta cee": "deltaeff",
    "\u03b4cee": "deltaeff",
    "deltaeff": "deltaeff",
}


def _canon_factor(s):
    x = str(s or "").lower().strip()
    for k, v in _ALIAS.items():
        x = x.replace(k, v)
    return x


def _factor_match(claim_factor, rec_factor):
    if not claim_factor:
        return True
    cf = _canon_factor(claim_factor)
    rf = _canon_factor(rec_factor)
    return cf in rf or rf in cf


def verify_claims(claims, records, abs_tol=0.01, rel_tol=0.03):
    ev_map = {retrieval.generate_evidence_id(r): r for r in records}
    report = []
    for c in claims:
        ids = c.get("evidence_ids") or []
        matched = [ev_map[i] for i in ids if i in ev_map]
        status = "unsupported"
        reason = ""
        if not matched:
            reason = "no matching evidence id"
        elif c["claim_type"] == "direct":
            ok = False
            for r in matched:
                if not (_region_match(c["region"], r.get("region")) and
                        _year_match(c["year"], r.get("year")) and
                        _factor_match(c["factor"], r.get("factor"))):
                    continue
                # accept if the value matches the coefficient OR the p-value/count
                v_val = _rec_value(r, False)
                v_alt = _rec_value(r, True)
                if _close(c["value"], v_val, abs_tol, rel_tol) or _close(c["value"], v_alt, abs_tol, rel_tol):
                    ok = True
                    break
            status = "supported" if ok else "unsupported"
            if not ok:
                reason = "value/province/year/factor mismatch"
        elif c["claim_type"] == "derived_difference":
            vals = [_as_float(r.get("value")) for r in matched]
            vals = [v for v in vals if v is not None]
            ok = False
            for i in range(len(vals)):
                for j in range(len(vals)):
                    if i == j:
                        continue
                    if _close(c["value"], vals[j] - vals[i], abs_tol, rel_tol):
                        ok = True
                        break
                if ok:
                    break
            status = "supported" if ok else "unsupported"
            if not ok:
                reason = "difference could not be reproduced from evidence"
        else:
            status = "requires_review"
            reason = "non-direct claim type"
        report.append({"claim": c, "status": status, "reason": reason})
    return report


def summarize(report):
    out = {"total": len(report), "supported": 0, "unsupported": 0, "requires_review": 0}
    for x in report:
        out[x["status"]] = out.get(x["status"], 0) + 1
    return out


def verify_answer(question, answer, evidence, records, call_llm):
    claims = extract_claims(answer, evidence, call_llm)
    report = verify_claims(claims, records)
    return report


# ------------------------------------------------------------------ repair
def repair_answer(question, answer, report, evidence, call_llm):
    bad = [x for x in report if x["status"] != "supported"]
    if not bad:
        return answer
    user = (f"QUESTION:\n{question}\n\nORIGINAL ANSWER:\n{answer}\n\n"
            f"VERIFICATION REPORT (unsupported/review):\n{json.dumps(bad, ensure_ascii=False)}\n\n"
            f"EVIDENCE:\n{evidence}")
    try:
        return call_llm(prompts.NUMERICAL_REPAIR_PROMPT, user)
    except Exception:
        return answer


def verify_and_repair(question, answer, evidence, records, call_llm, max_rounds=1):
    """Extract -> verify -> (repair -> re-verify) up to max_rounds."""
    report = verify_answer(question, answer, evidence, records, call_llm)
    rounds = 0
    while rounds < max_rounds and summarize(report)["unsupported"] > 0:
        answer = repair_answer(question, answer, report, evidence, call_llm)
        report = verify_answer(question, answer, evidence, records, call_llm)
        rounds += 1
    return answer, report
