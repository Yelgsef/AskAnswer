"""Per-business registry for verified dynamic DuckDB macros."""
import hashlib
import re

from .ingest import quote
from .prompts import MACRO_GUIDANCE


def safe_name(proposed):
    """Map a model name to a deterministic, non-shadowing SQL identifier."""
    raw = str(proposed).strip().lower()
    slug = re.sub(r"[^a-z0-9_]+", "_", raw).strip("_") or "macro"
    digest = hashlib.sha256(str(proposed).encode("utf-8")).hexdigest()[:10]
    return f"dm_{slug[:40]}_{digest}"


def create_sql(name, parameters, body):
    params = ", ".join(p["name"] for p in parameters)
    return f"CREATE MACRO {quote(name)}({params}) AS TABLE {body}"


class MacroRegistry:
    """A business-local registry; entries are only added after semantic validation."""

    def __init__(self):
        self.entries = []
        self.rejections = []

    def register(self, db, candidate, plan, validation):
        if validation.get("status") != "SEMANTICALLY_VERIFIED":
            return None
        proposed = candidate["macro_name"]
        if any(e["name"] == proposed for e in self.entries):
            self.rejections.append({"name": proposed, "reason": "duplicate macro name"})
            return None
        sql_hash = hashlib.sha256(candidate["sql_body"].encode("utf-8")).hexdigest()
        if any(e["sql_hash"] == sql_hash for e in self.entries):
            self.rejections.append({"name": proposed, "reason": "duplicate macro SQL"})
            return None
        name = safe_name(proposed)
        existing = {str(r[0]).lower() for r in db.execute("SELECT function_name FROM duckdb_functions()").fetchall()}
        if name.lower() in existing or any(e["registered_name"] == name for e in self.entries):
            self.rejections.append({"name": proposed, "reason": f"registered identifier collision: {name}"})
            return None
        parameters = candidate["parameters"]
        db.execute(create_sql(name, parameters, candidate["sql_body"]))
        entry = {
            "name": proposed,
            "registered_name": name,
            "description": str(candidate.get("description", plan.get("description", "")))[:1000],
            "parameters": parameters,
            "output_columns": candidate["output_columns"],
            "sql_body": candidate["sql_body"],
            "source_tables": sorted(candidate.get("source_tables", plan.get("required_tables", []))[:32]),
            "rulebook_evidence": plan.get("rulebook_evidence", [])[:8],
            "validation_status": validation["status"],
            "validation_evidence": validation.get("evidence", []),
            "sql_hash": sql_hash,
            "relevant_question_indices": sorted(plan.get("question_indices", [])),
        }
        self.entries.append(entry)
        self.entries.sort(key=lambda e: (e["name"], e["sql_hash"]))
        return entry

    def context(self, question_index=None):
        entries = [e for e in self.entries if question_index is None or not e["relevant_question_indices"] or question_index in e["relevant_question_indices"]]
        if not entries:
            return ""
        lines = [MACRO_GUIDANCE, "AVAILABLE VERIFIED BUSINESS MACROS:"]
        for e in entries:
            params = ", ".join(f"{p['name']} {p['type']}" for p in e["parameters"])
            returns = ", ".join(e["output_columns"])
            lines.append(f"{e['registered_name']}({params})")
            lines.append(f"Purpose: {e['description']}")
            lines.append(f"Returns: {returns}")
            if e.get("rulebook_evidence"):
                lines.append("Rulebook evidence: " + " | ".join(e["rulebook_evidence"]))
        return "\n".join(lines)

    def as_dict(self):
        return {"entries": self.entries, "rejections": self.rejections}
