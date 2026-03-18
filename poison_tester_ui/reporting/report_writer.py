import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

from poison_tester_ui.utils.io import save_json

@dataclass
class RunReport:
    meta: Dict[str, Any]
    environment: Dict[str, Any]
    seed: int
    data: Dict[str, Any]
    preprocess: Dict[str, Any]
    model: Dict[str, Any]
    results: Dict[str, Any]
    plots: Dict[str, str]

def write_run_report(report: RunReport, run_dir: str) -> None:
    save_json(os.path.join(run_dir, "report.json"), asdict(report))

    md = []
    md.append("# Poison Tester Report\n")
    md.append(f"- run_id: `{report.meta.get('run_id')}`")
    md.append(f"- output_dir: `{report.meta.get('output_dir')}`\n")

    md.append("## Baseline\n")
    for k, v in report.results.get("baseline", {}).items():
        md.append(f"- {k}: **{v:.4f}**" if isinstance(v, float) else f"- {k}: `{v}`")

    md.append("\n## Attack\n")
    attack = report.results.get("attack", {})
    for k, v in attack.items():
        if isinstance(v, float):
            md.append(f"- {k}: **{v:.4f}**")
        else:
            md.append(f"- {k}: `{v}`")

    md.append("\n## Defense\n")
    for d in report.results.get("defense", []):
        md.append(f"### {d.get('name')}")
        for k, v in d.items():
            if k in ("name",):
                continue
            if isinstance(v, float):
                md.append(f"- {k}: **{v:.4f}**")
            elif isinstance(v, (int, str)) or v is None:
                md.append(f"- {k}: `{v}`")
        md.append("")

    md.append("\n## Post-defense\n")
    for k, v in report.results.get("post_defense", {}).items():
        md.append(f"- {k}: **{v:.4f}**" if isinstance(v, float) else f"- {k}: `{v}`")

    md.append("\n## Plots\n")
    for k, v in report.plots.items():
        md.append(f"- {k}: `{v}`")

    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))