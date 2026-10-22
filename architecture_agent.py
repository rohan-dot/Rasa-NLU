#!/usr/bin/env python3
"""
architecture_agent.py — Agent 2: read the audit report from Agent 1 (optionally spot-check
the repository) and design an agentic RAG + fine-tuning architecture that improves on it.

Outputs (in --out-dir)
  ARCHITECTURE.md   diagnosis -> targets -> system design (Mermaid) -> components ->
                    data & LoRA plan -> eval/ablations -> orchestration -> roadmap -> risks
  architecture.mmd  the Mermaid diagram alone
  scaffold/         a code skeleton for the proposed system (one LLM call per file)
  arch_notes.md, trace_arch.jsonl

Usage
  python architecture_agent.py --report agent_outputs/report.md [--root /path/to/lora_project] \
      --out-dir agent_outputs --model ... --base-url ...
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from agent_core import Agent, LLMClient, Tool, ToolRegistry, add_agent_args, add_llm_args, llm_from_args
from repo_tools import RepoTools, parse_roots

DESIGN_SYSTEM = """You are a senior applied scientist who designs agentic AI systems for retrieval-augmented generation (RAG) with LoRA-fine-tuned open-weight models. You will receive an audit report of an existing project (and, if available, read-only tools over its code). Your job: design the NEXT system — grounded in the report's evidence — and make it measurable.

Design doctrine
- Diagnose before prescribing. Attribute the current shortfall to one or more of: (a) DATA — quantity, quality, format, leakage, label noise; (b) RETRIEVAL — recall@k, chunking, embedder, hybrid/BM25, reranking, context budget; (c) GENERATION — base model capacity, LoRA config, prompt/template drift between train and inference, decoding; (d) EVALUATION — metric validity, eval-set size/contamination, no retrieval-only metrics; (e) TRAINING RECIPE — lr, steps, overfitting, loss masking, packing. Cite the report's evidence for each.
- Fix first-order problems first: eval validity and data before architecture; retrieval before generator tweaks when retrieval recall is unknown or low. LoRA knobs (r, alpha, target_modules, lr) are second-order — change them only with a stated reason from the evidence (e.g., attention-only targets, underfitting on train loss, dropout too high for tiny data).
- Agentic components earn their place only by fixing a diagnosed failure: query routing when question types differ; decomposition/multi-hop retrieval for compositional questions; HyDE or query rewriting when lexical mismatch hurts recall; CRAG-style evidence grading with re-retrieval when retrieved context is often irrelevant; a verifier/critic when faithfulness or answer-format errors dominate; abstention when evidence is insufficient. Each loop needs a budget and a stopping rule.
- Retrieval and fine-tuning are complementary: RAFT (train on oracle + distractor documents with citation-grounded reasoning), hard-negative mining from the retriever's own failures, synthetic QA generated from the corpus with validation, consistency filtering, and preference pairs (DPO/ORPO) mined from verifier judgments. Train the model on the SAME prompt template and context format it will see at inference.
- Everything must be measurable: retrieval metrics (recall@k, MRR, nDCG) separate from end-to-end metrics (EM/F1/accuracy/ROUGE per the task; faithfulness; citation precision; abstention rate), fixed seeds, confidence intervals on small eval sets, and an ablation matrix that attributes gains to components.
- Respect the existing stack and compute: reuse their data formats, frameworks and serving (open-weight models on vLLM behind an OpenAI-compatible endpoint is the default assumption unless the report says otherwise). Prefer the cheapest intervention that addresses the diagnosed bottleneck; sequence work so each phase has an exit criterion.
- When the report covers several fine-tuning methods (e.g. QuanTA vs LoRA), decide which one the new system carries forward and why (evidence: results, fairness of the comparison, trainable params, serving support — e.g. vLLM serves LoRA adapters natively; a non-LoRA adapter must be merged into the weights to serve). If the comparison was confounded, the plan must include the controlled re-run that settles it.
- The fine-tuned generator must sit INSIDE the retrieval loop: trained on the same chunk format, template and k it will see at inference (RAFT), and evaluated end-to-end with retrieval, not only standalone.
- Be concrete: name files, functions, prompts, config keys and metrics; give numbers (k, chunk sizes, distractor ratio, lr) with the reasoning behind them.
"""

EXPLORE_USER = """Read the audit report (and the extracted slide deck) below. Then use the tools to confirm the three or four specifics your design depends on (e.g. the exact prompt template, dataset schema, how metrics are computed, what retrieval exists). Save notes as you go: at least one note per bottleneck you diagnose (topic='diagnosis') and one per design decision (topic='decision'), each citing the report section, slide number or file. Finish with a plain-text summary: diagnosed bottlenecks ranked by impact, and the headline design.

{report_block}
{slides_block}
{extra}"""

DESIGN_USER = """Write ARCHITECTURE.md for the next system. Base every claim on the audit report, the slide deck and the notes below; do not invent facts about the existing project. If a part of the current system was NOT available to the audit (e.g. the RAG code lives elsewhere), say so explicitly, design that part from the deck's description and the task, and mark every assumption as ASSUMED so it can be checked against the real code later.

Required structure (Markdown, use these exact headings):
# <Project name> — agentic RAG + LoRA improvement architecture
## 1. Diagnosis
Table: symptom | evidence (report section / file) | likely cause (data / retrieval / generation / eval / recipe) | confidence. Then 3–5 sentences on the ranked bottlenecks.
## 2. Targets
Baseline numbers from the report -> target numbers; what counts as success; what would falsify the diagnosis.
## 3. System overview
One paragraph, then ONE Mermaid diagram (```mermaid, flowchart TD) showing the online path (query -> ... -> answer) and the offline loops (eval harness, data flywheel, fine-tuning). Use subgraphs.
## 4. Component specifications
For each component: purpose | inputs/outputs | model or algorithm choice with rationale | key config (concrete values) | failure mode it addresses | how to evaluate it in isolation. Include at minimum: query analysis/routing, retrieval (index, chunking, hybrid, k), reranking, evidence grading + re-retrieval loop, generator (the LoRA model and its prompt), verifier/critic, abstention/fallback, orchestrator state and stopping rules.
## 5. Data and fine-tuning plan
Dataset construction (RAFT mix: oracle + distractors, distractor ratio, citation-grounded CoT format, train/inference template parity), synthetic data and filtering, hard-negative mining, preference data; LoRA/QLoRA config as a diff from the current one with reasons; training schedule; loss masking; checkpoints to keep.
## 6. Evaluation and ablation plan
Harness design; retrieval metrics vs end-to-end metrics vs faithfulness; baselines; ablation matrix (rows = components toggled, cols = metrics); seeds and CIs; LLM-judge prompt sketch if used; contamination checks.
## 7. Orchestration
Node/edge list of the agent graph (name, input state, output state, tool calls, budget, exit condition), a worked trace for one question, latency/cost budget.
## 8. Roadmap
Phases with scope, effort, exit criterion and the metric each phase must move.
## 9. Risks and mitigations
## 10. Scaffold index
The files the code scaffold will contain and one line on each.

=== AUDIT REPORT ===
{report}

=== SLIDE DECK (extracted) ===
{slides}

=== NOTES FROM SPOT CHECKS ===
{notes}

=== EXPLORATION SUMMARY ===
{summary}
"""

MANIFEST_USER = """Based on ARCHITECTURE.md below, list the files for a minimal, runnable Python skeleton of the proposed system. Respond with ONLY a JSON array of objects {{"path": "...", "purpose": "..."}} (6–{max_files} entries; paths relative to the scaffold root). Cover: config, orchestrator/agent graph, retrieval, rerank, evidence grading, generation client (OpenAI-compatible / vLLM), verifier, RAFT dataset builder, LoRA training script, evaluation harness, README.

=== ARCHITECTURE.md ===
{architecture}
"""

FILE_USER = """Write the complete contents of `{path}` — {purpose}.

Requirements
- Consistent with ARCHITECTURE.md (below) and with the other scaffold files: {others}.
- Python 3.10+; standard library plus the libraries the architecture names. Use the raw `openai.OpenAI` client for model calls (no LangChain). Load settings from the scaffold's config file.
- Complete interfaces, dataclasses, CLI (argparse) and docstrings. Heavy internals may be marked `# TODO` with a one-line note, but every function must have a correct signature and a sensible default behaviour (e.g. returns an empty list) so the modules import and the CLI runs.
- For README.md: how to run each stage end to end.
Output exactly ONE fenced code block containing the file, nothing else.

=== ARCHITECTURE.md ===
{architecture}
"""

DEFAULT_MANIFEST = [
    {"path": "README.md", "purpose": "how to run each stage: index, retrieve, generate, verify, build RAFT data, train LoRA, evaluate"},
    {"path": "config.yaml", "purpose": "all settings: endpoints, model names, LoRA config, retrieval k, chunking, budgets"},
    {"path": "pipeline/__init__.py", "purpose": "package marker"},
    {"path": "pipeline/config.py", "purpose": "typed settings loader for config.yaml"},
    {"path": "pipeline/retrieval.py", "purpose": "chunking, hybrid BM25 + dense index build and search"},
    {"path": "pipeline/rerank.py", "purpose": "cross-encoder reranking and evidence grading"},
    {"path": "pipeline/generate.py", "purpose": "OpenAI-compatible client wrapper for the LoRA model with the training prompt template"},
    {"path": "pipeline/verify.py", "purpose": "faithfulness / format verifier and abstention logic"},
    {"path": "pipeline/orchestrator.py", "purpose": "agent graph: route -> retrieve -> grade -> (re-retrieve) -> generate -> verify, with budgets"},
    {"path": "data/build_raft_dataset.py", "purpose": "RAFT dataset builder: oracle + distractor docs, citation-grounded targets"},
    {"path": "train/train_lora.py", "purpose": "LoRA/QLoRA fine-tuning script with the recommended config"},
    {"path": "eval/run_eval.py", "purpose": "evaluation harness: retrieval metrics, end-to-end metrics, ablation switches"},
]

_FENCE_BLOCK = re.compile(r"```[a-zA-Z0-9_+.-]*\n(.*?)```", re.DOTALL)


def extract_code_block(text: str) -> str:
    blocks = _FENCE_BLOCK.findall(text)
    if blocks:
        return max(blocks, key=len).rstrip() + "\n"
    return text.strip() + "\n"


def extract_mermaid(text: str) -> str | None:
    m = re.search(r"```mermaid\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() + "\n" if m else None


def parse_manifest(text: str, max_files: int) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            items = [
                {"path": str(x["path"]).strip().lstrip("/"), "purpose": str(x.get("purpose", "")).strip()}
                for x in arr
                if isinstance(x, dict) and x.get("path")
            ]
            if items:
                return items[:max_files]
        except (json.JSONDecodeError, TypeError):
            pass
    return DEFAULT_MANIFEST[:max_files]


def _safe_target(base: Path, rel: str) -> Path | None:
    cand = (base / rel).resolve()
    try:
        cand.relative_to(base.resolve())
    except ValueError:
        return None
    return cand


def run_architecture_agent(
    report_path: str | Path | list[str | Path],
    out_dir: str | Path,
    llm: LLMClient,
    *,
    root: str | Path | dict[str, Path] | list[str] | None = None,
    slides: str | Path | None = None,
    hint: str = "",
    max_turns: int = 15,
    context_chars: int = 240_000,
    synthesis_max_tokens: int = 8192,
    scaffold: bool = True,
    max_scaffold_files: int = 12,
    verbose: bool = True,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_paths = [Path(r) for r in (report_path if isinstance(report_path, (list, tuple)) else [report_path])]
    if len(report_paths) == 1:
        report = report_paths[0].read_text(encoding="utf-8", errors="replace")
    else:  # several audits (e.g. one per cluster) -> one document with a header per report
        report = "\n\n".join(f"=== REPORT {i + 1}: {rp} ===\n" + rp.read_text(encoding="utf-8", errors="replace") for i, rp in enumerate(report_paths))
    report_path = report_paths[0]
    t0 = time.time()
    if slides is None:  # Agent 1 writes slides_outline.md next to report.md
        cand = report_path.parent / "slides_outline.md"
        slides = cand if cand.exists() else None
    slides_text = Path(slides).read_text(encoding="utf-8", errors="replace").strip() if slides else ""
    if slides_text.startswith("(no .pptx"):
        slides_text = ""

    # tools: repo spot-checks (if roots are given) + the report itself
    registry = ToolRegistry()
    roots: dict[str, Path] = {}
    if root:
        roots = root if isinstance(root, dict) else parse_roots([str(r) for r in (root if isinstance(root, (list, tuple)) else [root])])
        repo = RepoTools(roots=roots, extra_files=report_paths)
        registry.register_specs(repo.tool_specs())
    registry.register(
        Tool(
            name="read_report",
            description="Re-read a line range of the audit report.",
            parameters={"type": "object", "properties": {"start_line": {"type": "integer"}, "max_lines": {"type": "integer"}}},
            fn=lambda start_line=1, max_lines=200: _read_range(report, int(start_line or 1), int(max_lines or 200)),
        )
    )
    agent = Agent(
        llm,
        registry,
        DESIGN_SYSTEM,
        name="architecture-agent",
        max_turns=max_turns,
        context_char_budget=context_chars,
        trace_path=out_dir / "trace_arch.jsonl",
        verbose=verbose,
    )
    if len(report) <= 80_000:
        report_block = "=== AUDIT REPORT ===\n" + report
    else:
        report_block = "=== AUDIT REPORT (first part; use read_report for the rest) ===\n" + report[:80_000]
    extra = ""
    if roots:
        if len(roots) == 1 and "" in roots:
            extra += f"\nThe project directory is available read-only (root: {roots['']}).\n"
        else:
            extra += "\nProject directories are available read-only under labels; tool paths start with the label: " + "; ".join(f"`{k}/` = {v}" for k, v in roots.items()) + "\n"
    else:
        extra += "\nNo project directory was provided; rely on the report.\n"
    if hint:
        extra += f"Additional context from the user: {hint}\n"
    slides_block = ("=== SLIDE DECK (extracted) ===\n" + slides_text[:40_000]) if slides_text else "(no slide deck provided)"
    result = agent.run(EXPLORE_USER.format(report_block=report_block, slides_block=slides_block, extra=extra))
    (out_dir / "arch_notes.md").write_text("# Design notes\n" + agent.notes_markdown() + "\n\n# Summary\n" + result.final_text, encoding="utf-8")

    # design document
    architecture = agent.generate_long(
        DESIGN_SYSTEM,
        DESIGN_USER.format(report=report, slides=slides_text[:40_000] or "(none)", notes=agent.notes_markdown(), summary=result.final_text or "(none)"),
        max_tokens=synthesis_max_tokens,
    )
    arch_path = out_dir / "ARCHITECTURE.md"
    arch_path.write_text(architecture + "\n", encoding="utf-8")
    mermaid = extract_mermaid(architecture)
    if mermaid:
        (out_dir / "architecture.mmd").write_text(mermaid, encoding="utf-8")
    if verbose:
        print(f"[arch] ARCHITECTURE.md written ({len(architecture)} chars)", file=sys.stderr)

    # code scaffold: manifest, then one call per file
    written: list[str] = []
    if scaffold:
        scaffold_dir = out_dir / "scaffold"
        scaffold_dir.mkdir(exist_ok=True)
        manifest_text = agent.generate_long(DESIGN_SYSTEM, MANIFEST_USER.format(architecture=architecture[:40_000], max_files=max_scaffold_files), max_tokens=2048, max_continues=0)
        manifest = parse_manifest(manifest_text, max_scaffold_files)
        (scaffold_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        others = ", ".join(m["path"] for m in manifest)
        for item in manifest:
            target = _safe_target(scaffold_dir, item["path"])
            if target is None:
                if verbose:
                    print(f"[arch] skipping unsafe path {item['path']!r}", file=sys.stderr)
                continue
            text = agent.generate_long(
                DESIGN_SYSTEM,
                FILE_USER.format(path=item["path"], purpose=item["purpose"] or "part of the scaffold", others=others, architecture=architecture[:40_000]),
                max_tokens=synthesis_max_tokens,
                max_continues=2,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(extract_code_block(text), encoding="utf-8")
            written.append(item["path"])
            if verbose:
                print(f"[arch] scaffold: {item['path']}", file=sys.stderr)

    meta = {
        "reports": [str(rp.resolve()) for rp in report_paths],
        "slides": str(Path(slides).resolve()) if slides else None,
        "roots": {k or "(root)": str(v) for k, v in roots.items()} or None,
        "turns": result.turns,
        "tool_calls": result.tool_calls,
        "notes": len(result.notes),
        "scaffold_files": written,
        "usage": agent.llm.usage,
        "seconds": round(time.time() - t0, 1),
        "model": llm.cfg.model,
        "tool_transport": llm.transport_used,
    }
    (out_dir / "arch_run.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if verbose:
        print(f"[arch] done: {arch_path} (+{len(written)} scaffold files, {meta['seconds']}s)", file=sys.stderr)
    return arch_path


def _read_range(text: str, start: int, n: int) -> str:
    lines = text.splitlines()
    start = max(1, start)
    chunk = lines[start - 1 : start - 1 + max(1, n)]
    head = f"# report lines {start}-{start + len(chunk) - 1} of {len(lines)}"
    return head + "\n" + "\n".join(f"{start + i:5d}| {ln}" for i, ln in enumerate(chunk))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Agent 2: design an agentic RAG + LoRA architecture from the audit report")
    p.add_argument("--report", action="append", required=True, help="report.md from report_agent.py (repeatable: one audit per cluster)")
    p.add_argument("--root", action="append", default=None, help="project directory for read-only spot checks; repeatable, optionally labelled (quanta=/path)")
    p.add_argument("--out-dir", default="agent_outputs")
    p.add_argument("--slides", default=None, help="slides_outline.md from Agent 1 (default: the one next to --report, if any)")
    p.add_argument("--hint", default="", help="constraints for the design, e.g. 'must run on 2x A100, no external APIs'")
    p.add_argument("--no-scaffold", action="store_true", help="write ARCHITECTURE.md only")
    p.add_argument("--max-scaffold-files", type=int, default=12)
    add_llm_args(p)
    add_agent_args(p, default_max_turns=15)
    args = p.parse_args(argv)
    llm = llm_from_args(args)
    path = run_architecture_agent(
        args.report,
        args.out_dir,
        llm,
        root=args.root,
        slides=args.slides,
        hint=args.hint,
        max_turns=args.max_turns,
        context_chars=args.context_chars,
        synthesis_max_tokens=args.synthesis_max_tokens,
        scaffold=not args.no_scaffold,
        max_scaffold_files=args.max_scaffold_files,
        verbose=not args.quiet,
    )
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
