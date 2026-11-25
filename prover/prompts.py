"""Prompts for a first attempt and for repairing a failed one.

The first-attempt wording follows what Lean prover models (DeepSeek-Prover-V2, Goedel-Prover-V2)
are trained on: "Complete the following Lean 4 code" plus a request for a proof plan.
"""
from __future__ import annotations

from .engine import Message
from .jobs import Job
from .lean import skeleton

SYSTEM = ("You are an expert in Lean 4 and Mathlib. You write complete proofs that compile. "
          "You never change the statement you are asked to prove.")

PLAN = ("Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan "
        "outlining the main proof steps and strategies.\n"
        "The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the "
        "construction of the final formal proof.")

RULES = ("Rules: Lean 4 with Mathlib. Keep the theorem statement exactly as given. No sorry, no admit, no axioms, "
         "no native_decide, no # commands, no double quotes, no new notation or attributes (only @[simp]). "
         "Put helper lemmas before the main theorem. End your answer with one ```lean4 code block that contains "
         "the complete file.")


def clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 20)].rstrip() + "\n[... cut for length]"


def clip_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(0, limit // 2 - 20)
    return text[:half] + "\n-- [... middle of the file cut for length ...]\n" + text[-half:]


def _background(job: Job, defs: list[tuple[str, str]], budget: int) -> list[str]:
    parts: list[str] = []
    if job.title:
        parts.append(f"Goal: {job.title}")
    if defs:
        room = int(budget * 0.5)
        chosen, used = [], 0
        for _, snippet in defs:
            if used + len(snippet) > room:
                continue
            chosen.append(snippet)
            used += len(snippet)
        if chosen:
            parts.append("Definitions from the project that the statement uses:\n```lean4\n"
                         + "\n\n".join(chosen) + "\n```")
    if job.context:
        parts.append("What is known about this step:\n" + clip(job.context, int(budget * 0.2)))
    if job.sketch:
        parts.append("Proof idea:\n" + clip(job.sketch, int(budget * 0.15)))
    if job.dead:
        parts.append("Approaches that already failed:\n- " + clip("\n- ".join(job.dead), int(budget * 0.08)))
    return parts


def first_prompt(job: Job, defs: list[tuple[str, str]], budget: int, note: str | None = None) -> list[Message]:
    parts = _background(job, defs, budget)
    if note:
        parts.append("Your previous answer could not be used: " + note)
    parts.append("Complete the following Lean 4 code:\n\n```lean4\n" + skeleton(job) + "```")
    parts.append(PLAN)
    parts.append(RULES)
    return [("system", SYSTEM), ("user", "\n\n".join(parts))]


def repair_prompt(job: Job, defs: list[tuple[str, str]], file_text: str, feedback: list[str],
                  budget: int) -> list[Message]:
    parts = _background(job, defs, int(budget * 0.5))
    parts.append("This Lean 4 file does not compile yet:\n\n```lean4\n" + clip_middle(file_text, int(budget * 0.35))
                 + "```")
    parts.append("Lean reported:\n" + clip("\n".join(f"- {f}" for f in feedback), int(budget * 0.15)))
    parts.append("Explain briefly what is wrong, then give the corrected proof. " + RULES)
    return [("system", SYSTEM), ("user", "\n\n".join(parts))]
