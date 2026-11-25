"""Finds the project definitions a statement uses, so the model sees what the names mean.

A local model has never seen `PEqualNP.MetaComplexity.HasCircuitSize` and friends; without the
definitions it can only guess. This index scans the Lean sources of the project (not the
thousands of proof files under Outside/) and returns the source of each definition a statement
mentions, plus one level of the definitions those refer to, within a character budget.
"""
from __future__ import annotations

import re
from pathlib import Path

_DECL = re.compile(
    r"^(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|noncomputable|partial|nonrec)\s+)*"
    r"(def|abbrev|structure|inductive|class|instance|theorem|lemma|opaque)\s+([^\s:({\[]+)")
_NAMESPACE = re.compile(r"^namespace\s+(\S+)")
_SECTION = re.compile(r"^(?:noncomputable\s+)?section\b")
_END = re.compile(r"^end\b")
_QUALIFIED = re.compile(r"\bPEqualNP(?:\.[A-Za-z_][A-Za-z0-9_']*)+")
_IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*")
_MAX_LINES = 45


class DefinitionIndex:
    def __init__(self, lean_dir: str | Path, package: str = "PEqualNP") -> None:
        self.decls: dict[str, tuple[str, list[str]]] = {}  # full name -> (snippet, namespace parts)
        root = Path(lean_dir) / package
        if not root.exists():
            return
        for path in sorted(root.rglob("*.lean")):
            if "Outside" in path.parts:
                continue
            try:
                self._scan(path.read_text(encoding="utf-8").split("\n"))
            except (OSError, UnicodeDecodeError):
                continue

    def _scan(self, lines: list[str]) -> None:
        scopes: list[list[str]] = []  # each entry: the namespace parts it opened ([] for a section)
        for i, line in enumerate(lines):
            m = _NAMESPACE.match(line)
            if m:
                scopes.append(m.group(1).split("."))
                continue
            if _SECTION.match(line):
                scopes.append([])
                continue
            if _END.match(line):
                if scopes:
                    scopes.pop()
                continue
            m = _DECL.match(line)
            if not m:
                continue
            ns = [p for scope in scopes for p in scope]
            name = m.group(2)
            full = name if name.startswith("PEqualNP.") or name.startswith("_root_.") else ".".join(ns + [name])
            full = full.removeprefix("_root_.")
            start = i
            # Keep a doc comment (/-- ... -/) or `--` lines sitting right above the declaration.
            if start > 0 and lines[start - 1].rstrip().endswith("-/"):
                k = start - 1
                while k > 0 and "/-" not in lines[k] and start - k < 12:
                    k -= 1
                if "/-" in lines[k]:
                    start = k
            while start > 0 and lines[start - 1].lstrip().startswith("--") and i - start < 12:
                start -= 1
            end = i + 1
            while end < len(lines) and end - i < _MAX_LINES:
                nxt = lines[end]
                if nxt and not nxt[0].isspace() and (_DECL.match(nxt) or _NAMESPACE.match(nxt) or
                                                    _END.match(nxt) or _SECTION.match(nxt) or
                                                    nxt.startswith(("@[", "/--", "theorem", "lemma", "open ",
                                                                    "variable", "set_option", "attribute"))):
                    break
                end += 1
            snippet = "\n".join(lines[start:end]).rstrip()
            self.decls.setdefault(full, (snippet, ns))

    def resolve(self, name: str, ns: list[str] | None = None) -> str | None:
        if name in self.decls:
            return name
        for k in range(len(ns or []), 0, -1):
            cand = ".".join((ns or [])[:k] + [name])
            if cand in self.decls:
                return cand
        parts = name.split(".")
        while len(parts) > 2:  # a field or a dotted function: fall back to its structure
            parts = parts[:-1]
            if ".".join(parts) in self.decls:
                return ".".join(parts)
        return None

    def for_statement(self, statement: str, budget: int = 6000) -> list[tuple[str, str]]:
        chosen: list[str] = []
        seen: set[str] = set()

        def add(full: str | None) -> None:
            if full and full not in seen:
                seen.add(full)
                chosen.append(full)

        first = [self.resolve(n) for n in dict.fromkeys(_QUALIFIED.findall(statement))]
        for full in first:
            add(full)
        # Dot calls such as `C.eval` or `x4.size`: look them up on the structures already chosen.
        dotted = dict.fromkeys(re.findall(r"\.([a-z][A-Za-z0-9_']*)\b", statement))
        for full in list(chosen):
            for method in dotted:
                cand = f"{full}.{method}"
                if cand in self.decls:
                    add(cand)
        # One level deeper: what the chosen definitions mention.
        for full in list(chosen):
            snippet, ns = self.decls[full]
            for ident in dict.fromkeys(_IDENT.findall(snippet)):
                if ident[0].isupper() or ident.startswith("PEqualNP."):
                    add(self.resolve(ident, ns))
        out, used = [], 0
        for full in chosen:
            snippet = self.decls[full][0]
            if used + len(snippet) > budget:
                continue
            out.append((full, snippet))
            used += len(snippet)
        return out
