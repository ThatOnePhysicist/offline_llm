"""A unit of work: one Lean theorem to prove, from the desk or from a local file."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_STEP_ID = re.compile(r"L-[a-z]+-\d+")
_STEP_FILE = re.compile(r"lean/PEqualNP/Outside/[A-Za-z0-9_]+\.lean")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_']*(\.[A-Za-z_][A-Za-z0-9_']*)*")


@dataclass
class Job:
    id: str
    title: str
    module: str                       # full name of the theorem to prove
    signature: str                    # everything between the name and `:=`, e.g. ": ∀ n, ..."
    imports: list[str]                # modules imported besides Mathlib
    file: str | None = None           # repository path of the proof (desk steps)
    context: str = ""
    sketch: str = ""
    dead: list[str] = field(default_factory=list)
    preamble: str = ""                # local problems: definitions that come before the theorem
    lease_id: str | None = None
    raw: dict | None = None
    import_mathlib: bool = True

    @property
    def is_search_shard(self) -> bool:
        return bool(self.raw and ("shard" in self.raw or (self.raw.get("gap") or {}).get("shard_spec")))

    @property
    def header(self) -> str:
        return f"theorem {self.module} {self.signature}"

    @classmethod
    def from_desk(cls, answer: dict) -> "Job":
        g = answer["gap"]
        gid, path, module = g["id"], g["file"], g["module"]
        if not _STEP_ID.fullmatch(gid) or not _STEP_FILE.fullmatch(path) or not _NAME.fullmatch(module):
            raise ValueError("the desk's answer has an unexpected shape")
        imports = [m for m in (g.get("imports") or []) if re.fullmatch(r"[A-Za-z0-9_.]+", m)]
        sketch = g.get("sketch") or ""
        context = g.get("context") or ""
        if sketch and sketch in context:
            sketch = ""  # the desk often repeats the sketch inside the context
        dead = [d if isinstance(d, str) else str(d) for d in (g.get("dead") or [])]
        return cls(id=gid, title=g.get("title", ""), module=module, signature=": " + g["statement"],
                   imports=imports, file=path, context=context, sketch=sketch, dead=dead,
                   lease_id=(answer.get("lease") or {}).get("id"), raw=answer)

    @classmethod
    def from_file(cls, path: str | Path) -> "Job":
        """A local problem: imports, optional definitions, then one theorem ending in `sorry`.

            import Mathlib
            def double (n : Nat) := n + n
            theorem double_even (n : Nat) : Even (double n) := by
              sorry
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        imports, body = [], []
        for line in text.splitlines():
            m = re.match(r"\s*import\s+(\S+)\s*$", line)
            if m and not any(b.strip() for b in body):
                imports.append(m.group(1))
            else:
                body.append(line)
        rest = "\n".join(body)
        found = list(re.finditer(r"(?m)^theorem\s+([^\s:({\[]+)(.*?):=\s*by\s+sorry\s*$", rest, re.S))
        if not found:
            raise ValueError(f"{path}: expected a `theorem NAME ... := by sorry` at the end")
        m = found[-1]
        preamble = rest[:m.start()].strip()
        return cls(id=f"local:{path.stem}", title=f"local problem {path.name}", module=m.group(1),
                   signature=m.group(2).strip(), imports=[i for i in imports if i != "Mathlib"],
                   preamble=preamble, import_mathlib="Mathlib" in imports)
