"""Turns a model reply into a candidate proof of one given theorem.

The statement is never taken from the model: the worker keeps the exact text it was given and
only takes the model's helper lemmas and the proof after `:=`. A reply that changed the
statement is rejected with a message the model sees on its next attempt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .jobs import Job

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)(?:```|\Z)", re.S)
_DECL = re.compile(
    r"(?m)^(?:@\[[^\]\n]*\]\s*)?(?:(?:private|protected|noncomputable|nonrec)\s+)*"
    r"(?:theorem|lemma)\s+(?P<name>[^\s:({\[]+)")
_TOP_LEVEL = re.compile(
    r"^(?:@\[|theorem\b|lemma\b|def\b|abbrev\b|example\b|instance\b|structure\b|inductive\b|class\b|"
    r"namespace\b|section\b|end\b|open\b|set_option\b|private\b|protected\b|noncomputable\b|variable\b|"
    r"universe\b|attribute\b|#|import\b|axiom\b|opaque\b|macro\b|syntax\b|notation\b)")


class CandidateError(Exception):
    """The reply cannot be used; the message is shown to the model next time."""


@dataclass
class Candidate:
    proof: str                                       # text after `:=` of the main theorem
    helpers: str = ""                                # declarations before the main theorem
    opens: list[str] = field(default_factory=list)   # `open ...` lines, placed at the top
    main_options: list[str] = field(default_factory=list)  # `set_option ... in` above the theorem


def code_blocks(text: str) -> list[tuple[str, str]]:
    return [(m.group(1).lower(), m.group(2)) for m in _FENCE.finditer(text)]


def pick_code(text: str) -> str:
    """The last Lean code block that contains a theorem (models often sketch code in the plan)."""
    blocks = [b for lang, b in code_blocks(text) if lang in ("lean4", "lean", "")]
    for body in reversed(blocks):
        if _DECL.search(strip_comments(body)):
            return body
    if blocks:
        return blocks[-1]
    if _DECL.search(text):
        return text
    raise CandidateError("no Lean code block was found in the reply")


def strip_comments(code: str) -> str:
    """Removes `--` line comments and (nested) `/- -/` block comments, keeping line breaks."""
    out: list[str] = []
    i, n, depth = 0, len(code), 0
    while i < n:
        if code.startswith("/-", i):
            depth += 1
            i += 2
        elif depth:
            if code.startswith("-/", i):
                depth -= 1
                i += 2
            else:
                if code[i] == "\n":
                    out.append("\n")
                i += 1
        elif code.startswith("--", i):
            j = code.find("\n", i)
            i = n if j == -1 else j
        else:
            out.append(code[i])
            i += 1
    return "".join(out)


def _match_signature(code: str, pos: int, signature: str) -> int | None:
    """Matches `signature` at `pos` ignoring whitespace; returns the end position."""
    want = "".join(signature.split())
    k, i = 0, pos
    while k < len(want):
        while i < len(code) and code[i].isspace():
            i += 1
        if i >= len(code) or code[i] != want[k]:
            return None
        i += 1
        k += 1
    return i


def _short(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def candidate_from(reply: str, job: Job) -> Candidate:
    code = strip_comments(pick_code(reply))
    lines = code.splitlines()
    decls = list(_DECL.finditer(code))
    if not decls:
        raise CandidateError("the code block has no `theorem`; write the full theorem with its proof")

    def preference(m: re.Match) -> int:
        name = m.group("name")
        if name == job.module:
            return 3
        if _short(name) == _short(job.module):
            return 2
        return 1

    main = end_of_sig = None
    for m in sorted(reversed(decls), key=preference, reverse=True):
        end = _match_signature(code, m.end(), job.signature)
        if end is not None:
            main, end_of_sig = m, end
            break
    if main is None:
        raise CandidateError(
            "the reply changed the theorem statement; copy the statement exactly as given and only write the proof")

    rest = code[end_of_sig:]
    stripped = rest.lstrip()
    if not stripped.startswith(":="):
        raise CandidateError("expected `:=` right after the theorem statement")
    proof_and_tail = stripped[2:]

    # The proof runs until the next declaration that starts at column 0.
    proof_lines, tail_started = [], False
    for k, line in enumerate(proof_and_tail.split("\n")):
        if k > 0 and line and not line[0].isspace() and _TOP_LEVEL.match(line):
            tail_started = True
        if tail_started:
            break
        proof_lines.append(line)
    proof = "\n".join(proof_lines).rstrip()
    if not proof.strip():
        raise CandidateError("the theorem has no proof after `:=`")
    if not proof.startswith((" ", "\n")):
        proof = " " + proof

    # Everything before the main theorem: helpers, open lines, options for the main theorem.
    head_lines = code[:main.start()].split("\n")
    opens, helpers, options = [], [], []
    for line in head_lines:
        s = line.strip()
        if s.startswith("import "):
            continue
        if re.match(r"^(namespace|section|end)\b", line) or re.match(r"^noncomputable\s+section\b", line):
            continue  # the worker puts helpers in its own namespace
        if re.match(r"^open\s", line) and not s.endswith(" in"):
            opens.append(s)
            continue
        helpers.append(line)
    while helpers and not helpers[-1].strip():
        helpers.pop()
    while helpers and re.match(r"^set_option\s+\S+\s+\S+\s+in\s*$", helpers[-1]):
        options.insert(0, helpers.pop().strip())
    return Candidate(proof=proof, helpers="\n".join(helpers).strip("\n"), opens=opens, main_options=options)
