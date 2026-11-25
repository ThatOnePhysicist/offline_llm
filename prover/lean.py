"""Building the proof file, the site's format rules, and checking with Lean."""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .extract import Candidate
from .jobs import Job

STANDARD_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
ALLOWED_OPTIONS = ("maxRecDepth", "maxHeartbeats", "synthInstance.maxSize", "synthInstance.maxHeartbeats")
MAX_FILE_BYTES = 512 * 1024

# Format rules from https://pequalnp.com/join.txt (rules a, c, d), plus anything that would
# need an extra axiom. A file that breaks one of these is never treated as a finished proof.
_FORBIDDEN = [
    (re.compile(r"\bsorry\b"), "contains sorry"),
    (re.compile(r"\badmit\b"), "contains admit"),
    (re.compile(r"\bnative_decide\b"), "uses native_decide, which is not allowed"),
    (re.compile(r"\bbv_decide\b"), "uses bv_decide, which needs native evaluation"),
    (re.compile(r"decide\s*\+\s*native"), "uses decide +native, which is not allowed"),
    (re.compile(r"(?m)^\s*(?:private\s+|noncomputable\s+)*axiom\b"), "declares an axiom"),
    (re.compile(r"(?m)^\s*#[A-Za-z]"), "has a # command (#eval, #check, ...), which is not allowed"),
    (re.compile(r"(?m)^\s*(?:local\s+|scoped\s+)?(?:macro|macro_rules|syntax|notation|infixl?|infixr|prefix|"
                r"postfix|elab|elab_rules|declare_syntax_cat|initialize|builtin_initialize)\b"),
     "defines new syntax or runs code while compiling, which is not allowed"),
    (re.compile(r"\b(?:run_cmd|run_tac|run_elab|run_meta)\b"), "runs code while compiling, which is not allowed"),
    (re.compile(r"(?m)^\s*(?:local\s+|scoped\s+)?attribute\b"), "uses the attribute command, which is not allowed"),
    (re.compile(r"\b(?:unsafe|implemented_by|extern)\b"), "uses unsafe code"),
    (re.compile(r"ofReduceBool|trustCompiler|Lean\.reduceBool"), "relies on compiled evaluation (an extra axiom)"),
    (re.compile(r'"'), "contains a double quote; string literals are not allowed"),
]
_ATTRIBUTE = re.compile(r"@\[([^\]]*)\]")
_SET_OPTION_LINE = re.compile(r"^set_option (%s) \d+ in$" % "|".join(re.escape(o) for o in ALLOWED_OPTIONS))
_MESSAGE = re.compile(r"^(?P<file>[^\n]*?\.lean):(?P<line>\d+):(?P<col>\d+): (?P<sev>error|warning|info)[^:\n]*: ", re.M)


def assemble(job: Job, cand: Candidate) -> str:
    """The complete proof file: imports, opens, helpers in their own namespace, then the theorem."""
    out: list[str] = []
    if job.import_mathlib:
        out.append("import Mathlib")
    out += [f"import {m}" for m in job.imports]
    out.append("")
    if job.preamble:
        out += [job.preamble.strip(), ""]
    opens = list(dict.fromkeys(o for o in cand.opens if o))
    if opens:
        out += opens + [""]
    helpers = cand.helpers.strip("\n")
    if helpers.strip():
        out += [f"namespace {job.module}", "", helpers, "", f"end {job.module}", "", f"open {job.module} in"]
    out += cand.main_options
    out.append(f"{job.header} :={cand.proof.rstrip()}")
    return "\n".join(out) + "\n"


def skeleton(job: Job) -> str:
    return assemble(job, Candidate(proof=" by\n  sorry"))


def format_problems(text: str) -> list[str]:
    """Violations of the submission format rules; empty means the file is acceptable."""
    problems: list[str] = []
    for pattern, message in _FORBIDDEN:
        if pattern.search(text) and message not in problems:
            problems.append(message)
    for attr in _ATTRIBUTE.findall(text):
        if attr.strip() not in ("simp", "local simp"):
            problems.append(f"uses the attribute @[{attr.strip()}]; only @[simp] and @[local simp] are allowed")
    lines = text.split("\n")
    seen_code = False
    for k, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^import\s", line):
            if seen_code:
                problems.append("an import that is not at the top of the file")
            continue
        if stripped:
            seen_code = True
        if "set_option" in line:
            if not _SET_OPTION_LINE.match(line.rstrip()):
                problems.append("set_option is only allowed as its own top-level line `set_option <name> <number> in` "
                                "for maxRecDepth, maxHeartbeats, synthInstance.maxSize or synthInstance.maxHeartbeats")
            else:
                nxt = next((l for l in lines[k + 1:] if l.strip()), "")
                if not re.match(r"^(set_option|theorem|lemma|def|abbrev|private|protected|noncomputable|@\[|instance|structure|inductive)\b", nxt):
                    problems.append("a set_option line must sit directly above the declaration it serves")
    if len(text.encode("utf-8")) > MAX_FILE_BYTES:
        problems.append("the file is larger than 512 KiB")
    return list(dict.fromkeys(problems))


@dataclass
class CheckResult:
    ok: bool
    errors: list[str]
    output: str
    seconds: float
    timed_out: bool = False
    uses_sorry: bool = False
    axioms: list[str] | None = None
    messages: list[dict] = field(default_factory=list)

    def feedback(self, limit: int = 12) -> list[str]:
        out = list(self.errors[:limit])
        if self.timed_out:
            out.append("Lean did not finish within the time limit; the proof is too slow (avoid huge decide or simp calls)")
        if self.uses_sorry:
            out.append("the proof still uses sorry")
        if self.axioms is not None:
            extra = [a for a in self.axioms if a not in STANDARD_AXIOMS]
            if extra:
                out.append("the proof depends on non-standard axioms: " + ", ".join(extra))
        return out


def find_lake() -> str | None:
    found = shutil.which("lake")
    if found:
        return found
    candidate = Path.home() / ".elan" / "bin" / "lake"
    return str(candidate) if candidate.exists() else None


class LeanProject:
    """A checkout of the pequalnp-proofs Lean project (or any Lake project) used for checking."""

    def __init__(self, repo: str | Path, timeout: float = 900.0, lake: str | None = None) -> None:
        self.repo = Path(repo).expanduser().resolve()
        self.lean_dir = self.repo / "lean" if (self.repo / "lean" / "lakefile.toml").exists() else self.repo
        if not any((self.lean_dir / f).exists() for f in ("lakefile.toml", "lakefile.lean")):
            raise FileNotFoundError(f"no Lake project found at {self.repo} (expected lean/lakefile.toml)")
        self.lake = lake or find_lake()
        if not self.lake:
            raise FileNotFoundError("`lake` not found; install Lean with elan (https://github.com/leanprover/elan)")
        self.timeout = timeout

    def build(self, modules: list[str], timeout: float = 7200.0) -> tuple[bool, str]:
        """Builds the modules a proof imports (once; later runs reuse Lake's cache)."""
        targets = [m for m in modules if m != "Mathlib" and not m.startswith("Mathlib.")]
        if not targets:
            return True, ""
        proc = subprocess.run([self.lake, "build", *targets], cwd=self.lean_dir, capture_output=True,
                              text=True, timeout=timeout)
        return proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]

    def check(self, text: str, theorem: str, path: Path, timeout: float | None = None) -> CheckResult:
        """Compiles `text` (written to `path`) and reports errors, sorry and axioms of `theorem`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + f"\n#print axioms {theorem}\n", encoding="utf-8")
        start = time.monotonic()
        # Own process group, so a timeout also stops the `lean` process that lake starts.
        proc = subprocess.Popen([self.lake, "env", "lean", str(path)], cwd=self.lean_dir, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            output, _ = proc.communicate(timeout=timeout or self.timeout)
            code, timed_out = proc.returncode, False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            output, _ = proc.communicate()
            code, timed_out = -1, True
        seconds = time.monotonic() - start
        return parse_output(output, code, timed_out, seconds, theorem, text)


_AXIOM_LINE = re.compile(r"^'[^'\n]+' (?:depends on axioms: \[[^\]]*\]|does not depend on any axioms)\s*$", re.M)


def parse_output(output: str, code: int, timed_out: bool, seconds: float, theorem: str, text: str) -> CheckResult:
    lines = text.split("\n")
    messages = []
    # `#print axioms` prints without a position prefix; keep it out of the message bodies.
    body_text = _AXIOM_LINE.sub("", output)
    marks = list(_MESSAGE.finditer(body_text))
    for k, m in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(body_text)
        body = body_text[m.end():end].strip()
        messages.append({"line": int(m.group("line")), "col": int(m.group("col")), "severity": m.group("sev"), "text": body})
    errors = []
    for msg in messages:
        if msg["severity"] != "error":
            continue
        line_no = msg["line"]
        src = lines[line_no - 1].strip() if 0 < line_no <= len(lines) else ""
        where = f"line {line_no}" + (f" (`{src[:120]}`)" if src else "")
        errors.append(f"{where}: {msg['text'][:700]}")
    if code not in (0, -1) and not errors and not timed_out:
        errors.append(output.strip()[-700:] or f"lean exited with code {code}")
    uses_sorry = re.search(r"declaration uses [`']sorry[`']", output) is not None
    axioms = None
    m = re.search(r"'%s' depends on axioms: \[(.*?)\]" % re.escape(theorem), output, re.S)
    if m:
        axioms = [a.strip() for a in m.group(1).split(",") if a.strip()]
    elif re.search(r"'%s' does not depend on any axioms" % re.escape(theorem), output):
        axioms = []
    found = axioms is not None
    if not found and not errors and not timed_out:
        errors.append(f"the theorem {theorem} was not found after compiling")
    ok = (not errors and not timed_out and not uses_sorry and found
          and all(a in STANDARD_AXIOMS for a in axioms or []))
    return CheckResult(ok, errors, output, seconds, timed_out, uses_sorry, axioms, messages)
