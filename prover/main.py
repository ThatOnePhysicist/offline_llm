"""The continuous worker: take a step, try to prove it with the local model, check with Lean, repeat."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .defs import DefinitionIndex
from .desk import Desk, DeskError, Heartbeat
from .engine import GenerationSettings, LocalModel, ScriptedModel
from .extract import CandidateError, candidate_from
from .jobs import Job
from .lean import LeanProject, assemble, format_problems, skeleton
from .prompts import first_prompt, repair_prompt

try:
    from rich.console import Console
    _console = Console()

    def say(message: str, style: str | None = None) -> None:
        _console.print(f"[dim]{datetime.now():%H:%M:%S}[/dim] ", end="")
        _console.print(message, style=style, markup=False, highlight=False)
except ImportError:  # rich is optional here
    def say(message: str, style: str | None = None) -> None:
        print(f"{datetime.now():%H:%M:%S} {message}", flush=True)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


class RunLog:
    """Everything the worker did, on disk: events.jsonl plus each prompt, reply and Lean file."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.events = self.root / "events.jsonl"

    def event(self, kind: str, **data) -> None:
        record = {"time": datetime.now().isoformat(timespec="seconds"), "event": kind, **data}
        with self.events.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def job_dir(self, job: Job) -> Path:
        d = self.root / "steps" / _safe(job.id)
        d.mkdir(parents=True, exist_ok=True)
        return d


@dataclass
class Outcome:
    status: str                  # proved | out_of_time | gone | stopped | does_not_elaborate
    text: str = ""
    attempts: int = 0
    notes: list[str] = field(default_factory=list)


class Worker:
    def __init__(self, model, lean: LeanProject, runs: RunLog, minutes: float, max_attempts: int = 0,
                 max_repairs: int = 2, check_timeout: float = 600.0, stop: threading.Event | None = None) -> None:
        self.model = model
        self.lean = lean
        self.runs = runs
        self.minutes = minutes
        self.max_attempts = max_attempts
        self.max_repairs = max_repairs
        self.check_timeout = check_timeout
        self.stop = stop or threading.Event()
        self.index = DefinitionIndex(lean.lean_dir)
        s = getattr(model, "settings", GenerationSettings())
        # Prompt room in characters (Lean text runs about 3 characters per token).
        self.budget = max(4000, (s.ctx_size - s.max_tokens - 256) * 3)

    def work(self, job: Job, heartbeat: Heartbeat | None = None) -> Outcome:
        deadline = time.monotonic() + self.minutes * 60
        jd = self.runs.job_dir(job)

        def interrupted() -> str | None:
            if self.stop.is_set():
                return "stopped"
            if heartbeat is not None and heartbeat.gone.is_set():
                return "gone"
            return None

        ok, log = self.lean.build(job.imports)
        if not ok:
            say("building the imported modules failed:\n" + log[-1500:], "red")
            return Outcome("out_of_time", notes=["imports did not build"])
        skel = skeleton(job)
        res = self.lean.check(skel, job.module, jd / "skeleton.lean", timeout=self.check_timeout)
        self.runs.event("skeleton", step=job.id, seconds=round(res.seconds, 1), errors=res.errors[:3])
        if res.timed_out:
            say("Lean could not even check the bare statement within the time limit", "red")
            return Outcome("out_of_time", notes=["the statement alone timed out"])
        if res.errors:
            say("the statement does not elaborate: " + res.errors[0][:300], "red")
            return Outcome("does_not_elaborate", notes=res.errors[:3])
        check_seconds = max(res.seconds, 10.0)
        defs = self.index.for_statement(job.signature + "\n" + job.preamble, budget=self.budget // 2)
        say(f"statement checks in {res.seconds:.0f}s; {len(defs)} project definitions go into the prompt")

        attempt = repairs = 0
        last_text: str | None = None
        last_feedback: list[str] = []
        note: str | None = None
        while True:
            why = interrupted()
            if why:
                return Outcome(why, attempts=attempt)
            if deadline - time.monotonic() < check_seconds + 60:
                break
            if self.max_attempts and attempt >= self.max_attempts:
                break
            attempt += 1
            if last_text is None or repairs >= self.max_repairs:
                messages, mode, repairs = first_prompt(job, defs, self.budget, note), "fresh", 0
            else:
                messages, mode = repair_prompt(job, defs, last_text, last_feedback, self.budget), "repair"
                repairs += 1
            note = None
            say(f"attempt {attempt} ({mode}), {int((deadline - time.monotonic()) // 60)} min left")
            gen = self.model.chat(messages, should_stop=lambda: bool(interrupted()) or time.monotonic() > deadline)
            (jd / f"attempt_{attempt:02d}_prompt.txt").write_text(messages[-1][1], encoding="utf-8")
            (jd / f"attempt_{attempt:02d}_reply.txt").write_text(gen.text, encoding="utf-8")
            say(f"  {gen.tokens} tokens in {gen.seconds:.0f}s ({gen.tokens_per_second:.1f}/s), stop: {gen.stop_reason}")
            if gen.stop_reason == "context_full" and gen.tokens == 0:
                # The prompt alone filled the context: give later prompts less room.
                self.budget = max(2000, int(self.budget * 0.75))
                say(f"  the prompt did not fit the context; shrinking prompts to {self.budget} characters")
                last_text = None
                continue
            try:
                try:
                    cand = candidate_from(gen.answer, job)
                except CandidateError:
                    cand = candidate_from(gen.text, job)  # the code may sit inside the reasoning
            except CandidateError as err:
                say(f"  unusable reply: {err}")
                self.runs.event("attempt", step=job.id, n=attempt, mode=mode, result="unusable", detail=str(err))
                last_text, note = None, str(err)
                continue
            text = assemble(job, cand)
            problems = format_problems(text)
            if problems:
                say("  breaks the format rules: " + "; ".join(problems))
                self.runs.event("attempt", step=job.id, n=attempt, mode=mode, result="format", detail=problems)
                (jd / f"attempt_{attempt:02d}.lean").write_text(text, encoding="utf-8")
                last_text, last_feedback = text, problems
                continue
            res = self.lean.check(text, job.module, jd / f"attempt_{attempt:02d}.lean", timeout=self.check_timeout)
            (jd / f"attempt_{attempt:02d}_lean.txt").write_text(res.output, encoding="utf-8")
            check_seconds = max(check_seconds, res.seconds)
            self.runs.event("attempt", step=job.id, n=attempt, mode=mode, result="proved" if res.ok else "failed",
                            errors=len(res.errors), seconds=round(res.seconds, 1), tokens=gen.tokens)
            if res.ok:
                (jd / "PROVED.lean").write_text(text, encoding="utf-8")
                return Outcome("proved", text=text, attempts=attempt)
            feedback = res.feedback()
            say(f"  Lean: {len(res.errors)} error(s)" + (f"; first: {feedback[0][:200]}" if feedback else ""))
            last_text, last_feedback = text, feedback
        return Outcome("out_of_time", attempts=attempt)


def notify(message: str) -> None:
    say(message, "bold green")
    if sys.platform == "darwin" and shutil.which("osascript"):
        safe = re.sub(r"[^A-Za-z0-9 .,:_()-]", " ", message)[:200]
        subprocess.run(["osascript", "-e", f'display notification "{safe}" with title "offline_llm prover"'],
                       check=False, capture_output=True)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)


def _default_branch(repo: Path) -> str:
    ref = _git(repo, "symbolic-ref", "--short", "refs/remotes/origin/HEAD").stdout.strip()
    return ref.split("/", 1)[1] if "/" in ref else "main"


def _github_handle() -> str | None:
    if not shutil.which("gh"):
        return None
    out = subprocess.run(["gh", "api", "user", "--jq", ".login"], capture_output=True, text=True)
    return out.stdout.strip() or None


def preflight(repo: Path, submit: bool) -> list[str]:
    problems = []
    if not (repo / "tools" / "pnp.sh").exists():
        problems.append(f"{repo} does not look like a clone of pequalnp-proofs (no tools/pnp.sh)")
        return problems
    dirty = [l for l in _git(repo, "status", "--porcelain").stdout.splitlines() if not l.endswith(".pnp/")
             and ".pnp/" not in l]
    if dirty:
        problems.append("the pequalnp-proofs checkout has uncommitted changes (a pull request must add one file): "
                        + "; ".join(dirty[:5]))
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch != _default_branch(repo):
        problems.append(f"the checkout is on branch {branch}; switch to {_default_branch(repo)} first")
    if submit:
        origin = _git(repo, "remote", "get-url", "origin").stdout.strip()
        if "open-goat/pequalnp-proofs" in origin:
            problems.append("origin points at open-goat/pequalnp-proofs; clone YOUR fork so pull requests come from it")
        if not shutil.which("gh") or subprocess.run(["gh", "auth", "status"], capture_output=True).returncode != 0:
            problems.append("--submit needs the GitHub CLI signed in: gh auth login")
    return problems


def deliver(job: Job, text: str, repo: Path, runs: RunLog, submit: bool) -> bool:
    """Writes the proof where the pull request expects it; with submit, opens the pull request."""
    target = repo / job.file
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    proved_dir = runs.root / "proved"
    proved_dir.mkdir(exist_ok=True)
    (proved_dir / f"{_safe(job.id)}.lean").write_text(text, encoding="utf-8")
    runs.event("proved", step=job.id, file=str(target))
    notify(f"Proof of {job.id} checked locally")
    if not submit:
        say(f"The proof is at {target}. To open the pull request:\n  cd {repo} && tools/pnp.sh submit {job.id}", "green")
        return True
    proc = subprocess.run(["bash", "tools/pnp.sh", "submit", job.id], cwd=repo, capture_output=True, text=True)
    say(proc.stdout.strip() or proc.stderr.strip())
    if proc.returncode != 0:
        say("submitting failed; the proof file is still in place:\n" + proc.stderr[-1500:], "red")
        runs.event("submit_failed", step=job.id, stderr=proc.stderr[-1500:])
        return False
    _git(repo, "checkout", "-q", _default_branch(repo))
    runs.event("submitted", step=job.id)
    return True


def run_desk(args, worker: Worker, runs: RunLog, stop: threading.Event) -> int:
    repo = Path(args.repo).expanduser().resolve()
    handle = args.handle or os.environ.get("PNP_HANDLE") or _github_handle()
    if not handle:
        say("no GitHub handle: pass --handle or sign in with gh auth login", "red")
        return 2
    problems = preflight(repo, args.submit)
    if problems:
        for p in problems:
            say(p, "red")
        return 2
    desk = Desk(handle)
    say(f"working for {handle}; {args.minutes:g} minutes per step; Ctrl-C hands the current step back and stops")
    backoff = 60
    while not stop.is_set():
        try:
            offer = desk.next(hard=args.hard)
        except DeskError as err:
            say(f"{err}; trying again in {backoff // 60} min", "yellow")
            stop.wait(backoff)
            backoff = min(backoff * 2, 1800)
            continue
        backoff = 60
        if not offer.gap:
            wait = max(offer.retry_after, 120)
            say(f"no step for now: {offer.note or 'nothing free'} (asking again in {wait // 60} min)")
            runs.event("wait", note=offer.note, seconds=wait)
            stop.wait(wait)
            continue
        try:
            job = Job.from_desk(offer.raw)
        except (KeyError, ValueError) as err:
            say(f"unexpected answer from the desk ({err}); stopping", "red")
            return 1
        state = repo / ".pnp"
        state.mkdir(exist_ok=True)
        (state / f"{job.id}.json").write_text(json.dumps(offer.raw, indent=1), encoding="utf-8")
        say(f"took {job.id}: {job.title}", "bold")
        runs.event("take", step=job.id, title=job.title, lease=job.lease_id)
        if job.is_search_shard:
            desk.release(job.lease_id, "other")
            say("  a range of a search: that needs a search program, not a proof; handed back")
            runs.event("release", step=job.id, reason="other", why="search shard")
            continue
        beat = Heartbeat(desk, job.lease_id, log=say)
        beat.start()
        try:
            outcome = worker.work(job, beat)
        except KeyboardInterrupt:
            beat.stop()
            desk.release(job.lease_id, "out_of_time")
            runs.event("release", step=job.id, reason="out_of_time", why="interrupted")
            raise
        except Exception:
            beat.stop()
            err = traceback.format_exc()
            say("the worker hit an error; handing the step back:\n" + err[-1500:], "red")
            desk.release(job.lease_id, "out_of_time")
            runs.event("error", step=job.id, traceback=err[-3000:])
            stop.wait(60)
            continue
        beat.stop()
        runs.event("outcome", step=job.id, status=outcome.status, attempts=outcome.attempts)
        if outcome.status == "proved":
            if not deliver(job, outcome.text, repo, runs, args.submit):
                return 1
            if not args.submit:
                return 0  # stop so the proof gets looked at and submitted
        elif outcome.status == "gone":
            say(f"{job.id} was settled or withdrawn while we worked on it")
        else:
            reason = "does_not_elaborate" if outcome.status == "does_not_elaborate" else "out_of_time"
            desk.release(job.lease_id, reason)
            say(f"handed {job.id} back ({reason}) after {outcome.attempts} attempt(s)")
            runs.event("release", step=job.id, reason=reason, attempts=outcome.attempts)
        if args.once or outcome.status == "stopped":
            break
    return 0


def run_local(args, worker: Worker, runs: RunLog, stop: threading.Event) -> int:
    path = Path(args.problem).expanduser().resolve()
    job = Job.from_file(path)
    say(f"working on {job.module} from {path.name}; {args.minutes:g} minutes per round, Ctrl-C to stop")
    rounds = 0
    while not stop.is_set():
        rounds += 1
        outcome = worker.work(job)
        runs.event("outcome", step=job.id, status=outcome.status, attempts=outcome.attempts, round=rounds)
        if outcome.status == "proved":
            out = path.with_name(path.stem + ".proved.lean")
            out.write_text(outcome.text, encoding="utf-8")
            notify(f"Proved {job.module}; written to {out.name}")
            return 0
        if outcome.status == "does_not_elaborate":
            say("fix the problem file first: " + "; ".join(outcome.notes), "red")
            return 1
        if outcome.status == "stopped" or args.once:
            break
        say(f"round {rounds} found nothing; starting another")
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="worker.py",
        description="Continuously try to prove Lean theorems with the local model: steps from the "
                    "pequalnp.com desk (default) or one local problem file (--problem).")
    p.add_argument("--model", help="path to the .gguf model (a Lean prover model works best)")
    p.add_argument("--model-name", help="display name for the model")
    p.add_argument("--repo", default=os.environ.get("PNP_REPO", "~/pequalnp-proofs"),
                   help="your clone of your fork of open-goat/pequalnp-proofs (default: $PNP_REPO or ~/pequalnp-proofs)")
    p.add_argument("--problem", help="work on this local .lean file (one theorem ending in sorry) instead of the desk")
    p.add_argument("--handle", help="GitHub handle for the desk (default: $PNP_HANDLE or gh's login)")
    p.add_argument("--minutes", type=float, default=30.0,
                   help="time on one step before handing it back (the site asks for at least 20); per round with --problem")
    p.add_argument("--attempts", type=int, default=0, help="most attempts per step (default: as many as fit)")
    p.add_argument("--repairs", type=int, default=2, help="repair rounds on one attempt before starting fresh")
    p.add_argument("--ctx", type=int, default=8192, help="context window in tokens")
    p.add_argument("--max-tokens", type=int, default=4096, help="most tokens per reply")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--template", default="auto", help="chat format: auto (from the model), chatml, plain, or a llama.cpp template name")
    p.add_argument("--check-timeout", type=float, default=600.0, help="seconds Lean may take on one attempt")
    p.add_argument("--submit", action="store_true", help="open the pull request automatically (tools/pnp.sh submit) and keep going")
    p.add_argument("--hard", action="store_true", help="ask the desk for the open problems the attack is stuck on")
    p.add_argument("--once", action="store_true", help="stop after one step (or one round)")
    p.add_argument("--runs", default="prover_runs", help="where prompts, replies and Lean output are kept")
    p.add_argument("--scripted", help=argparse.SUPPRESS)  # testing: canned replies instead of a model
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model and not args.scripted:
        say("pass --model /path/to/model.gguf", "red")
        return 2
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    runs = RunLog(args.runs)
    try:
        repo = Path(args.repo).expanduser()
        lean = LeanProject(repo, timeout=args.check_timeout)
    except FileNotFoundError as err:
        say(str(err), "red")
        return 2
    settings = GenerationSettings(ctx_size=args.ctx, max_tokens=args.max_tokens, temperature=args.temperature)
    if args.scripted:
        model = ScriptedModel.from_file(args.scripted)
    else:
        model = LocalModel(args.model, args.model_name, settings, args.template, log=say)
    worker = Worker(model, lean, runs, args.minutes, args.attempts, args.repairs, args.check_timeout, stop)
    try:
        if args.problem:
            return run_local(args, worker, runs, stop)
        return run_desk(args, worker, runs, stop)
    except KeyboardInterrupt:
        say("stopped", "yellow")
        return 130
