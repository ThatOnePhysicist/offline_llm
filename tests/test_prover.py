"""Tests for the prover worker.  Run: python -m unittest discover -s tests -v

The last two tests compile real Lean files; they run only when PNP_REPO points at a
pequalnp-proofs checkout whose Lean project is built (and `lake` is installed).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prover import main as prover_main  # noqa: E402
from prover.engine import ScriptedModel, strip_reasoning  # noqa: E402
from prover.extract import CandidateError, candidate_from, strip_comments  # noqa: E402
from prover.jobs import Job  # noqa: E402
from prover.lean import CheckResult, LeanProject, assemble, format_problems, parse_output, skeleton  # noqa: E402

STATEMENT = "∀ (n : ℕ), n + 0 = n"
DESK_ANSWER = {
    "gap": {"id": "L-meta-9999", "title": "adding zero", "statement": STATEMENT,
            "file": "lean/PEqualNP/Outside/L_meta_9999.lean", "module": "PEqualNP.Outside.L_meta_9999",
            "imports": ["PEqualNP.Defs"], "context": "SKETCH: use simp.", "sketch": "use simp.", "dead": []},
    "lease": {"id": "lease-1", "expires_at": "2026-11-25T03:00:00Z"},
}


def job() -> Job:
    return Job.from_desk(DESK_ANSWER)


REPLY = """<think>The goal is trivial.</think>
### Proof plan
Induction is not needed; `Nat.add_zero` closes it.

```lean4
import Mathlib
import PEqualNP.Defs

open Nat

/-- a helper -/
lemma my_helper (n : ℕ) : n + 0 = n := Nat.add_zero n

set_option maxHeartbeats 400000 in
theorem PEqualNP.Outside.L_meta_9999 : ∀ (n : ℕ),
    n + 0 = n := by
  intro n
  exact my_helper n -- done
```
That completes the proof."""


class ExtractTests(unittest.TestCase):
    def test_candidate_keeps_statement_and_takes_proof(self):
        cand = candidate_from(strip_reasoning(REPLY), job())
        self.assertIn("exact my_helper n", cand.proof)
        self.assertNotIn("done", cand.proof)            # comments are stripped
        self.assertIn("lemma my_helper", cand.helpers)
        self.assertEqual(cand.opens, ["open Nat"])
        self.assertEqual(cand.main_options, ["set_option maxHeartbeats 400000 in"])
        text = assemble(job(), cand)
        self.assertIn("theorem PEqualNP.Outside.L_meta_9999 : " + STATEMENT + " :=", text)
        self.assertIn("namespace PEqualNP.Outside.L_meta_9999", text)
        self.assertIn("open PEqualNP.Outside.L_meta_9999 in\nset_option maxHeartbeats 400000 in\ntheorem", text)
        self.assertEqual(format_problems(text), [])

    def test_changed_statement_is_rejected(self):
        reply = "```lean4\ntheorem PEqualNP.Outside.L_meta_9999 : ∀ (n : ℕ), 0 + n = n := by\n  simp\n```"
        with self.assertRaises(CandidateError):
            candidate_from(reply, job())

    def test_short_name_is_accepted(self):
        reply = "```lean\ntheorem L_meta_9999 : ∀ (n : ℕ), n + 0 = n := by simp\n```"
        self.assertEqual(candidate_from(reply, job()).proof.strip(), "by simp")

    def test_no_code(self):
        with self.assertRaises(CandidateError):
            candidate_from("I could not find a proof.", job())

    def test_unfinished_thinking_is_dropped(self):
        self.assertEqual(strip_reasoning("<think>still going"), "")

    def test_nested_comments(self):
        self.assertEqual(strip_comments("a /- x /- y -/ z -/ b -- c\nd").split(), ["a", "b", "d"])


class FormatTests(unittest.TestCase):
    def check(self, body: str) -> list[str]:
        return format_problems("import Mathlib\n\n" + body)

    def test_forbidden(self):
        self.assertTrue(self.check("theorem x : True := by sorry"))
        self.assertTrue(self.check("theorem x : 2 = 2 := by native_decide"))
        self.assertTrue(self.check("#eval 1\ntheorem x : True := trivial"))
        self.assertTrue(self.check('theorem x : "a" = "a" := rfl'))
        self.assertTrue(self.check("@[reducible] def f := 1"))
        self.assertTrue(self.check("theorem x : True := by\n  set_option maxHeartbeats 1 in\n  trivial"))
        self.assertTrue(self.check("set_option pp.all true in\ntheorem x : True := trivial"))
        self.assertTrue(self.check("axiom bad : False"))

    def test_allowed(self):
        self.assertEqual(self.check("@[simp] lemma a : True := trivial\n\n"
                                    "set_option maxRecDepth 4000 in\ntheorem x : True := trivial"), [])


class JobTests(unittest.TestCase):
    def test_desk_shape(self):
        j = job()
        self.assertEqual(j.signature, ": " + STATEMENT)
        self.assertEqual(j.sketch, "")  # repeated inside the context
        self.assertEqual(skeleton(j), "import Mathlib\nimport PEqualNP.Defs\n\ntheorem PEqualNP.Outside.L_meta_9999 : "
                         + STATEMENT + " := by\n  sorry\n")
        bad = json.loads(json.dumps(DESK_ANSWER))
        bad["gap"]["file"] = "../../etc/passwd"
        with self.assertRaises(ValueError):
            Job.from_desk(bad)

    def test_local_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "P.lean"
            p.write_text("import Mathlib\n\ndef double (n : ℕ) := n + n\n\n"
                         "theorem double_even (n : ℕ) : Even (double n) := by\n  sorry\n", encoding="utf-8")
            j = Job.from_file(p)
            self.assertEqual((j.module, j.signature), ("double_even", "(n : ℕ) : Even (double n)"))
            self.assertIn("def double", j.preamble)
            self.assertTrue(j.import_mathlib)


class LeanOutputTests(unittest.TestCase):
    # Copied from a real run of Lean 4.36.
    OUTPUT = ("/tmp/lt/Fmt.lean:3:8: warning: declaration uses `sorry`\n"
              "/tmp/lt/Fmt.lean:7:8: error(lean.unknownIdentifier): Unknown identifier `foo_bar`\n"
              "'t1' depends on axioms: [sorryAx]\n"
              "'t3' does not depend on any axioms\n")
    TEXT = "\n".join(["import Mathlib.Tactic.Ring", "", "theorem t1 : x := by", "  sorry", "", "",
                      "  exact foo_bar a b"])

    def test_errors_sorry_axioms(self):
        r = parse_output(self.OUTPUT, 1, False, 1.0, "t1", self.TEXT)
        self.assertFalse(r.ok)
        self.assertTrue(r.uses_sorry)
        self.assertEqual(r.axioms, ["sorryAx"])
        self.assertEqual(len(r.errors), 1)
        self.assertIn("line 7 (`exact foo_bar a b`)", r.errors[0])
        self.assertNotIn("depends on axioms", r.errors[0])

    def test_clean(self):
        r = parse_output("'t3' does not depend on any axioms\n", 0, False, 1.0, "t3", "")
        self.assertTrue(r.ok)


class FakeLean:
    """Accepts a file exactly when it contains the right proof line."""

    def __init__(self, lean_dir: Path, good: str) -> None:
        self.lean_dir = lean_dir
        self.good = good
        self.checked: list[str] = []

    def build(self, modules):
        return True, ""

    def check(self, text, theorem, path, timeout=None):
        self.checked.append(text)
        if "sorry" in text:
            return CheckResult(False, [], "", 1.0, uses_sorry=True, axioms=["sorryAx"])
        if self.good in text:
            return CheckResult(True, [], "", 1.0, axioms=[])
        return CheckResult(False, ["line 5: unknown identifier `wrong_lemma`"], "", 1.0, axioms=None)


class FakeDesk:
    instances: list["FakeDesk"] = []

    def __init__(self, handle, **_):
        self.handle = handle
        self.offers = [DESK_ANSWER]
        self.released: list[tuple[str, str]] = []
        self.beats = 0
        FakeDesk.instances.append(self)

    def next(self, hard=False):
        from prover.desk import Offer
        if self.offers:
            return Offer(self.offers.pop(0))
        return Offer({"gap": None, "note": "nothing"}, "nothing", 1)

    def beat(self, lease_id):
        self.beats += 1
        return True

    def release(self, lease_id, reason, program=None):
        self.released.append((lease_id, reason))
        return {"released": True}


WRONG = "```lean4\ntheorem PEqualNP.Outside.L_meta_9999 : ∀ (n : ℕ), n + 0 = n := by\n  intro n\n  exact wrong_lemma n\n```"
RIGHT = "```lean4\ntheorem PEqualNP.Outside.L_meta_9999 : ∀ (n : ℕ), n + 0 = n := by\n  intro n\n  exact Nat.add_zero n\n```"


def make_repo(tmp: Path) -> Path:
    repo = tmp / "fork"
    (repo / "tools").mkdir(parents=True)
    (repo / "lean").mkdir()
    (repo / "tools" / "pnp.sh").write_text("#!/bin/bash\necho submitted $2\n", encoding="utf-8")
    (repo / ".gitignore").write_text(".pnp/\n", encoding="utf-8")
    for cmd in (["init", "-q", "-b", "main"], ["add", "."], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"]):
        subprocess.run(["git", *cmd], cwd=repo, check=True, capture_output=True)
    return repo


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.repo = make_repo(self.dir)
        self.orig_desk = prover_main.Desk
        prover_main.Desk = FakeDesk
        FakeDesk.instances.clear()

    def tearDown(self):
        prover_main.Desk = self.orig_desk
        self.tmp.cleanup()

    def run_desk(self, answers, minutes=5.0, submit=False):
        args = prover_main.build_parser().parse_args(["--repo", str(self.repo), "--handle", "someone",
                                                      "--minutes", str(minutes), "--once"]
                                                     + (["--submit"] if submit else []))
        lean = FakeLean(self.repo / "lean", "exact Nat.add_zero n")
        runs = prover_main.RunLog(self.dir / "runs")
        model = ScriptedModel(answers)
        worker = prover_main.Worker(model, lean, runs, minutes, max_attempts=3, stop=threading.Event())
        code = prover_main.run_desk(args, worker, runs, threading.Event())
        return code, model, lean, FakeDesk.instances[-1]

    def test_repair_then_prove_then_deliver(self):
        code, model, lean, desk = self.run_desk([WRONG, RIGHT])
        self.assertEqual(code, 0)
        self.assertEqual(len(model.calls), 2)
        self.assertIn("wrong_lemma", model.calls[1][-1][1])          # the repair prompt shows Lean's error
        proof = self.repo / "lean/PEqualNP/Outside/L_meta_9999.lean"
        self.assertTrue(proof.exists())
        self.assertIn("exact Nat.add_zero n", proof.read_text(encoding="utf-8"))
        self.assertTrue((self.repo / ".pnp" / "L-meta-9999.json").exists())  # for tools/pnp.sh submit
        self.assertEqual(desk.released, [])                          # kept for the pull request

    def test_out_of_attempts_hands_back(self):
        code, model, lean, desk = self.run_desk([WRONG])
        self.assertEqual(desk.released, [("lease-1", "out_of_time")])
        self.assertEqual(len(model.calls), 3)

    def test_submit_runs_pnp_and_returns_to_main(self):
        original = prover_main.preflight
        prover_main.preflight = lambda repo, submit: []  # no signed-in gh in the test sandbox
        try:
            code, *_ = self.run_desk([RIGHT], submit=True)
        finally:
            prover_main.preflight = original
        self.assertEqual(code, 0)
        events = (self.dir / "runs" / "events.jsonl").read_text(encoding="utf-8")
        self.assertIn('"event": "submitted"', events)
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=self.repo,
                                capture_output=True, text=True).stdout.strip()
        self.assertEqual(branch, "main")

    def test_preflight_refuses_a_dirty_checkout(self):
        (self.repo / "stray.txt").write_text("x", encoding="utf-8")
        problems = prover_main.preflight(self.repo, submit=False)
        self.assertTrue(any("uncommitted" in p for p in problems), problems)


REPO = os.environ.get("PNP_REPO")


@unittest.skipUnless(REPO and Path(REPO, "lean").exists(), "set PNP_REPO to a built pequalnp-proofs checkout")
class RealLeanTests(unittest.TestCase):
    def test_local_problem_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            problem = Path(d) / "Swap.lean"
            problem.write_text("import Mathlib.Tactic.Ring\n\ntheorem swap_add (a b : ℕ) : a + b = b + a := by\n  sorry\n",
                               encoding="utf-8")
            wrong = "```lean4\ntheorem swap_add (a b : ℕ) : a + b = b + a := by\n  exact foo_bar a b\n```"
            right = "Plan: commutativity.\n```lean4\ntheorem swap_add (a b : ℕ) : a + b = b + a := by\n  ring\n```"
            model = ScriptedModel([wrong, right])
            lean = LeanProject(REPO)
            runs = prover_main.RunLog(Path(d) / "runs")
            worker = prover_main.Worker(model, lean, runs, minutes=20, max_attempts=3)
            out = worker.work(Job.from_file(problem))
            self.assertEqual(out.status, "proved", out)
            self.assertIn("Unknown identifier", model.calls[1][-1][1])

    def test_helpers_open_and_options_compile(self):
        with tempfile.TemporaryDirectory() as d:
            problem = Path(d) / "Twice.lean"
            problem.write_text("import Mathlib.Tactic.Ring\n\ntheorem twice (n : ℕ) : n + n = 2 * n := by\n  sorry\n",
                               encoding="utf-8")
            reply = ("```lean4\nimport Mathlib.Tactic.Ring\n\nopen Nat\n\n"
                     "lemma add_self_eq (n : ℕ) : n + n = 2 * n := by ring\n\n"
                     "set_option maxHeartbeats 400000 in\n"
                     "theorem twice (n : ℕ) : n + n = 2 * n := by\n  exact add_self_eq n\n```")
            worker = prover_main.Worker(ScriptedModel([reply]), LeanProject(REPO),
                                        prover_main.RunLog(Path(d) / "runs"), minutes=20, max_attempts=1)
            out = worker.work(Job.from_file(problem))
            self.assertEqual(out.status, "proved", out)
            self.assertIn("namespace twice", out.text)
            self.assertIn("open twice in\nset_option maxHeartbeats 400000 in\ntheorem twice", out.text)

    def test_definitions_found_for_a_desk_statement(self):
        from prover.defs import DefinitionIndex
        idx = DefinitionIndex(Path(REPO) / "lean")
        found = dict(idx.for_statement("¬ PEqualNP.MetaComplexity.HasCircuitSize 9 f 15"))
        self.assertIn("PEqualNP.MetaComplexity.HasCircuitSize", found)
        self.assertTrue(any("FanInTwo" in name for name in found), list(found))


if __name__ == "__main__":
    unittest.main()
