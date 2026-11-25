# offline_llm
Rust inference engine with Python UI AI offline prompt.

- `chat_script_r.py`: the interactive chat (`make` builds and runs it).
- `worker.py`: a continuous Lean prover that keeps working on a problem unattended (below).

## Continuous worker

Like SETI@home, but for proofs: the worker runs your local model in a loop and works on one
theorem at a time, either a step from the [pequalnp.com](https://pequalnp.com) desk (the open
P vs NP attack) or a Lean problem file of your own.

For each step it:

1. Takes a step from the desk, checks that the bare statement compiles, and keeps the lease
   alive every 9 minutes while it works.
2. Builds a prompt with the statement, the project definitions the statement uses (looked up in
   the Lean sources, so the model knows what `HasCircuitSize` or `FanInTwo` mean), and the
   fleet's notes on the step.
3. Generates a proof, keeps the statement exactly as given, and checks the file against the
   site's format rules, then with Lean (errors, `sorry`, and that only the standard axioms are used).
4. When Lean rejects it, it shows the model the file and Lean's errors and asks for a fix (two
   repair rounds, then a fresh attempt), until the time for the step runs out.
5. Then, honestly:
   - A proof that checks is written where the pull request expects it. With `--submit`, the
     worker opens the PR itself through the project's `tools/pnp.sh submit` and goes on.
   - Otherwise it hands the step back (`out_of_time`, or `does_not_elaborate` if the statement
     itself does not compile) and asks for the next one.

It follows the desk's rules in [join.txt](https://pequalnp.com/join.txt):

- One step at a time, and at least 20 minutes on each (default 30).
- Lease renewal, and the desk's `retry_after_s` is always honoured.
- At most one request every 5 seconds.
- Search shards are handed back, since they need a search program rather than a proof.

### Setup (once)

1. **Rebuild the Rust extension.** `generate` gained options (context size, token limit,
   sampling, early stop), the model's own chat template, and a fix for streamed UTF-8. Before,
   Lean symbols such as `∀` and `→` could be cut in half between tokens and come out as `�`.
   ```bash
   make build          # first build recompiles llama.cpp (a few minutes)
   ```
   The chat script works unchanged: `generate(prompt, callback, temperature)` and
   `count_tokens(text)` keep the signatures it calls. `generate` still defaults to an 8192-token
   context and a 1024-token reply, and returns the stop reason (`complete`, `length_limit`,
   `context_full`, or `stopped` when the callback returns `False`). A prompt that does not fit returns `context_full`, so
   auto-retry can trim and try again. Token counts come from `last_usage()`.

   If your working copy has changes that never reached GitHub (server sync, auto-retry), commit
   them first and merge this branch. Only `src/lib.rs`, `Cargo.toml`, `Makefile`, `README.md` and
   `.gitignore` overlap. For `src/lib.rs` and `Cargo.toml` take this branch's version; it keeps
   the return values your chat script relies on. Your `chat_script_r.py` and
   `conversation_sync.py` are not touched.

2. **Get a Lean prover model.** General models (Qwen-Coder, DeepSeek-R1-Distill) rarely write
   Lean that compiles. Models trained for Lean do much better. All three below are Qwen3 or
   DeepSeek architectures, which the pinned llama.cpp loads:
   - [Pythagoras-Prover-4B](https://huggingface.co/Pythagoras-LM/Pythagoras-Prover-4B)
     (Apache-2.0, Qwen3-4B). It is the strongest small open prover on miniF2F and small enough to
     leave room for Lean and a long context on 16 GB. Use a GGUF build at Q8_0 or Q6_K (the
     model page links community quantizations, or convert it with llama.cpp's
     `convert_hf_to_gguf.py`), with `--ctx 16384 --max-tokens 8192`.
   - [Goedel-Prover-V2-8B](https://huggingface.co/Goedel-LM/Goedel-Prover-V2-8B) (Apache-2.0,
     Qwen3-8B). Close behind, at twice the size. Q4_K_M, `--ctx 16384 --max-tokens 8192`.
   - [DeepSeek-Prover-V2-7B](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-7B), as a GGUF
     build such as `Mungert/DeepSeek-Prover-V2-7B-GGUF` (Q4_K_M). Weaker than the two above, and
     its attention uses about three times the memory per token, so keep `--ctx 8192`.

   These were trained on an older Lean and Mathlib than the pequalnp project uses, so expect
   renamed-lemma errors; the repair loop feeds those back. Put the `.gguf` file in
   `~/ai_build/models/`.

3. **Lean and the pequalnp project.**
   ```bash
   curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
   gh auth login                                   # only needed for --submit
   gh repo fork open-goat/pequalnp-proofs --clone  # your fork, in ~/pequalnp-proofs
   cd ~/pequalnp-proofs && tools/pnp.sh setup      # fetches prebuilt Mathlib (several GB)
   cd lean && lake build PEqualNP.Defs             # once; later checks reuse it
   ```

### Run

```bash
caffeinate -is ./ai_env/bin/python worker.py \
  --model ~/ai_build/models/DeepSeek-Prover-V2-7B-Q4_K_M.gguf \
  --repo ~/pequalnp-proofs
```

`caffeinate` keeps the Mac awake while it runs. Ctrl-C hands the current step back and stops.

- Add `--submit` to open the pull request automatically and keep going. Without it, the worker
  stops when it has a checked proof and prints the one command to submit it.
- `--minutes 45` gives each step more time; `--hard` asks for the open problems the attack is
  stuck on.
- `--problem my_theorem.lean` works on your own file instead of the desk: one theorem ending in
  `:= by sorry`, after any imports and definitions. It keeps going round after round until it
  finds a proof and writes `my_theorem.proved.lean`.
- `make worker MODEL=... PNP_REPO=...` does the same as the command above.

Everything is logged in `prover_runs/`:

- `events.jsonl` has one line per step and attempt.
- `steps/<id>/` holds each prompt, reply, Lean file and Lean output.

### What to expect

Most open steps on the desk today are hard: research-level lemmas, or computational lower
bounds whose accepted proofs run to thousands of lines. A 7B model will hand most steps back
after its time runs out, which is normal and useful to the fleet (an honest "could not"). It
can land the occasional short step, and on your own problems of exercise size it is a fair
companion.

Each attempt takes a few minutes on an M2 Pro: generation at roughly 20 to 30 tokens per second,
plus 10 seconds to a few minutes for Lean. The model (4 to 6 GB) and Lean (2 to 4 GB while
checking) share memory, so close heavy apps on a 16 GB machine.

### Tests

```bash
python -m unittest discover -s tests -v
PNP_REPO=~/pequalnp-proofs python -m unittest discover -s tests   # adds two tests that run real Lean
```
