"""Continuous Lean prover worker built on the offline_llm Rust backend.

The worker takes a step from the pequalnp.com desk (or a local problem file), asks the local
model for a Lean 4 proof, checks every attempt with Lean, feeds the errors back, and either
delivers a checked proof or hands the step back with an honest reason. See worker.py.
"""
