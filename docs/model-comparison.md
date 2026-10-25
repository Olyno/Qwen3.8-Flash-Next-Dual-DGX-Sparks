# Comparing the local model against a reference

A repeatable way to find where this checkpoint loses time, as opposed to guessing. The idea is not "is it smart" — every task below was solved by both models — but *which turns are wasted and why*, so an improvement can be aimed and then re-measured.

## Method

- **Same harness.** Both arms run through the same client, with the same prompt, the same auto-approval, and the same wall-clock cap. One arm is the checkpoint this repository serves; the other is a stronger hosted model, used only as a reference point.
- **Same copy.** Each arm gets a byte-identical copy of the task directory, a neutral `AGENTS.md`, and no prior transcript.
- **No artefacts from validation.** Verifying a task bed means running it, which leaves build outputs, `__pycache__` and `.pytest_cache` in the *template*. Copied into an arm those are free hints — a stale output file is a partial answer. Clean the template after every validation run, and check the copy before starting an arm.
- **Mechanical acceptance.** `pytest`, exit codes, or a required stdout line. No model judges another model.
- **Fresh tasks only.** A task whose tooling was built in an earlier session is not an instrument: on the Kimi-import bed, both models "solved" it by running the session importer, so the run measured the tool, not the model.
- **One variable.** The third task below has two failures in sequence, so the trace shows *how* each model narrows the cause rather than whether it can type.

## Results

| task | reference | local model |
|---|---|---|
| 4 seeded bugs, 9 tests | 11 turns, 31 s, 9/9 | 7 turns, 50 s, 9/9 |
| 19-test implementation from a partial file | 32 turns, 176 s, 28.8k completion | 20 turns, 631 s, 20.8k completion, 19/19 |
| diagnosis: wrong field, then a 1.0 s budget against 11.4 s of work | 15 turns, 77 s, 4,950 reasoning | 19 turns, 411 s, 9,574 reasoning |

Both models reach the acceptance criteria in all three. The local model is **leaner on coding tasks** (7 vs 11 turns; 20 vs 32) and heavier only on diagnosis. Its wall-clock is not a thinking cost: 31.6 s per turn against 5.5 s is decode speed on one GB10 versus a cloud endpoint, and its token counts are lower on both coding tasks.

## What the diagnosis trace shows

Both models found both defects. They differ in **when they verify a replacement**:

- Reference: times the heavy pass (11.4 s) against the budget (1.0 s) at turn 6, replaces it with a closed form at turn 8, and proves the closed form reproduces the original output at turn 10 — one replacement, one proof.
- Local model: reads the failure, rewrites at turn 8, rewrites again at 9, breaks its own file at 10, spends 11–12 restoring the function header it clobbered, rewrites a third time at 13, verifies at 14, rewrites a fourth time at 16, and only then passes.

Four attempts instead of one, plus a self-inflicted regression. Nothing is missing in knowledge or correctness; the missing discipline is **measure the failing operation before editing it, and prove equivalence in the same turn you replace an implementation**.

## What this buys

- A measurement that survives model changes: any future checkpoint can be dropped into the local arm and scored on the same three tasks, with turns-to-diagnosis as the headline number.
- A named target for an improvement, instead of "make it better": the four-rewrite pattern is visible in the trace and countable from a transcript, so a fix can be shown to reduce it.
- A warning about where improvement cannot come from: the local model is already leaner in turns and tokens than the reference on coding work. A change that only shortens answers or prunes tools cannot help; the target is verification placement on diagnosis, and the 95-turn session's habit of probing before acting.

## Limit of this instrument

Three tasks, one run each per arm. It shows where turns go, not a ranking: a
model can be leaner here and still lose on longer work, and one of the three
tasks was too small to separate the arms at all. Treat it as a way to aim a
change and re-measure, not as a scoreboard.
