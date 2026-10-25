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

Clean runs, one per arm, both reaching `VERIFY-OK`:

| arm | turns | tool calls | edits | reasoning | wall |
|---|---|---|---|---|---|
| reference | 19 | 24 | 2 (one per file) | 11,124 | 117 s |
| local model | 19 | 20 | 2 (one per file) | 19,588 | 751 s |

Turns and edit counts are identical. The local model spends **1.76x the reasoning
tokens** and **6.4x the wall time**, and the wall difference is decode speed:
its 751 s is 19 turns on one GB10, the reference's 117 s is 19 turns on a hosted
endpoint.

An earlier version of this section claimed the local model rewrote the same file
four times, breaking it once. That was an artefact: the task template still held
an output file from validating the bed, every arm started with a partial answer
in the workspace, and the stale file changed the sequence of failures. With a
clean template the pattern does not appear. The claim is withdrawn, and the
method section now requires checking the template before each arm — which is
how this was caught.

## Limit of this instrument

Three tasks, one run each per arm. It shows where turns go, not a ranking: a
model can be leaner here and still lose on longer work, and one of the three
tasks was too small to separate the arms at all. Treat it as a way to aim a
change and re-measure, not as a scoreboard.
