# How this fork runs

**Branch policy:** `main` is the only branch. No `experiments/*` — they rot and
fork the truth.

**Experiments** run as **git worktrees on the bench box** (msi):

```bash
cd ~/fork && git worktree add -q ../wt-<topic>   # pinned checkout of main
# boot arms with experiments/v30/launch_v30.sh (or a ride_<X>.sh driver),
# rows land under ~/v30_bench/, verdicts under docs/verdicts/ in the worktree
```

Each experiment owns one topic branch *only inside its worktree*; nothing is
pushed from it. When a measured win lands (speed, quality, context, reactivity),
it is folded into `main` as a structured commit — launcher knobs into
`start.sh`/`recipes/`, tooling into `tools/` or `experiments/`, the number into
`docs/verdicts/` + the README table — and pushed. Negative results get a
verdict doc too (see DFLASH/DDTREE: dead ends stay dead by being written down).

**Benchmark discipline** (protocol is fixed, do not "improve" it mid-lane):
- one heavy job per box; the gate owns the GPU until it closes
- decodebench 600 tok @ temp 0.6, contexts 1k+100k, prose/code/entropy/copy;
  every speed claim is priced against its A1-floor row (17.1-18.8 tok/s)
- spec-decoding claims need per-position acceptance from `/metrics` AND an
  output-equality check — never validate at batch-1 only
- after ANY container restart, discard the first run (JIT warm-up), and pin
  the image by digest before believing a number (`/v1/models` + image id)
- report accepted-per-step and steps/s separately; their product is tok/s and
  mixing them hides which lever moved

**Commits:** default author, no co-author trailers, no per-doc micro-commits,
no mega-commits. Each step: one logical change, message says what + why +
measurement where applicable.
