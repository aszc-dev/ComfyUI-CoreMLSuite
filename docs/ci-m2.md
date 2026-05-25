# Self-hosted M2 runner — setup

Tier 2 (ANE + integration + bench) runs on a self-hosted GitHub Actions
runner registered against the maintainer's M-series Mac. Hosted macOS
runners on GitHub do not expose the Apple Neural Engine, so the ANE
half of the matrix has to live on real hardware.

## One-time runner setup

1. **Install dependencies on the Mac.** Python 3.12.x (matching the
   `requires-python = ">=3.12,<3.13"` pin), `uv`, and `git`. The workflow
   clones and manages the ComfyUI checkout itself (see step 3), so you do
   not pre-install ComfyUI.

   ```bash
   brew install python@3.12 uv git
   ```

2. **Register the runner.** From the repo Settings → Actions → Runners
   → New self-hosted runner, follow the macOS-ARM instructions. Add the
   labels exactly: `self-hosted`, `macOS`, `ARM64`, `coreml` (the
   workflow `runs-on` clause requires all four).

   ```bash
   mkdir ~/actions-runner && cd ~/actions-runner
   curl -O -L https://github.com/actions/runner/releases/download/v2.317.0/actions-runner-osx-arm64-2.317.0.tar.gz
   tar xzf actions-runner-osx-arm64-2.317.0.tar.gz
   ./config.sh --url https://github.com/<owner>/<repo> \
               --token <REGISTRATION_TOKEN> \
               --labels self-hosted,macOS,ARM64,coreml \
               --name "$(hostname)-m2"
   ./svc.sh install && ./svc.sh start   # run as a launchd service
   ```

3. **Persist `COMFY_DIR` for the runner.** Point it at a **dedicated,
   runner-owned** ComfyUI directory — **not** your personal dev checkout.
   The workflow does `git reset --hard` on it and rewrites its
   `custom_nodes/ComfyUI-CoreMLSuite` symlink, so it must be disposable.
   Add it to the runner's `.env`:

   ```bash
   echo 'COMFY_DIR=/Users/<you>/actions-runner/comfyui' >> ~/actions-runner/.env
   ```

   You don't have to clone ComfyUI yourself: the `Set up ComfyUI checkout`
   step clones it on first run, checks out the right ref (latest or the
   pinned SHA — see *ComfyUI version under test*), installs ComfyUI's deps,
   and symlinks `$COMFY_DIR/custom_nodes/ComfyUI-CoreMLSuite` →
   `$GITHUB_WORKSPACE` so the server always loads the checked-out PR. If
   `COMFY_DIR` is a real node directory rather than a symlink, the step
   fails loudly instead of deleting it.

4. **Provide the SD1.5 checkpoint.** Drop
   `v1-5-pruned-emaonly.safetensors` (~4 GB) into
   `$COMFY_DIR/models/checkpoints/`. This is the one heavy artifact that
   stays as a runner-local cache; everything derived from it (the
   `.mlmodelc` UNet variants) is converted automatically by the
   `Convert UNet variants if missing` step and cached across runs. Override
   the filename with `CKPT_NAME` in the runner `.env` if needed.

   Order is free: do this before or after the first run. The setup step
   initialises the ComfyUI repo **in place** (`git init`, not `git clone`,
   which would refuse a non-empty directory) and uses `checkout -f`, which
   never removes untracked files — so a checkpoint you dropped in first is
   preserved.

## Triggers

The Tier 2 workflow (`.github/workflows/tier2.yml`) runs:

- **On PR label `run-m2`** — maintainers add the label to opt a PR
  into the ANE lane (the runner is not free; default off).
- **Nightly at 04:00 UTC** via `schedule:`.
- **Manually** via the workflow_dispatch button.

## ComfyUI version under test

Tier 2 runs a **hybrid** strategy, keyed on the trigger, so the suite tracks a
moving host without making PRs flaky to overnight upstream drift:

| Trigger | ComfyUI ref | ComfyUI deps |
|---|---|---|
| **schedule** (nightly) | latest `origin/master` | its own `requirements.txt`, capped by `constraints/comfy-ceiling.txt` |
| **PR label `run-m2`** / **dispatch** | pinned `requires-comfyui` SHA | the frozen `comfy` uv group |

- **Nightly = canary.** It pulls the latest ComfyUI and installs *ComfyUI's
  own* dependency set. The frozen `comfy` uv group cannot track a moving host
  by hand (a latest checkout needs e.g. `comfyui-frontend-package==1.44.19`,
  `comfy_aimdo`, `alembic`, `blake3` that an older pin never listed), so latest
  mode defers to upstream's `requirements.txt`. Upstream API or dependency
  breakage surfaces here, in CI, instead of in a user's install.
- **PR / dispatch = reproducible gate.** It checks out the pinned
  `requires-comfyui` SHA and uses the frozen `comfy` group — the known-good
  combination — so a PR fails for its own reasons, not because ComfyUI moved.

The resolved ComfyUI SHA and mode are written to the job step summary (and
`COMFY_SHA` in the env), so any failure names the exact commit it hit.

**The toolchain ceiling is deliberate.** `constraints/comfy-ceiling.txt` caps
`torch<2.8` / `numpy<2` / `coremltools 9` while installing latest ComfyUI's
requirements. If upstream ever hard-requires something past those bounds the
nightly install *fails on purpose* — that is the signal that coremltools /
`ml-stable-diffusion` need a deliberate Phase-5-style bump, not a silent float
that would break the ANE path (see `docs/deps.md`).

`pyproject.toml`'s `requires-comfyui` is both the PR-gate ref and the
published-compatibility declaration for the Comfy registry. Bump it once a
newer ComfyUI is validated (the nightly canary is what tells you it's safe).

## Artifacts

- Bench JSON/MD are uploaded as `bench-results`.
- M2 golden image regressions surface as test failures in
  `tests/m2/test_golden_image.py`; diff PNG is written next to the
  golden under `tests/m2/_latest_generated.png` (gitignored).

## When the runner is down

If the maintainer's Mac is offline, the workflow queues until the
runner comes back. Cancel a stuck run from the Actions UI; the gate is
not blocking by default (Tier 0 + Tier 1 carry PR status). Tier 2 is
"good to merge once it goes green," not "blocked until then."

## Replacing the integration e2e

The legacy `tests/integration/test_basic_conversion_1_5.py` checked
CoreML output against an MPS reference image at PSNR > 25 dB. That
reference path is broken on macOS 26.x with torch 2.0.1 (see Phase 1
Gate report). Phase 4 moves the same coverage to
`tests/m2/test_golden_image.py`, which:

- runs the Core ML pipeline only (no MPS reference),
- asserts SHA256 against `tests/m2/goldens/sd15_seed42.sha256`,
- falls back to PSNR ≥ 40 dB if the hash drifts.

This removes the human-eyeball dependency: a regression is now a
numerical fail, not a "looks different to me."
