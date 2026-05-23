# Self-hosted M2 runner — setup

Tier 2 (ANE + integration + bench) runs on a self-hosted GitHub Actions
runner registered against the maintainer's M-series Mac. Hosted macOS
runners on GitHub do not expose the Apple Neural Engine, so the ANE
half of the matrix has to live on real hardware.

## One-time runner setup

1. **Install dependencies on the Mac.** Python 3.11.x (matching the
   `requires-python` pin), `uv`, `git`, plus the ComfyUI checkout at the
   path the workflow expects (default: `$HOME/dev/ComfyUI`). The
   workflow reads `COMFY_DIR` from the runner's env.

   ```bash
   brew install python@3.11 uv git
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

3. **Persist `COMFY_DIR` for the runner.** The workflow needs to know
   where the ComfyUI checkout lives. Add it to the runner's `.env`:

   ```bash
   echo 'COMFY_DIR=/Users/<you>/dev/ComfyUI' >> ~/actions-runner/.env
   ```

4. **Pre-convert the baseline SD1.5 model.** The bench step expects
   `$COMFY_DIR/models/unet/v1-5-pruned-emaonly_1x512x512_se_unet.mlmodelc`.
   Run the conversion once manually:

   ```bash
   cd $GITHUB_WORKSPACE
   uv run python bench/scripts/convert_sd15.py
   ```

   Re-runs of the same combination are a no-op; the converter skips when
   the .mlmodelc already exists.

## Triggers

The Tier 2 workflow (`.github/workflows/tier2.yml`) runs:

- **On PR label `run-m2`** — maintainers add the label to opt a PR
  into the ANE lane (the runner is not free; default off).
- **Nightly at 04:00 UTC** via `schedule:`.
- **Manually** via the workflow_dispatch button.

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
