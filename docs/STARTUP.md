# Start the local inspector

From the repository root in PowerShell, use this recommended path:

```powershell
.\start-inspector.ps1
```

The launcher uses the existing project `.venv`, CPU, port 8765, the M10 cleaned
contours, and the original M8 capture gate. It opens the browser only after the
server has bound successfully. Keep the terminal open; Ctrl+C stops this
instance. Nothing is uploaded, no weights are downloaded, no existing server
is stopped, and no model or threshold is replaced. M12 and M13 remain research
comparisons, not silently selected upgrades.

Check readiness without starting a server or opening a browser:

```powershell
.\start-inspector.ps1 -Check
```

If 8765 is already occupied, choose another port explicitly:

```powershell
.\start-inspector.ps1 -Port 8770
```

An occupied port is not proof that the correct inspector is running. The
launcher will not kill a process, silently move ports or reuse an unknown app.
Both readiness and actual startup check the port; a later binding race fails
with a message rather than claiming the website is ready.

## Recovering from a startup error

| Message | What to do |
|---|---|
| Project Python environment missing / missing module | Run `uv sync --locked --extra cpu --group dev` in the repository. This is an explicit dependency installation, not performed by startup. |
| Experiment storage not configured | Copy `.env.example` to `.env` only if `.env` does not already exist. Set `ANOMALY_ARTIFACT_ROOT` to the absolute folder containing your completed `runs/`. |
| Experiment storage missing | Check that the configured drive/folder is mounted. Do not create empty run folders to bypass verification. |
| ResNet weights missing | Set `TORCH_HOME` to the existing cache containing `hub/checkpoints/resnet18-f37072fd.pth`, or restore the verified original checkpoint. |
| Weights failed SHA-256 | Restore the pinned checkpoint; do not rename arbitrary weights or disable integrity checks. |
| Model references missing / incompatible | Restore the matching completed M4, M5, M8 and M10 runs. A plain Git checkout does not contain private feature banks or calibration artifacts. |
| CUDA initialization failed | Use the default CPU environment, or separately configure the locked CUDA profile documented in `ENVIRONMENT.md`. |
| Port occupied / unavailable | Close your own earlier instance or choose `-Port 8770`; no other program will be stopped. |

When a restored artifact set is unavailable, reproduce the existing pipeline
in order: [VisA preparation](../README.md#dataset), M3/M4 memories, M5
calibration, M8 gate, then M10 localization. These are substantial explicit
research runs, not quick startup repair. The scripts and frozen configurations
are linked in the README. Do not run training merely to fix a wrong local path.

For an explicit missing-checkpoint acquisition, run:

```powershell
.venv\Scripts\python.exe scripts/prepare_m11_encoders.py --encoder resnet18
```

This separately authorized command downloads/verifies only pinned ResNet
weights. ConvNeXt is not required; omitting `--encoder` would prepare both.
Startup itself never calls the acquisition command.

## Alternative shells and diagnostics

If PowerShell blocks local script execution, do not change machine-wide policy.
Run the same entry point directly:

```powershell
.venv\Scripts\python.exe scripts/run_inspector.py --open-browser
.venv\Scripts\python.exe scripts/run_inspector.py --check --port 8770
```

The PowerShell wrapper resolves its paths relative to its own folder, so it can
also be invoked by absolute path from another working directory. `--debug` on
the Python entry point exposes a traceback for development. The older
`run_m6_prototype.py`, `run_m8_prototype.py`, `run_m10_prototype.py` and
`run_m12_prototype.py` remain available for historical reproduction/comparison;
they are not the recommended first launch path.

## Manual startup checks

1. Run `-Check`: expect verified runtime information and “no server was started”.
2. Start the inspector and visit its printed URL. Select a supported category;
   inspect an existing permitted local example. Recapture must have no anomaly
   score/heatmap; a passed capture exposes the frozen policy and evidence.
3. In another terminal, repeat `-Check` on that port. Expect an occupied-port
   message and exit code 2. The running page should remain available.
4. Stop only your own instance with Ctrl+C. The port should then be reusable.
5. The automated tests cover missing configuration/storage/weights, corrupt
   checkpoints, missing model artifacts and bind failures without modifying
   your real files. Do not delete your weights just to test an error message.

Health is available at `/api/health`; source M5, gate and localization identities
are reported there. Read [the system card](SYSTEM_CARD.md) for privacy/safety
limits. Successful startup verifies software/artifacts, not detection accuracy.
