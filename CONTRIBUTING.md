# Contributing

This repository prioritizes reproducible research over rapid feature growth.

Before changing code, read the project specification and data-governance files.
Keep reusable logic in `src/`, use thin scripts, and add focused tests for new
behaviour. Dataset files, private inspection images, weights, and local run
outputs must not be committed.

Run these checks before proposing a change:

```powershell
uv run --extra cpu ruff format --check .
uv run --extra cpu ruff check .
uv run --extra cpu mypy src scripts
uv run --extra cpu pytest
```

Changes to a data split, evaluation metric, threshold policy, or robustness
severity require an explicit rationale and a versioned configuration update.
Never describe model output as proof of product quality or defect root cause.
