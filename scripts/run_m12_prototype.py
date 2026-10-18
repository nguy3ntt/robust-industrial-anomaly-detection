"""Explicit v2 gate comparison; historical M10 configuration is unchanged."""

from pathlib import Path

from run_m8_prototype import main

if __name__ == "__main__":
    raise SystemExit(main(default_config=Path("configs/prototype/m12.yaml")))
