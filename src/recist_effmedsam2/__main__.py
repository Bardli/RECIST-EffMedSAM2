"""Allow running ``python -m recist_effmedsam2`` directly."""

from .runner.run_benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
