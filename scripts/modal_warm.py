"""Keep one container of the deployed SmearLens app warm, or let it scale to zero again.

    uv run python scripts/modal_warm.py on      # one L40S always up, about 2 $/h even when idle
    uv run python scripts/modal_warm.py off     # back to zero; the warm container stops after 15 min idle
    uv run python scripts/modal_warm.py status

The next `modal deploy` resets these overrides to the values in scripts/modal_app.py.
"""

import argparse
import warnings

import modal

APP = "smearlens-app"
CLASS = "Analyzer"


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn the warm SmearLens container on or off.")
    parser.add_argument("action", choices=("on", "off", "status"))
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message=".*update_autoscaler.*")
    analyzer = modal.Cls.from_name(APP, CLASS)()
    if args.action == "status":
        settings = analyzer.update_autoscaler()
    else:
        settings = analyzer.update_autoscaler(min_containers=1 if args.action == "on" else 0)
    print(f"{APP}/{CLASS}: min_containers={settings.min_containers}, scaledown_window={settings.scaledown_window}s")


if __name__ == "__main__":
    main()
