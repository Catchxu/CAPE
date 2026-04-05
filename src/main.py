import argparse

from .pipeline import run_pipeline
from .utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified CTA pipeline entrypoint.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to an experiment YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
