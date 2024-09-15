#!/usr/bin/env python3
import os
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PyTorch microbenchmarks and compare to golden data"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file (must contain 'fp_16_enabled' and 'models')"
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Path to write logs (overrides default in Microbenchmarks)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the pass/fail threshold multiplier"
    )
    # you can add more arguments here as needed by Microbenchmarks
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    args = parse_args()
    config = load_config(args.config)

    # If you want to override Microbenchmarks defaults, set them on args:
    if args.log_path:
        setattr(args, "log_path", args.log_path)
    if args.threshold is not None:
        setattr(args, "threshold", args.threshold)

    # Instantiate and run
    tester = PyTorchTests(args, config)
    tester.run()
    tester.comparator()


if __name__ == "__main__":
    main()