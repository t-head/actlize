#!/usr/bin/env python3
"""Collect runtime environment information and write it as JSON."""

import argparse
import json
import os
import subprocess
import sys


def command_output(command):
    """Run a command and return its output, or a marker when unavailable."""
    try:
        result = subprocess.run(
            command, text=True, capture_output=True, check=False
        )
    except FileNotFoundError:
        return "not found"
    return (result.stdout or result.stderr).strip()


def collect_env_info(docker_image):
    """Collect build container, SDK, and compiler information."""
    return {
        "docker_image": docker_image,
        "hostname": command_output(["hostname"]),
        "python_version": sys.version,
        "ppu_sdk": os.environ.get("PPU_SDK", ""),
        "ppu_home": os.environ.get("PPU_HOME", ""),
        "hgcc_version": command_output(["hgcc", "--version"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect runtime environment information as JSON."
    )
    parser.add_argument(
        "--docker-image",
        required=True,
        help="Docker image used by the current CI job",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the generated JSON file",
    )
    args = parser.parse_args()
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(
            collect_env_info(args.docker_image),
            output_file,
            indent=2,
            sort_keys=True,
        )
        output_file.write("\n")


if __name__ == "__main__":
    main()
