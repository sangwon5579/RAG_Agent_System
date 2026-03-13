from __future__ import annotations
import argparse
import sys
from pathlib


def main() -> None:
    from .settings import load_settings
    from .agent import Agent

    settings = load_settings()
    if not settings.openai_api_key:
        print("Error: OPENAI_API_KEY is not set in the environment variables.")
        sys.exit(1)

    agent = Agent(settings)
    agent.run()


if __name__ == "__main__":
    main()