"""Entrypoint for the Hermes Ads Research pipeline.

Usage:
    python main.py --once                 # run a single research cycle
    python main.py --loop --interval 60    # run continuously
"""
from __future__ import annotations

import argparse
import json

from config import settings
from core.logging_config import configure_logging, get_logger
from core.orchestrator import Orchestrator, build_loop

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes Ads Research crypto prediction pipeline")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=settings.poll_interval_seconds)
    args = parser.parse_args()

    configure_logging()
    logger.info(
        "starting_pipeline",
        assets=settings.asset_list,
        kronos_mode=settings.kronos_mode,
        paper_trading=settings.paper_trading,
    )

    orchestrator = Orchestrator()

    if args.loop:
        settings.poll_interval_seconds = args.interval
        loop = build_loop(orchestrator)
        loop.run_forever()
        return

    # default: single cycle
    decisions = orchestrator.run_cycle()
    print(json.dumps([d.model_dump(mode="json") for d in decisions], indent=2, default=str))


if __name__ == "__main__":
    main()
