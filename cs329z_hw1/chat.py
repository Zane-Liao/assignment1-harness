"""Talk to your agent in the terminal (course-provided).

    uv run python -m cs329z_hw1.chat [--mode confirm|auto] [--memory-dir DIR]

This builds one session with ``tests.adapters.run_agent_session(LM(), None,
config)``, so it runs your full Cardinal toolset on the model named by
CS329Z_MODEL. You are the user: in confirm mode you are asked to approve
tool calls, and ``ask_user`` questions are put to you. Type ``exit`` or
press Ctrl-D to leave.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `tests.adapters` imports from any directory

from cs329z_hw1.llm import LM
from cs329z_hw1.types import AgentConfig
from cs329z_hw1.user import ConsoleUser


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with your Cardinal Agent.")
    parser.add_argument("--mode", choices=["confirm", "auto"], default="confirm")
    parser.add_argument("--memory-dir", type=Path, default=Path(".lm_cache/chat_memory"))
    args = parser.parse_args()

    from tests import adapters  # your wiring

    config = AgentConfig(
        mode=args.mode,
        user=ConsoleUser(),
        memory_dir=args.memory_dir,
        aux_lm=LM(tag="chat/aux"),
    )
    session = adapters.run_agent_session(LM(tag="chat"), None, config)
    print(f"Cardinal Agent ({args.mode} mode, memory in {args.memory_dir}). Type exit to leave.")
    while True:
        try:
            message = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.lower() in ("exit", "quit"):
            break
        if not message:
            continue
        result = session.send(message)
        note = "" if result.status == "done" else f" [stopped: {result.status}]"
        print(f"\nagent{note}> {result.text}")


if __name__ == "__main__":
    main()
