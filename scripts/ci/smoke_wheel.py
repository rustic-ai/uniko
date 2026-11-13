#!/usr/bin/env python3
"""Import an installed uniko wheel and exercise one write and one read.

Run by the release workflow against each built wheel, from a fresh
environment, before anything is published. A wheel can build and link yet
fail to import: Python extensions link with `-undefined dynamic_lookup` on
macOS, so a missing symbol only surfaces at `import uniko`. That is how the
0.4.0 macOS wheels nearly shipped broken.

    python -I scripts/ci/smoke_wheel.py            # import, write, recall
    python -I scripts/ci/smoke_wheel.py --open-only  # import and open a store

`--open-only` is for the CUDA wheel, which runs on a runner with the CUDA
runtime libraries but no GPU.
"""

from __future__ import annotations

import sys


def main() -> int:
    import uniko

    print(f"uniko {uniko.__version__} from {uniko.__file__}")
    memory = uniko.Uniko.in_memory_sync()
    if "--open-only" not in sys.argv:
        agent = memory.agent("smoke")
        agent.session("smoke").observe_sync(
            uniko.Turn("user", "The release smoke test hikes in the Alps")
        )
        bundle = agent.recall_sync("hiking")
        if not bundle.items:
            print("recall returned nothing for a just-written turn", file=sys.stderr)
            return 1
        print(f"recall ok: {len(bundle.items)} item(s)")
        # shutdown needs every Agent / Session handle dropped first.
        del agent, bundle
    memory.shutdown_sync()
    print("smoke ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
