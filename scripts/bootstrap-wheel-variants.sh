#!/usr/bin/env bash
# Populate each GPU wheel-variant binding directory with the canonical Python
# package from bindings/uniko-py.
#
# The variant dirs (bindings/uniko-cuda, bindings/uniko-metal) track only
# Cargo.toml, pyproject.toml, README.md, and a one-line build.rs in git; their
# `python/` package is copied in here (and gitignored) so maturin's
# `python-source = "python"` finds the same `uniko` package the CPU wheel
# ships. Copy, not symlink — maturin does not follow symlinks for python-source.
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCE="bindings/uniko-py/python"
if [[ ! -d "$SOURCE" ]]; then
    echo "ERROR: $SOURCE does not exist. Run from the repo root." >&2
    exit 1
fi

VARIANTS=(cuda metal)
for variant in "${VARIANTS[@]}"; do
    dir="bindings/uniko-${variant}"
    if [[ ! -d "$dir" ]]; then
        echo "WARN: $dir does not exist, skipping" >&2
        continue
    fi
    target="$dir/python"
    rm -rf "$target"
    cp -r "$SOURCE" "$target"
    find "$target" -name __pycache__ -type d -prune -exec rm -rf {} +
    # Drop any locally-compiled extension copied from a `maturin develop` tree;
    # maturin drops the freshly-built GPU extension in its place. (CI checkouts
    # never have one — the .so is gitignored.)
    find "$target" \( -name '*.so' -o -name '*.pyd' -o -name '*.dylib' \) -delete
    echo "Bootstrapped: $target"
done
echo "Done. ${#VARIANTS[@]} variant directories populated."
