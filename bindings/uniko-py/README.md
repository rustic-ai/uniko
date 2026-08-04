# uniko (Python bindings)

Async-first Python SDK for the [uniko](https://github.com/rustic-ai/uniko)
cognitive memory engine, built with [PyO3](https://pyo3.rs) and
[maturin](https://www.maturin.rs).

> **Status:** alpha. The full async surface (recall, answer, query, ingest,
> goals/tasks, and the Locy logic surface), the synchronous `*_sync` skins, and
> a complete `py.typed` type stub all ship today. Prebuilt wheels are not yet
> published — build from source with `maturin` (below).

## Building locally

```bash
# from bindings/uniko-py/ (uv-managed)
uv run maturin develop
uv run python -c "import uniko; print(uniko.__file__)"
```

A C/C++ toolchain and `protobuf-compiler` (`protoc`) must be on `PATH` — the
uniko stack statically links the ONNX runtime (via uni-db's `provider-onnx`)
and several native dependencies. On Linux, `mold` is also required: the
workspace's `.cargo/config.toml` forces it as the link backend.

## Example

```python
import asyncio
import uniko


async def main() -> None:
    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("assistant")
    session = agent.session("conversation-1")
    await session.observe(uniko.Turn("user", "I prefer tea over coffee."))
    bundle = await agent.recall("beverage preference")
    for item in bundle.items:
        print(item.content)


asyncio.run(main())
```

Every verb also has a blocking `*_sync` twin for callers outside an event loop
(scripts, notebooks, sync handlers). It blocks on the shared runtime and
releases the GIL across the Rust work:

```python
import uniko

uni = uniko.Uniko.in_memory_sync()
agent = uni.agent("assistant")
session = agent.session("conversation-1")
session.observe_sync(uniko.Turn("user", "I prefer tea over coffee."))
for item in agent.recall_sync("beverage preference").items:
    print(item.content)
```
