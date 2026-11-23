# uniko-metal

Apple Silicon (Metal / CoreML) build of
[uniko](https://github.com/rustic-ai/uniko) — the same `import uniko` API as the
CPU `uniko` package, with the ONNX Runtime CoreML execution provider enabled for
GPU/ANE-accelerated NLP/embedding inference on macOS arm64, and candle's Metal
kernels compiled in for local LLM inference.

Install **one** of `uniko`, `uniko-cuda`, or `uniko-metal` — they all provide
the `uniko` import and cannot coexist. `uniko-metal` is macOS/Apple-Silicon
only; the CoreML provider is statically linked (nothing extra to install).

```sh
pip install uniko-metal
```

See the main repository for documentation.
