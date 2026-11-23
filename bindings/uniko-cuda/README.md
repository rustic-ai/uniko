# uniko-cuda

NVIDIA CUDA build of [uniko](https://github.com/rustic-ai/uniko) — the same
`import uniko` API as the CPU `uniko` package, with the ONNX Runtime CUDA
execution provider enabled for GPU-accelerated NLP/embedding inference, and
candle's CUDA kernels compiled in for local LLM inference.

Install **one** of `uniko`, `uniko-cuda`, or `uniko-metal` — they all provide
the `uniko` import and cannot coexist. `uniko-cuda` expects a CUDA-capable
NVIDIA driver on the host; the CUDA runtime libraries are resolved at load time
(not bundled), matching the standard onnxruntime-gpu model.

```sh
pip install uniko-cuda
```

See the main repository for documentation.
