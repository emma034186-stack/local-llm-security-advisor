# Local LLM Security Advisor

Gradio chat interface powered by a local GGUF security LLM (Foundation-Sec-8B). Supports PDF upload for document-based Q&A and streams responses with live token-speed display.

## Architecture

```
User (browser)
    │
    ▼
Gradio UI (port 8071)
    │
    ▼
llama-cpp-python  ──▶  local GGUF model (Foundation-Sec-8B-Instruct Q8_0)
    │
    ▼  (optional)
PyPDF2  ──▶  PDF text extraction → injected into prompt context
```

## Features

- Streaming inference with live tok/s display
- PDF upload — extracted text is injected as context
- Stop-generation button
- GPU-accelerated via llama-cpp-python (CUDA)
- Optimized for RTX 4060 Laptop (configurable via code)

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU acceleration, install the CUDA build of llama-cpp-python:

```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python==0.2.90 --force-reinstall
```

### 2. Download the model

Download `foundation-sec-8b-instruct-q8_0.gguf` (or any compatible GGUF) and place it in the project directory, or set the `MODEL_PATH` environment variable:

```bash
export MODEL_PATH=/path/to/your-model.gguf
```

### 3. Run

```bash
python chat_local_gguf_with_pdf.py
```

Open http://localhost:8071

## Model

[Foundation-Sec-8B](https://huggingface.co/fdtn-ai/Foundation-Sec-8B) — security-focused LLM by Cisco Talos, fine-tuned for vulnerability analysis and remediation.

## GPU Configuration

Default settings target RTX 4060 Laptop (8 GB VRAM). Key parameters in `load_local_gguf_model()`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `n_gpu_layers` | -1 | Full GPU offload |
| `n_batch` | 1024 | Parallel batch size |
| `low_vram` | True | Required for laptop GPU |
| `n_ctx` | 2048 | Context window |

Adjust `n_gpu_layers` or `low_vram` for different hardware.
