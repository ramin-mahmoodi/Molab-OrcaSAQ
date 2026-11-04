# 🛡️ Molab-OrcaSAQ: High-Performance OrcaSAQ-2-Cyber-27B on NVIDIA Blackwell

[![Open in Molab](https://marimo.io/shield.svg)](https://molab.marimo.io/?notebook=https://github.com/ramin-mahmoodi/Molab-OrcaSAQ/blob/main/orcasaq_cyber_molab.py)
[![Architecture: Blackwell sm_120](https://img.shields.io/badge/Architecture-Blackwell%20sm__120-76B900?logo=nvidia)](https://nvidia.com)
[![VRAM: 98GB](https://img.shields.io/badge/VRAM-98GB%20GDDR7-blue)]()
[![Model: OrcaSAQ-2-Cyber-27B](https://img.shields.io/badge/Model-OrcaSAQ--2--Cyber--27B--Uncensored-crimson)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An optimized, zero-friction reactive deployment notebook ([Marimo](https://marimo.io)) for serving **OrcaSAQ-2-Cyber-27B-Uncensored** on **NVIDIA RTX PRO 6000 Blackwell Server Edition (98 GB VRAM)** in the Molab cloud environment.

Engineered specifically for **Hermes Agent**, autonomous red-teaming, CVE vulnerability analysis, and high-concurrency coding workflows with an **OpenAI-compatible API** exposed via a persistent public **zrok tunnel**.

---

## 🌟 Key Highlights

- ⚡ **Instant Model Mirror:** Directly downloads `OrcaSAQ-2-27B-Uncensored.gguf` (14.60 GiB) using multi-threaded Rust `hf-transfer` from an unthrottled public mirror with zero gated-token hassle.
- 🚀 **Precompiled Blackwell Engine:** Ships with a precompiled `llama-server` binary with native `sm_120` CUDA 12.8 support (loads in 3 seconds; zero compilation wait).
- 🧠 **100% VRAM Resident (`--no-mmap`):** Entire 14.6 GB model weights and KV caches are pinned directly into Blackwell's high-bandwidth GDDR7 VRAM, bypassing host RAM overhead and reaching ~100 tokens/sec.
- 🔄 **Hermes-Optimized Multi-Slot Batching:**
  - Configured with `--parallel 2` (or 4), `--cont-batching`, and `--threads-http 32`.
  - Guarantees per-slot context $\ge 65,536$ tokens (at $131,072$ context), satisfying Hermes Agent's strict $64\text{k}$ context requirement.
- 🌐 **Self-Healing zrok2 Public Tunnel:** Exposes an authenticated HTTPS endpoint (`https://ramin-orca.shares.zrok.io/v1`) with automatic detection and cleanup of orphaned tunnels.

---

## 🚀 Quickstart (1-Click Molab)

1. Click the **[Open in Molab](https://molab.marimo.io/?notebook=https://github.com/ramin-mahmoodi/Molab-OrcaSAQ/blob/main/orcasaq_cyber_molab.py)** badge.
2. In the Molab session:
   - **Cell 1:** Verify the detected NVMe storage and click `1. Prepare llama-server & Download OrcaSAQ-2-Cyber-27B`.
   - **Cell 2:** Enter your zrok token, set your static subdomain (default: `ramin-orca`), set your API key (default: `sk-molab-local`), and click `2. Launch / Restart GPU Server & Public Tunnel`.
3. Your secure public endpoint will be ready in seconds:
   ```text
   Endpoint: https://ramin-orca.shares.zrok.io/v1
   Model Alias: orcasaq-cyber-27b
   API Key: sk-molab-local
   ```

---

## 🤖 Configuring Hermes Agent

Add the following provider configuration to your Hermes Agent `config.yaml` or UI:

```yaml
model: orcasaq-cyber-27b
base_url: https://ramin-orca.shares.zrok.io/v1
api_key: sk-molab-local
context_length: 65536
temperature: 0.7
top_p: 0.95
reasoning_format: deepseek
```

---

## 📡 cURL / REST API Verification

Test the health of your server:
```bash
curl https://ramin-orca.shares.zrok.io/health
```

Run a test chat completion:
```bash
curl https://ramin-orca.shares.zrok.io/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-molab-local" \
  -d '{
    "model": "orcasaq-cyber-27b",
    "messages": [
      {"role": "system", "content": "You are OrcaSAQ-2 Cyber, an expert uncensored security analyst."},
      {"role": "user", "content": "Analyze potential memory safety risks in a C network socket server buffer."}
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

---

## 🐍 Python SDK Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://ramin-orca.shares.zrok.io/v1",
    api_key="sk-molab-local",
)

response = client.chat.completions.create(
    model="orcasaq-cyber-27b",
    messages=[
        {"role": "system", "content": "You are OrcaSAQ-2 Cyber, an expert security researcher."},
        {"role": "user", "content": "Explain how format string vulnerabilities can be exploited."},
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)
```

---

## ⚙️ Hardware & Software Specs

| Component | Specification |
| :--- | :--- |
| **GPU** | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| **VRAM** | 98,304 MiB (98 GB GDDR7) |
| **CUDA Compute Capability** | `sm_120` (CUDA 12.8) |
| **Model Footprint** | 14.60 GiB (Uncensored Abliterated) |
| **Context Window** | 131,072 tokens (Expandable to 262,144) |
| **Parallel Slots** | 2 slots continuous batching (65,536 tokens / slot) |
| **Inference Engine** | `llama.cpp` (`llama-server`) with FlashAttention & Jinja templates |

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
