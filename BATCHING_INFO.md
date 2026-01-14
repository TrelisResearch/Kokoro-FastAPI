# Kokoro Batch Inference via Transformers

## PR: huggingface/transformers#35790

**URL:** https://github.com/huggingface/transformers/pull/35790
**Author:** eustlb
**Branch:** `eustlb:add-style-tts-2`
**Status:** Open (awaiting merge)

## What It Does

Adds StyleTTS 2 to Transformers library with support for Kokoro-82M checkpoints.

## Batch Inference Benefits

- ~50% speed improvement at batch size 128
- ~8% faster than original codebase at BS=1
- Mask support for padded inputs

## Installation (Pre-Merge)

```bash
pip install git+https://github.com/eustlb/transformers.git@add-style-tts-2
```

## Alternative: NimbleEdge/kokoro

https://github.com/NimbleEdge/kokoro

Fork with batch inference + ONNX export. Uses `torch.bmm` instead of `torch.matmul` and mask-based alignment matrix computation.
