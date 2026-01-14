# Kokoro-FastAPI Scaling Plan

## Chosen Approach: Multiple Workers + Queuing (Not Batching)

**Target deployment**: Modal with H100

**Rationale**:
- Transformers batching PR only gives ~50% speedup at BS=128, ~8% at BS=1
- Kokoro model is small (~500MB) - H100 has 80GB VRAM
- Can fit 8+ model copies easily on single H100
- Simpler than implementing batch inference

| Keep from Kokoro-FastAPI | Add/Modify |
|--------------------------|------------|
| OpenAI-compatible API | Request queue with backpressure |
| Streaming + non-streaming | Uvicorn `--workers 8` |
| Format conversion (mp3/opus/wav) | `--limit-concurrency` per worker |
| Text normalization | Modal deployment config |
| Voice combining | |
| Smart text splitting | |

## Current Behavior Notes

- **Streaming**: Supported via `stream=True` (default) or `stream=False`
- **Chunk ordering**: Chunks from different requests CAN interleave (not per-request sequential)

---

## Summary of Findings

### A) Is there batching? Dynamic or continuous?

**Current state: Text-chunking exists, but NO request-level batching.**

| Type | Present? | Details |
|------|----------|---------|
| Text chunking | ✅ Yes | Text split into 175-250 token chunks (max 450), content-aware |
| Request batching | ❌ No | Each request processed independently |
| GPU batch inference | ❌ No | Model processes one chunk at a time sequentially |

**The chunking is dynamic/content-aware:**
- Respects sentence boundaries and punctuation
- Adjusts based on pause tags `[pᴀsᴇ=duration]`
- Token counts guide chunk size decisions
- Config: `target_min_tokens=175`, `target_max_tokens=250`, `absolute_max_tokens=450`

**Concurrency control:**
- `asyncio.Semaphore(4)` limits concurrent chunk processing
- This throttles concurrency but does NOT batch inputs together

---

### B) What happens when multiple requests hit the endpoint simultaneously?

**Flow for 2 concurrent requests:**

```
Request A ─┬─→ TTSService (singleton) ─┬─→ Semaphore (4 slots) ─┬─→ KokoroV1 Model
Request B ─┘                           └─────────────────────────┘   (SEQUENTIAL)
```

1. **FastAPI level**: Both handled by async event loop concurrently
2. **Service level**: Both share the same `TTSService` singleton (protected by init lock)
3. **Model level**: Both share the same `KokoroV1` model instance
4. **Chunk processing**: Both compete for 4 semaphore slots
5. **GPU inference**: **Fundamentally sequential** - the Kokoro model processes one text/chunk at a time
6. **GPU memory**: Shared, monitored, cleared when threshold (80%) exceeded
7. **No queuing**: Requests proceed as resources allow, no explicit queue

**Key bottleneck**: The underlying model inference is sequential. Even with the semaphore allowing 4 "concurrent" chunks, the actual GPU inference serializes.

**Key files:**
- `api/src/services/tts_service.py:31` - Semaphore definition
- `api/src/routers/openai_compatible.py:49-72` - Singleton with lock
- `api/src/inference/kokoro_v1.py` - Model inference (sequential)

---

### C) What would it take to support better batching?

**Option 1: Request-Level Queue + Batching (Moderate effort)**
- Add a queue (asyncio.Queue or Redis) to buffer incoming requests
- Batch N requests together before calling model
- Requires waiting for batch to fill OR timeout
- Trade-off: latency vs throughput

**Option 2: True GPU Batch Inference (Significant effort)**
- Modify `KokoroV1.generate()` to accept batch of texts
- The underlying `KPipeline` would need to support batched tensors
- Most TTS models (including Kokoro) may not support this out-of-box
- Would require changes to the Kokoro library itself

**Option 3: Dynamic Batching with SemaphoreQueue (Moderate effort)**
- Collect requests that arrive within a time window (e.g., 50ms)
- Process them together if the model supports it
- Similar to NVIDIA Triton's dynamic batching

**Option 4: Multiple Worker Processes (Low effort, limited benefit)**
- Run multiple Uvicorn workers with `--workers N`
- Each worker has its own model copy (high VRAM usage)
- Doesn't improve per-request latency, just throughput

**Recommended investigation:**
1. Check if `KPipeline.generate_from_tokens()` can accept batched inputs
2. Measure current latency/throughput under concurrent load
3. Profile where time is spent (tokenization, model inference, audio encoding)

---

---

## Detailed: Semaphore Behavior with 5 Concurrent Requests

The semaphore (`asyncio.Semaphore(4)`) at `tts_service.py:31` is **class-level** and shared across all requests.

**5 requests, each with 2 chunks = 10 chunks total:**

```
Time 0ms:   All 5 requests start, begin processing first chunks
Time 2ms:   Chunks A1, B1, C1, D1 acquire semaphore (4/4 used)
            Chunk E1 BLOCKS waiting for semaphore
Time 3-100ms: GPU processes A1→B1→C1→D1 SEQUENTIALLY
              (semaphore doesn't enable parallel GPU inference)
Time ~100ms: A1 finishes, E1 acquires slot, A2 queued
...continues until all done
```

**What semaphore does:**
- ✅ Limits concurrent async tasks in flight (memory protection)
- ❌ Does NOT enable parallel GPU inference
- ❌ Does NOT batch inputs together
- ❌ Does NOT make individual requests faster
- ❌ Does NOT limit incoming requests (queue grows unbounded)

**Chunk interleaving**: Chunks from different requests CAN interleave (A1→B1→A2→B2 not A1→A2→B1→B2)

**Overwhelm risk**: No limits at Uvicorn or FastAPI level. With 100 simultaneous requests:
- All 100 start processing concurrently
- 4 chunks in semaphore, rest queue in memory (unbounded)
- GPU still processes sequentially
- Memory grows until OOM or client timeouts

---

## Critical Files

| File | Purpose | For Batching |
|------|---------|--------------|
| `api/src/inference/kokoro_v1.py` | Current inference backend | **Replace** with transformers |
| `api/src/services/tts_service.py` | TTSService, semaphore | **Modify** for batch queue |
| `api/src/routers/openai_compatible.py` | API endpoints | Minor changes |
| `api/src/services/text_processing/text_processor.py` | smart_split() | Keep as-is |
| `api/src/services/streaming_audio_writer.py` | Format conversion | Keep as-is |

---

## Implementation Plan: Multi-Worker + Queuing

### Step 1: Add Uvicorn Worker Configuration

Modify entrypoint to support multiple workers:
```bash
# docker/scripts/entrypoint.sh
uvicorn api.src.main:app --host 0.0.0.0 --port 8880 \
  --workers 8 \
  --limit-concurrency 10
```

- `--workers 8`: 8 model copies, 8 parallel inference streams
- `--limit-concurrency 10`: Max 10 requests per worker before rejection/queuing

### Step 2: Add Request Queue (Optional)

If we want queuing instead of rejection when capacity exceeded:
- Add Redis or `asyncio.Queue` based request buffer
- Return 503 with retry-after when queue is full
- File: `api/src/middleware/queue.py` (new)

### Step 3: Modal Deployment Config

Create `modal_app.py`:
```python
import modal

app = modal.App("kokoro-tts")

@app.function(
    gpu="H100",
    image=modal.Image.from_dockerfile("Dockerfile.gpu"),
    concurrency_limit=80,  # 8 workers × 10 concurrent each
)
def serve():
    # Uvicorn with 8 workers
```

### Step 4: Adjust Semaphore per Worker

Currently semaphore is class-level (shared across requests in same worker).
With 8 workers, each worker has its own semaphore(4).

Consider reducing to `Semaphore(1)` since each worker should focus on 1 request at a time:
- File: `api/src/services/tts_service.py:31`

### Step 5: Test Under Load

```bash
# Simulate 50 concurrent requests
hey -n 100 -c 50 -m POST -H "Content-Type: application/json" \
  -d '{"input":"Hello world","voice":"af_heart"}' \
  http://localhost:8880/v1/audio/speech
```

---

## Critical Files to Modify

| File | Change |
|------|--------|
| `docker/scripts/entrypoint.sh` | Add `--workers 8 --limit-concurrency 10` |
| `api/src/services/tts_service.py:31` | Consider `Semaphore(1)` per worker |
| `modal_app.py` (new) | Modal deployment configuration |
| `api/src/middleware/queue.py` (new, optional) | Request queue with backpressure |

---

## Resources

- Transformers PR (alternative, not using): https://github.com/huggingface/transformers/pull/35790
- NimbleEdge fork (alternative): https://github.com/NimbleEdge/kokoro
- Modal docs: https://modal.com/docs
