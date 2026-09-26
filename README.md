# LTX-2.5 Video Generation: RunPod Serverless Worker

Text-to-video with [LTX-2.5](https://github.com/Lightricks/LTX-2)'s **distilled** two-stage
pipeline. It renders at half size and then upscales 2× with a fixed, fast schedule. It's the video
backend for `ig-story-pipeline`.

## Deploy

1. **Hugging Face access.** LTX-2.5 is gated. Accept its terms at
   https://huggingface.co/Lightricks/LTX-2.5, then create a **Read** token.
2. **Network volume.** About 100 GB, in a datacenter that has your GPU type. The models (~66 GB) are
   downloaded there on the first job and reused on every later cold start.
3. **Endpoint** (Serverless → New Endpoint → GitHub repo → this repo):
   - GPU: 48 GB PRO (L40S / RTX 6000 Ada) or 80 GB. Below 60 GB the transformer runs in fp8.
   - Allowed CUDA versions: **13.x only**, because the image uses torch 2.13 built for CUDA 13.2.
   - Min workers 0, max workers 2. Execution timeout: **60 min** (the first job includes the download).
   - Network volume: attach the volume from step 2.
   - Environment variables: `HF_TOKEN=<your read token>`.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `HF_TOKEN` | none | Hugging Face read token (required for the gated model) |
| `MODEL_DIR` | `/runpod-volume/models/ltx-2.5` | where the model files live |
| `LTX_QUANTIZATION` | `fp8-cast` below 60 GB VRAM, else `none` | transformer weight format |
| `LTX_OFFLOAD` | `cpu` below 60 GB VRAM, else `none` | stream weights from CPU RAM (slower, fits 48 GB cards) |

## API

Input (`test_input.json`):

```json
{"input": {"prompt": "...", "width": 704, "height": 1280, "num_frames": 49, "fps": 24, "seed": 42}}
```

- `width` and `height` must be multiples of 64.
- `num_frames` is snapped to 8n+1.
- `negative_prompt`, `guidance_scale` and `num_inference_steps` are accepted but ignored, because the distilled schedule is fixed.

Output:

```json
{"video": "data:video/mp4;base64,...", "duration_seconds": 2.04, "resolution": "704x1280",
 "fps": 24, "num_frames": 49, "seed": 42, "generation_time_seconds": 40.1, "load_time_seconds": 0}
```

The MP4 includes LTX-2's generated audio track. `ig-story-pipeline` drops it and uses its own voice and music.
When a job fails, the output is `{"error": "...", "traceback": "..."}`.

## Pinning

The Dockerfile installs LTX-2 from a fixed commit (`LTX2_COMMIT`), because `handler.py` depends on
its APIs. When you bump the commit, re-check the handler against `ltx_pipelines/distilled.py`
(`main()`).
