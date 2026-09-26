"""
RunPod Serverless handler for LTX-2.5 text-to-video (distilled two-stage pipeline).

Input:
{
    "input": {
        "prompt": "Gentle waves rolling onto a quiet beach at sunset ...",   # required
        "width": 704,          # optional, default 704  (multiple of 64)
        "height": 1280,        # optional, default 1280 (multiple of 64)
        "num_frames": 121,     # optional, default 121; snapped to 8n+1
        "fps": 24,             # optional, default 24
        "seed": 42             # optional, random if not set
    }
}
negative_prompt, guidance_scale and num_inference_steps are accepted and ignored: the
distilled pipeline has a fixed schedule and no classifier-free guidance.

Output:
{
    "video": "data:video/mp4;base64,...",
    "duration_seconds": 5.04, "resolution": "704x1280", "fps": 24, "num_frames": 121,
    "seed": 42, "generation_time_seconds": 81.2, "load_time_seconds": 0.0
}
On failure: {"error": "...", "traceback": "..."}.

Models are downloaded once from Hugging Face into the network volume (MODEL_DIR) and loaded
once per worker. The pipeline is built with the library's own CLI argument parser so model
paths, quantization and offload settings resolve exactly as `python -m ltx_pipelines.distilled`.

Environment:
    HF_TOKEN          Hugging Face read token (LTX-2.5 is gated; accept its terms first)
    MODEL_DIR         where models live, default /runpod-volume/models/ltx-2.5
    LTX_QUANTIZATION  fp8-cast | none | ... ; default: fp8-cast below 60 GB of VRAM, else none
    LTX_OFFLOAD       none | cpu | disk ; default none (use cpu if a job runs out of memory)
"""

import base64
import os
import random
import sys
import tempfile
import time
import traceback

import runpod
import torch

HF_REPO = "Lightricks/LTX-2.5"
MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models/ltx-2.5")
FILES = {
    "transformer": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoder": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video_vae": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "audio_vae": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "upsampler": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
}

PIPELINE = None
ARGS = None


def ensure_models() -> dict:
    """Download any missing model files into MODEL_DIR (the network volume)."""
    paths = {k: os.path.join(MODEL_DIR, f) for k, f in FILES.items()}
    missing = [FILES[k] for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        from huggingface_hub import snapshot_download

        print(f"Downloading {len(missing)} model file(s) (~66 GB on first run) to {MODEL_DIR} ...")
        start = time.time()
        snapshot_download(
            HF_REPO,
            allow_patterns=missing,
            local_dir=MODEL_DIR,
            token=os.environ.get("HF_TOKEN") or None,
        )
        print(f"Download finished in {time.time() - start:.0f}s")
    return paths


def quantization() -> str | None:
    q = os.environ.get("LTX_QUANTIZATION")
    if q is None:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        q = "fp8-cast" if vram_gb < 60 else "none"
        print(f"GPU {torch.cuda.get_device_name(0)} ({vram_gb:.0f} GB) -> quantization {q}")
    return None if q.lower() == "none" else q


def load_pipeline():
    """Build the distilled pipeline once, exactly as the library's CLI does."""
    global PIPELINE, ARGS
    if PIPELINE is not None:
        return 0.0

    start = time.time()
    paths = ensure_models()

    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.utils.args import (
        add_generated_keyframes_arg,
        default_2_stage_distilled_arg_parser,
        resolve_cli_params,
    )

    argv = [
        "--transformer-path", paths["transformer"],
        "--text-encoder-path", paths["text_encoder"],
        "--video-vae-path", paths["video_vae"],
        "--audio-vae-path", paths["audio_vae"],
        "--spatial-upsampler-path", paths["upsampler"],
        "--offload", os.environ.get("LTX_OFFLOAD", "none"),
        # required by the CLI parser; each job supplies its own
        "--prompt", "placeholder",
        "--output-path", "/tmp/placeholder.mp4",
    ]
    q = quantization()
    if q:
        argv += ["--quantization", q]

    # resolve_cli_params() reads the checkpoint path from sys.argv to pick model defaults.
    saved_argv = sys.argv
    sys.argv = ["ltx_pipelines.distilled", *argv]
    try:
        params = resolve_cli_params(distilled=True)
        parser = add_generated_keyframes_arg(
            default_2_stage_distilled_arg_parser(params=params, supports_auto_duration=True)
        )
        ARGS = parser.parse_args(argv)
    finally:
        sys.argv = saved_argv

    PIPELINE = DistilledPipeline(
        model_paths=ARGS.model_paths,
        spatial_upsampler_path=ARGS.spatial_upsampler_path,
        loras=tuple(ARGS.lora) if ARGS.lora else (),
        quantization=ARGS.quantization,
        compilation_config=ARGS.compile,
        offload_mode=ARGS.offload_mode,
        prompt_enhancer_gemma_root=ARGS.prompt_enhancer_gemma_root,
        diffvae_optimization=ARGS.diffvae_optimization,
    )
    took = time.time() - start
    print(f"Pipeline ready in {took:.0f}s")
    return took


def generate(prompt: str, width: int, height: int, num_frames: int, fps: int, seed: int) -> tuple[str, int]:
    """Run one generation; return the encoded MP4 path and the frame count actually used."""
    from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
    from ltx_pipelines.utils.helpers import snap_frames_to_grid
    from ltx_pipelines.utils.media_io import encode_video, resolve_hdr_color_space, vae_dtype_for_hdr

    num_frames = snap_frames_to_grid(max(9, num_frames))
    hdr = resolve_hdr_color_space(images=[], hdr=None)
    result = PIPELINE(
        prompt=prompt,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=float(fps),
        images=[],
        vae_dtype=vae_dtype_for_hdr(hdr, torch.bfloat16),
        color_space=hdr,
        enhance_prompt=False,
        enhance_static_cache=False,
        tiling_config=AUTO_TILING,
        generated_keyframes=ARGS.num_generated_keyframes,
    )
    out = os.path.join(tempfile.mkdtemp(), "output.mp4")
    encode_video(
        video=result.video,
        fps=fps,
        audio=result.audio,
        output_path=out,
        video_chunks_number=get_video_chunks_number(result.num_frames, result.tiling_config),
        color_space=hdr,
    )
    return out, result.num_frames


def handler(event: dict) -> dict:
    try:
        inp = event.get("input") or {}
        prompt = inp.get("prompt")
        if not prompt:
            return {"error": "Missing required field: prompt"}

        width = int(inp.get("width", 704))
        height = int(inp.get("height", 1280))
        if width % 64 or height % 64:
            return {"error": f"width and height must be multiples of 64, got {width}x{height}"}
        fps = int(inp.get("fps", 24))
        seed = inp.get("seed")
        seed = random.randint(0, 2**31 - 1) if seed is None else int(seed)

        load_time = load_pipeline()
        start = time.time()
        path, frames = generate(prompt, width, height, int(inp.get("num_frames", 121)), fps, seed)
        gen_time = time.time() - start

        with open(path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")
        return {
            "video": f"data:video/mp4;base64,{video_b64}",
            "duration_seconds": round(frames / fps, 2),
            "resolution": f"{width}x{height}",
            "fps": fps,
            "num_frames": frames,
            "seed": seed,
            "generation_time_seconds": round(gen_time, 1),
            "load_time_seconds": round(load_time, 1),
        }
    except torch.cuda.OutOfMemoryError as e:
        return {
            "error": f"CUDA out of memory: {e}. Set LTX_OFFLOAD=cpu on the endpoint, or use a larger GPU.",
            "traceback": traceback.format_exc(),
        }
    except Exception as e:  # report instead of crashing the worker
        return {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
