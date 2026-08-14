"""
VRAM accounting shared by the LoRA train/predict scripts.

The shape written here is the one evaluate_htr.py reads back:

    remarks["gpu"] = {
        "devices":       {...},   # static per-device name / total VRAM
        "model":         {...},   # params, footprint, VRAM right after load
        "peak_per_pass": {<pass_label>: {"devices": {...},
                                         "peak_reserved_total_GiB": float}},
    }

evaluate_htr.peak_vram_gib() looks up peak_per_pass[<pass_label>], so the label
must be the prediction key the same run writes into "predictions" (lora_zs,
qlora_zs, ...) — otherwise the vram_gib column stays empty.

Every train/predict script measures VRAM through these helpers and nowhere else,
so the vram_gib column of one eval table is the same quantity for every row.
"""

import torch


def gpu_devices_info() -> dict:
    """Static per-device info (name, total VRAM). Recorded once."""
    return {
        f"cuda:{i}": {
            "device_name": torch.cuda.get_device_name(i),
            "total_GiB": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
        }
        for i in range(torch.cuda.device_count())
    }


def reset_peak_vram() -> None:
    """
    Drop the allocator's cached segments, then reset the peak counters on every
    visible device.

    empty_cache() is not optional here. reset_peak_memory_stats() resets the
    peaks to the *current* values rather than to zero, and the caching allocator
    holds on to freed segments until it is told to let go. Resetting alone
    therefore seeds the next pass's `peak_reserved` with the previous pass's
    high-water pool, so a measured-in-isolation inference pass can only ever
    report at least what training reserved — which is how every fine-tuned run
    ended up with the same reserved total for both passes.

    reset_peak_memory_stats() does not lazy-init CUDA, so calling it before the
    caching allocator exists raises "Invalid device argument". Nothing has been
    allocated at that point, so the peaks are already zero — just skip.
    """
    if not torch.cuda.is_initialized():
        return
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def vram_snapshot() -> dict:
    """Current + peak-since-last-reset VRAM per device, in GiB."""
    return {
        f"cuda:{i}": {
            "allocated_GiB": round(torch.cuda.memory_allocated(i) / (1024**3), 2),
            "reserved_GiB": round(torch.cuda.memory_reserved(i) / (1024**3), 2),
            "peak_allocated_GiB": round(torch.cuda.max_memory_allocated(i) / (1024**3), 2),
            "peak_reserved_GiB": round(torch.cuda.max_memory_reserved(i) / (1024**3), 2),
        }
        for i in range(torch.cuda.device_count())
    }


def total_peak_reserved_gib(snapshot: dict) -> float:
    """Sum of peak reserved VRAM across devices — the single headline number."""
    return round(sum(d["peak_reserved_GiB"] for d in snapshot.values()), 2)


def pass_entry(snapshot: dict) -> dict:
    """One peak_per_pass entry: per-device detail plus the summed headline."""
    return {
        "devices": snapshot,
        "peak_reserved_total_GiB": total_peak_reserved_gib(snapshot),
    }


def model_vram_info(model) -> dict:
    """
    Parameter count, weight footprint and VRAM right after loading.

    get_memory_footprint() counts every parameter wherever it lives, so a
    footprint larger than what is allocated on the GPUs means device_map
    offloaded the remainder to CPU/disk — the silent cause of ~10x slower
    generation. `offloaded_GiB` makes that visible in the predictions JSON
    instead of only in a warning that scrolls past during loading.
    """
    n_params = sum(p.numel() for p in model.parameters())
    footprint_gib = model.get_memory_footprint() / (1024**3)
    after_load = vram_snapshot()
    on_gpu_gib = sum(d["allocated_GiB"] for d in after_load.values())
    return {
        "n_params": n_params,
        "n_params_B": round(n_params / 1e9, 2),
        "footprint_GiB": round(footprint_gib, 2),
        "offloaded_GiB": round(max(footprint_gib - on_gpu_gib, 0.0), 2),
        "after_load": after_load,
        "load_peak_reserved_GiB": total_peak_reserved_gib(after_load),
    }


def new_gpu_info(model) -> dict:
    """
    Build the remarks["gpu"] skeleton right after loading *model*.

    `load_peak_reserved_GiB` is mirrored at the top level on purpose:
    evaluate_htr.peak_vram_gib() falls back to it when the pass label is missing
    from peak_per_pass, so a label mismatch still reports the load peak instead
    of leaving the vram_gib column blank.

    Call reset_peak_vram() before loading the model, then fill
    ["peak_per_pass"][<pass_label>] with pass_entry(vram_snapshot()) after each
    measured pass.
    """
    info = model_vram_info(model)
    return {
        "devices": gpu_devices_info(),
        "model": info,
        "load_peak_reserved_GiB": info["load_peak_reserved_GiB"],
        "peak_per_pass": {},
    }


def print_model_vram(info: dict) -> None:
    """One-line load summary, plus a warning when weights were offloaded."""
    print(f"Model loaded: {info['n_params_B']:.2f}B params, "
          f"footprint {info['footprint_GiB']:.2f} GiB, "
          f"peak reserved {info['load_peak_reserved_GiB']:.2f} GiB")
    if info["offloaded_GiB"] > 0.5:
        print(f"  WARNING: {info['offloaded_GiB']:.2f} GiB of weights are NOT on the GPU "
              f"(device_map offloaded them to CPU/disk). Generation will be very slow — "
              f"make more GPUs visible or use --max-memory / 4-bit.")
