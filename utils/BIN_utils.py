# BIN_utils.py
# Utilities for packing/unpacking spike tensors, writing/reading .bin files,
# and round-tripping between audio <-> spikes <-> bin <-> audio.
# Supports saving/loading JSON sidecars in a different directory.
# Adds spectrogram generation from BIN files.
# Now with progress prints for long-running/batch operations.
# And: writing trims trailing zero (padded) frames to reduce .bin size.

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Spectrogram

from utils.encode import spike_encode, reconstruct_from_spikes
from utils.audio_utils import reconstruct_without_stretch

# -------------------------------
# Small utilities for progress
# -------------------------------

def _percent(i: int, total: int) -> int:
    if total <= 0:
        return 100
    return int(i * 100 / total)

def _fmt_s(seconds: float) -> str:
    return f"{seconds:.2f}s"

try:
    from time import perf_counter as _now
except Exception:  # very unlikely fallback
    import time
    _now = time.time

# Modes that are safe to 1-bit pack (0/1 spikes).
BINARY_MODES = {"phased_rate", "basic", "rate"}


# -------------------------------
# Low-level packing / unpacking
# -------------------------------

def _assert_binary_spikes(spikes: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """
    Ensure spikes are binary (0/1) before 1-bit packing. If slightly off 0/1, round.
    """
    if not torch.all((spikes >= -tol) & (spikes <= 1 + tol)):
        raise ValueError("Spikes are not within [0, 1] range; cannot 1-bit pack.")
    # If values are not exactly 0/1, round with a warning.
    if not torch.all((spikes < 0.5 + tol) | (spikes > 0.5 - tol)):
        print("⚠️  Spikes not exactly 0/1; rounding to nearest bit for packing.")
    return spikes.clamp(0.0, 1.0).round()


def pack_spikes_to_uint32(spikes: torch.Tensor, F: int) -> np.ndarray:
    """
    Pack a [T, F] binary spike tensor into uint32 words (row-major; 32 bits per word).
    """
    spikes = _assert_binary_spikes(spikes)
    T = int(spikes.shape[0])
    words_per_timestep = (F + 31) // 32

    spikes_np = spikes.detach().cpu().numpy().astype(np.uint8)

    packed = np.empty(T * words_per_timestep, dtype=np.uint32)
    idx = 0
    for t in range(T):
        row = spikes_np[t]
        if F % 32 != 0:
            pad = words_per_timestep * 32 - F
            if pad > 0:
                row = np.pad(row, (0, pad), constant_values=0)
        for i in range(words_per_timestep):
            word = 0
            base = i * 32
            for b in range(32):
                word |= (int(row[base + b]) & 1) << b
            packed[idx] = word
            idx += 1

    return packed


def unpack_uint32_to_spikes(words: np.ndarray, F: int) -> torch.Tensor:
    """
    Unpack uint32 words into a [T, F] binary spike tensor (float32, 0/1).
    """
    words = np.asarray(words, dtype=np.uint32)
    words_per_timestep = (F + 31) // 32
    if len(words) % words_per_timestep != 0:
        raise ValueError(
            f"Invalid .bin length: {len(words)} not divisible by words_per_timestep={words_per_timestep}"
        )
    T = len(words) // words_per_timestep

    spikes_np = np.zeros((T, F), dtype=np.uint8)
    idx = 0
    for t in range(T):
        for i in range(words_per_timestep):
            word = words[idx]
            idx += 1
            base = i * 32
            for b in range(min(32, F - base)):
                spikes_np[t, base + b] = (word >> b) & 1

    return torch.from_numpy(spikes_np.astype(np.float32))


# -------------------------------
# Metadata I/O helpers
# -------------------------------

def _json_path_for(bin_path: Path, json_out_dir: Optional[Path]) -> Path:
    """
    Derive JSON path for a given BIN path. If json_out_dir is provided,
    use <json_out_dir>/<stem>.json; otherwise place JSON next to BIN.
    """
    if json_out_dir is not None:
        return json_out_dir / (bin_path.stem + ".json")
    return bin_path.with_suffix(".json")


def write_meta_json(json_path: Path | str, meta: Dict) -> None:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(meta, f)


def read_meta_json(json_path: Path | str) -> Dict:
    with Path(json_path).open("r") as f:
        return json.load(f)


# -------------------------------
# Spikes + meta -> BIN (and back)
# -------------------------------

def _infer_t_real_from_spikes(spikes_bin01: torch.Tensor) -> int:
    """
    Fallback: infer T_real by trimming trailing all-zero rows.
    Expects 0/1 spikes (use _assert_binary_spikes before calling).
    Returns the number of non-padded (kept) frames.
    """
    if spikes_bin01.numel() == 0:
        return 0
    # any True where a frame has any spike
    nz = spikes_bin01.any(dim=1)
    if not torch.any(nz):
        return 0
    last = int(torch.nonzero(nz, as_tuple=False)[-1])
    return last + 1


def write_spikes_bin_with_meta(
    spikes: torch.Tensor,
    bin_path: Path | str,
    meta: Dict,
    *,
    mode: str,
    json_out_dir: Optional[Path | str] = None,
) -> None:
    """
    Write a spike tensor and metadata to .bin and .json (sidecar).

    Trims trailing padded (all-zero) frames so the .bin only contains the
    real timesteps. If meta['T_real'] is present, that is used; otherwise
    trailing zeros are inferred from the spike tensor itself.

    JSON can be written to a different directory via json_out_dir.
    """
    if mode not in BINARY_MODES:
        raise ValueError(f"Mode '{mode}' is not 1-bit encodable. Use one of {sorted(BINARY_MODES)}.")

    bin_path = Path(bin_path)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    json_out_dir = Path(json_out_dir) if json_out_dir is not None else None

    # Ensure binary for correct zero-trim inference.
    spikes_bin = _assert_binary_spikes(spikes)
    T_in = int(spikes_bin.shape[0])

    # Decide how many frames to save.
    T_real_meta = meta.get("T_real", None)
    if T_real_meta is not None:
        try:
            T_real_meta = int(T_real_meta)
        except Exception:
            T_real_meta = None

    if T_real_meta is not None and 0 <= T_real_meta <= T_in:
        T_save = T_real_meta
    else:
        T_save = _infer_t_real_from_spikes(spikes_bin)

    # Guard against pathological values
    T_save = max(0, min(T_save, T_in))
    spikes_to_write = spikes_bin[:T_save]

    # Compute F (columns per frame)
    if "F" in meta:
        F = int(meta["F"])
    else:
        F = int(meta.get("n_fft", 0)) // 2 + 1 if "n_fft" in meta else int(spikes_to_write.shape[1])

    # Write packed data
    packed = pack_spikes_to_uint32(spikes_to_write, F=F)
    packed.tofile(bin_path)

    # Prepare and write metadata
    meta = dict(meta)  # copy
    if T_save != T_in:
        meta["T_padded"] = T_in  # original (pre-trim) length for reference
    meta["T"] = int(T_save)
    meta.setdefault("F", int(spikes_to_write.shape[1]))
    write_meta_json(_json_path_for(bin_path, json_out_dir), meta)


def read_spikes_bin_with_meta(
    bin_path: Path | str,
    *,
    F: Optional[int] = None,
    json_dir: Optional[Path | str] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Read spikes and metadata from a .bin and sidecar .json (optionally in a different directory).
    """
    bin_path = Path(bin_path)
    json_dir = Path(json_dir) if json_dir is not None else None
    meta_path = _json_path_for(bin_path, json_dir)
    meta = read_meta_json(meta_path)

    if F is None:
        F = int(meta.get("F", (int(meta["n_fft"]) // 2 + 1)))

    words = np.fromfile(bin_path, dtype=np.uint32)
    spikes = unpack_uint32_to_spikes(words, F=F)
    return spikes, meta


# ---------------------------------------------
# Audio -> spikes -> BIN (+ JSON) convenience
# ---------------------------------------------

def encode_audio_to_bin(
    audio_path: Path | str,
    bin_out: Path | str,
    *,
    n_fft: int = 512,
    hop_length: int = 32,
    max_len: int = 4000,
    threshold: float = 0.003,
    mode: str = "phased_rate",
    normalize: bool = True,
    padding: bool = True,
    sample_rate_expected: Optional[int] = None,
    json_out_dir: Optional[Path | str] = None,
) -> Dict:
    """
    Load a mono .wav, create STFT, run spike_encode, and write .bin + .json.
    JSON can be saved to a different directory via json_out_dir.

    NOTE: Trailing padded frames are NOT written to .bin; meta["T"] reflects the saved length.
    """
    t0 = _now()
    audio_path = Path(audio_path)
    print(f"Encoding '{audio_path.name}' …", flush=True)
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] != 1:
        raise ValueError(f"Only mono audio supported: {audio_path}")
    if sample_rate_expected and sr != sample_rate_expected:
        print(f"Input sample rate {sr} != expected {sample_rate_expected}; continuing with {sr}.")
    original_length = int(waveform.shape[-1])

    stft = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1.0)
    spec = stft(waveform).squeeze(0)  # [F, T]

    spikes, logstft, log_min, log_max, mask = spike_encode(
        stft_tensor=spec,
        max_len=max_len,
        threshold=threshold,
        normalize=normalize,
        mode=mode,
        padding=padding
    )

    T_in, F = spikes.shape
    T_real = int(mask.sum().item()) if torch.is_tensor(mask) else None
    T_save = int(T_real) if (T_real is not None) else T_in

    meta = {
        "T": int(T_save),               # saved frames (post-trim)
        "F": int(F),
        "T_real": T_real,               # real frames prior to any padding
        "T_padded": int(T_in) if T_save != T_in else None,  # original padded length (if any)
        "log_min": float(log_min) if log_min is not None else None,
        "log_max": float(log_max) if log_max is not None else None,
        "original_length": original_length,
        "max_len": int(max_len),
        "n_fft": int(n_fft),
        "hop_length": int(hop_length),
        "sample_rate": int(sr),
        "mode": str(mode),
        "threshold": float(threshold),
        "normalize": bool(normalize),
        "padding": bool(padding),
    }

    # Write (writer will also trim if needed, and will keep meta["T"] as saved length)
    write_spikes_bin_with_meta(spikes, bin_out, meta, mode=mode, json_out_dir=json_out_dir)
    dt = _fmt_s(_now() - t0)
    pad_info = f", trimmed {T_in - T_save} padded frame(s)" if T_save != T_in else ""
    print(f"Encoded '{audio_path.name}' → '{Path(bin_out).name}' in {dt} "
          f"(T={T_save}, F={F}, sr={sr}{pad_info})", flush=True)
    return meta


# ---------------------------------------------
# BIN (+ JSON) -> reconstructed audio convenience
# ---------------------------------------------

def reconstruct_bin_to_audio(
    bin_path: Path | str,
    *,
    json_dir: Optional[Path | str] = None,
    out_wav: Optional[Path | str] = None,
    n_iter: int = 32,
) -> torch.Tensor:
    """
    Read spikes + meta from bin/json, reconstruct log-magnitude STFT via
    reconstruct_from_spikes, invert with Griffin-Lim, optionally save .wav.
    """
    t0 = _now()
    bin_path = Path(bin_path)
    print(f"→ Reconstructing audio from '{bin_path.name}' …", flush=True)
    spikes, meta = read_spikes_bin_with_meta(bin_path, json_dir=json_dir)
    mode = meta.get("mode", "phased_rate")

    mask = None
    if "T_real" in meta and meta["T_real"] is not None:
        T = int(meta["T"])
        T_real = int(meta["T_real"])
        # If T_real >= T (no padding in file), mask will be all ones below.
        mask = torch.zeros(T, dtype=torch.float32)
        mask[:min(T_real, T)] = 1.0

    recon_log = reconstruct_from_spikes(spikes, mode=mode, mask=mask, trim=mask is not None)
    logstft_FxT = recon_log.cpu().T  # [F, T]
    waveform = reconstruct_without_stretch(
        logstft_tensor=logstft_FxT,
        log_min=meta.get("log_min"),
        log_max=meta.get("log_max"),
        filename=str(out_wav) if out_wav else None,
        n_fft=int(meta["n_fft"]),
        hop_length=int(meta["hop_length"]),
        sample_rate=int(meta.get("sample_rate", 16000)),
        n_iter=n_iter,
        original_length=int(meta.get("original_length") or 0) or None,
    )
    dt = _fmt_s(_now() - t0)
    if out_wav:
        print(f"Reconstructed '{Path(out_wav).name}' from '{bin_path.name}' in {dt}", flush=True)
    else:
        print(f"Reconstructed tensor from '{bin_path.name}' in {dt}", flush=True)
    return waveform


# ---------------------------------------------
# Convenience: round-trip directory helpers
# ---------------------------------------------

def encode_dir_to_bins(
    input_dir: Path | str,
    bin_out_dir: Path | str,
    *,
    json_out_dir: Optional[Path | str] = None,
    n_fft: int = 512,
    hop_length: int = 16,
    max_len: int = 1500,
    threshold: float = 0.003,
    mode: str = "phased_rate",
    normalize: bool = True,
    padding: bool = True,
    sample_rate_expected: Optional[int] = None,
    audio_ext: str = ".wav",
) -> None:
    """
    Encode all audio files in input_dir to .bin in bin_out_dir and .json in json_out_dir (if provided).
    If json_out_dir is None, JSON sidecars are written next to each BIN.
    """
    t_all = _now()
    input_dir = Path(input_dir)
    bin_out_dir = Path(bin_out_dir)
    json_out_dir = Path(json_out_dir) if json_out_dir is not None else None

    bin_out_dir.mkdir(parents=True, exist_ok=True)
    if json_out_dir is not None:
        json_out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(input_dir.glob(f"*{audio_ext}"))
    total = len(wav_files)
    if not wav_files:
        print(f"No {audio_ext} files found in {input_dir}")
        return

    print(f"🎧 Encoding {total} file(s) from '{input_dir}' → BINs in '{bin_out_dir}'"
          + (f" (JSON in '{json_out_dir}')" if json_out_dir else " (JSON next to BIN)"),
          flush=True)

    ok = 0
    fail = 0
    for i, wav_path in enumerate(wav_files, start=1):
        t0 = _now()
        try:
            bin_out = bin_out_dir / (wav_path.stem + ".bin")
            print(f"[{i:>4}/{total}] ({_percent(i-1, total):>3}%) {wav_path.name} → {bin_out.name}", flush=True)
            encode_audio_to_bin(
                wav_path,
                bin_out,
                n_fft=n_fft,
                hop_length=hop_length,
                max_len=max_len,
                threshold=threshold,
                mode=mode,
                normalize=normalize,
                padding=padding,
                sample_rate_expected=sample_rate_expected,
                json_out_dir=json_out_dir,
            )
            ok += 1
            print(f"    Done in {_fmt_s(_now()-t0)} [{_percent(i, total):>3}%]\n", flush=True)
        except Exception as e:
            fail += 1
            print(f"    Failed on '{wav_path.name}' after {_fmt_s(_now()-t0)}: {e}\n", flush=True)

    print(f"✅ Finished encoding: {ok}/{total} succeeded, {fail} failed in {_fmt_s(_now()-t_all)}.", flush=True)


def reconstruct_dir_from_bins(
    bin_dir: Path | str,
    json_dir: Optional[Path | str],
    wav_out_dir: Path | str,
    *,
    n_iter: int = 32,
) -> None:
    """
    Reconstruct .wav files for every .bin in bin_dir using JSON found in json_dir.
    If json_dir is None, JSON is expected next to each BIN.
    """
    t_all = _now()
    bin_dir = Path(bin_dir)
    wav_out_dir = Path(wav_out_dir)
    json_dir = Path(json_dir) if json_dir is not None else None

    wav_out_dir.mkdir(parents=True, exist_ok=True)

    bin_files = sorted(bin_dir.glob("*.bin"))
    total = len(bin_files)
    if not bin_files:
        print(f"No .bin files found in {bin_dir}")
        return

    print(f"🔊 Reconstructing {total} audio file(s) from '{bin_dir}' → WAVs in '{wav_out_dir}'"
          + (f" (JSON in '{json_dir}')" if json_dir else " (JSON next to BIN)"),
          flush=True)

    ok = 0
    skip = 0
    fail = 0
    for i, bin_path in enumerate(bin_files, start=1):
        t0 = _now()
        # Confirm JSON exists (fail early with a helpful message)
        sidecar_json = _json_path_for(bin_path, json_dir)
        if not sidecar_json.exists():
            skip += 1
            print(f"[{i:>4}/{total}] ({_percent(i-1, total):>3}%) {bin_path.name} → ⚠️  Missing JSON: {sidecar_json}. Skipping.\n", flush=True)
            continue

        try:
            wav_out = wav_out_dir / (bin_path.stem + ".wav")
            print(f"[{i:>4}/{total}] ({_percent(i-1, total):>3}%) {bin_path.name} → {wav_out.name}", flush=True)
            reconstruct_bin_to_audio(bin_path, json_dir=json_dir, out_wav=wav_out, n_iter=n_iter)
            ok += 1
            print(f"    Done in {_fmt_s(_now()-t0)} [{_percent(i, total):>3}%]\n", flush=True)
        except Exception as e:
            fail += 1
            print(f"    Failed on '{bin_path.name}' after {_fmt_s(_now()-t0)}: {e}\n", flush=True)

    print(f"✅ Finished reconstruction: {ok}/{total} succeeded, {skip} skipped, {fail} failed in {_fmt_s(_now()-t_all)}.", flush=True)


# ---------------------------------------------
# NEW: BIN (+ JSON) -> spectrogram(s)
# ---------------------------------------------

def reconstruct_logstft_from_bin(
    bin_path: Path | str,
    *,
    json_dir: Optional[Path | str] = None,
    use_mask: bool = True,
    denormalize: bool = True,
) -> Tuple[torch.Tensor, Dict]:
    """
    Reconstruct the (optionally de-normalized) log-magnitude STFT [F, T] from a BIN+JSON.

    Args:
        bin_path: path to .bin
        json_dir: directory where the JSON sidecar is stored (if not next to BIN)
        use_mask: if True and T_real present, trim padded frames
        denormalize: if True and (log_min, log_max) present, map back to original log scale

    Returns:
        (logstft_FxT, meta)
        - logstft_FxT: torch.FloatTensor on CPU with shape [F, T]
    """
    t0 = _now()
    bin_path = Path(bin_path)
    print(f"→ Reconstructing log-STFT from '{bin_path.name}' …", flush=True)
    spikes, meta = read_spikes_bin_with_meta(bin_path, json_dir=json_dir)
    mode = meta.get("mode", "phased_rate")

    # Build mask from T_real if available
    mask = None
    if use_mask and "T_real" in meta and meta["T_real"] is not None:
        T = int(meta["T"])
        T_real = int(meta["T_real"])
        mask = torch.zeros(T, dtype=torch.float32)
        mask[:min(T_real, T)] = 1.0

    recon_log = reconstruct_from_spikes(spikes, mode=mode, mask=mask, trim=use_mask and mask is not None)
    logstft_FxT = recon_log.cpu().T  # [F, T] normalized (0..1) if normalize=True during encode
    if denormalize and meta.get("log_min") is not None and meta.get("log_max") is not None:
        log_min = float(meta["log_min"])
        log_max = float(meta["log_max"])
        logstft_FxT = logstft_FxT * (log_max - log_min) + log_min

    print(f"✓ Reconstructed log-STFT from '{bin_path.name}' in {_fmt_s(_now()-t0)} "
          f"(shape={tuple(logstft_FxT.shape)})", flush=True)
    return logstft_FxT, meta


def save_spectrogram_image_from_bin(
    bin_path: Path | str,
    *,
    json_dir: Optional[Path | str] = None,
    out_image: Path | str,
    denormalize: bool = True,
    to_db: bool = False,
    figsize: Tuple[int, int] = (10, 4),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_axes: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Generate and save a spectrogram image from a BIN+JSON.

    Args:
        bin_path: path to .bin
        json_dir: directory where JSON is stored (if different from BIN dir)
        out_image: file path to save (e.g., 'spec.png')
        denormalize: if True, map normalized log scale back using (log_min, log_max)
        to_db: if True, convert to dB (20*log10(magnitude)); otherwise plot log(mag) or normalized units
        figsize: matplotlib figure size
        vmin, vmax: optional color limits for imshow
        show_axes: if False, hide axes

    Returns:
        (S_for_plot, meta)
        - S_for_plot is the 2D array actually plotted (numpy [F, T]).
    """
    import matplotlib.pyplot as plt

    t0 = _now()
    bin_path = Path(bin_path)
    out_image = Path(out_image)
    print(f"→ Saving spectrogram for '{bin_path.name}' → '{out_image.name}' …", flush=True)

    logstft_FxT, meta = reconstruct_logstft_from_bin(
        bin_path, json_dir=json_dir, use_mask=True, denormalize=denormalize
    )  # [F, T]

    # Convert to the plotting domain
    # logstft_FxT is log(magnitude+1e-6) if denormalized; otherwise 0..1 normalized
    if to_db:
        # Convert to magnitude then dB
        magnitude = torch.exp(logstft_FxT) - 1e-6
        S = 20.0 * torch.log10(torch.clamp(magnitude, min=1e-6))
    else:
        # Plot logstft directly (denormalized or normalized)
        S = logstft_FxT

    S_np = S.numpy()

    # Time/frequency extents
    sr = int(meta.get("sample_rate", 16000))
    hop = int(meta["hop_length"])
    n_fft = int(meta["n_fft"])
    F = S_np.shape[0]
    T = S_np.shape[1]
    duration_sec = T * hop / sr
    freq_max = sr / 2.0

    extent = [0.0, duration_sec, 0.0, freq_max]  # x: seconds, y: Hz (0..Nyquist)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        S_np,
        aspect="auto",
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("dB" if to_db else ("log magnitude" if denormalize else "normalized"))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    title = Path(bin_path).stem
    ax.set_title(f"Spectrogram: {title}")

    if not show_axes:
        ax.set_axis_off()

    out_image.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_image, dpi=150)
    plt.close(fig)

    print(f"✓ Saved '{out_image.name}' (F={F}, T={T}, sr={sr}, hop={hop}, n_fft={n_fft}) "
          f"in {_fmt_s(_now()-t0)}", flush=True)

    return S_np, meta


def save_spectrograms_for_dir(
    bin_dir: Path | str,
    *,
    json_dir: Optional[Path | str] = None,
    out_dir: Path | str,
    pattern: str = "*.bin",
    denormalize: bool = True,
    to_db: bool = False,
    figsize: Tuple[int, int] = (10, 4),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_axes: bool = True,
) -> None:
    """
    Batch-generate spectrogram images for all BINs in a directory.

    Args:
        bin_dir: directory of BIN files
        json_dir: directory of JSON sidecars (if different from BIN dir)
        out_dir: where to save images
        pattern: glob pattern for BIN files
        denormalize, to_db, figsize, vmin, vmax, show_axes: passed to save_spectrogram_image_from_bin
    """
    t_all = _now()
    bin_dir = Path(bin_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(bin_dir.glob(pattern))
    total = len(files)
    if not files:
        print(f"No files matched {pattern} under {bin_dir}")
        return

    print(f"Generating spectrograms for {total} BIN(s) from '{bin_dir}' → images in '{out_dir}'"
          + (f" (JSON in '{json_dir}')" if json_dir else " (JSON next to BIN)"),
          flush=True)

    ok = 0
    skip = 0
    fail = 0
    for i, bin_path in enumerate(files, start=1):
        t0 = _now()
        img_path = out_dir / f"{bin_path.stem}.png"
        # Skip if JSON missing
        sidecar_json = _json_path_for(bin_path, Path(json_dir) if json_dir else None)
        if not sidecar_json.exists():
            skip += 1
            print(f"[{i:>4}/{total}] ({_percent(i-1, total):>3}%) {bin_path.name} → Missing JSON: {sidecar_json}. Skipping.\n", flush=True)
            continue

        try:
            print(f"[{i:>4}/{total}] ({_percent(i-1, total):>3}%) {bin_path.name} → {img_path.name}", flush=True)
            save_spectrogram_image_from_bin(
                bin_path,
                json_dir=json_dir,
                out_image=img_path,
                denormalize=denormalize,
                to_db=to_db,
                figsize=figsize,
                vmin=vmin,
                vmax=vmax,
                show_axes=show_axes,
            )
            ok += 1
            print(f"    ✓ Done in {_fmt_s(_now()-t0)} [{_percent(i, total):>3}%]\n", flush=True)
        except Exception as e:
            fail += 1
            print(f"    ❌ Failed on '{bin_path.name}' after {_fmt_s(_now()-t0)}: {e}\n", flush=True)

    print(f"Finished spectrograms: {ok}/{total} succeeded, {skip} skipped, {fail} failed in {_fmt_s(_now()-t_all)}.", flush=True)
