from __future__ import annotations

from typing import Optional, Tuple


def _match_length(signal, target_len: int):
    import numpy as np
    if len(signal) == target_len:
        return signal
    if len(signal) > target_len:
        return signal[:target_len]
    pad = target_len - len(signal)
    return np.pad(signal, (0, pad), mode='constant')


def denoise_dfnet3(y, sr: int, enable: bool = True) -> Tuple[Optional[object], str]:
    """
    DeepFilterNet3 denoise wrapper (robust + Windows-friendly).

    - Prefers native `deepfilternet` package API when available.
    - Falls back to `df` package API if present.
    - Preserves exact output length to keep downstream alignments stable.
    - On Windows, avoids soxr DLL issues by using resampy.

    Input assumptions: 1D mono array at `sr` (commonly 16k).
    Returns: (y_denoised or None, info)
    """
    if not enable:
        return None, 'disabled'

    # Robust numpy/librosa import (handle Windows soxr issues)
    try:
        import os, sys
        os.environ.setdefault('LIBROSA_RESAMPLER', 'resampy')
        try:
            import numpy as np  # noqa: F401
            import librosa
        except Exception:
            # Provide a benign stub for 'soxr' if librosa tries to import it
            import types as _types, importlib
            sys.modules.setdefault('soxr', _types.SimpleNamespace(__version__='0'))
            import numpy as np  # noqa: F401
            librosa = importlib.import_module('librosa')
    except Exception as e:
        return None, f'missing numpy/librosa ({e})'

    # 1) Prefer `df` package API (shipped by DeepFilterNet pip)
    try:
        from df.enhance import enhance, init_df

        try:
            model, df_state, _ = init_df()
        except Exception as e:
            # Common cause: onnxruntime missing
            return None, f'df init failed: {type(e).__name__}'

        target_sr = 48000
        x48 = y.astype('float32')
        if sr != target_sr:
            x48 = librosa.resample(x48, orig_sr=sr, target_sr=target_sr)
        out48 = enhance(model, df_state, x48)
        out16 = out48.astype('float32')
        if target_sr != sr:
            out16 = librosa.resample(out16, orig_sr=target_sr, target_sr=sr)
        out16 = _match_length(out16.astype('float32'), len(y))
        return out16, 'df package api'
    except Exception:
        pass

    # 2) Try official deepfilternet module (may not be importable)
    try:
        import deepfilternet as dfn

        try:
            model = dfn.DeepFilterNet.load_pretrained('deepfilternet3')
        except Exception:
            model = dfn.DeepFilterNet.from_pretrained('deepfilternet3')

        target_sr = getattr(model, 'sample_rate', 48000)
        x = y.astype('float32')
        if sr != target_sr:
            x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
        out = model.enhance(x, sr=target_sr)
        if target_sr != sr:
            out = librosa.resample(out.astype('float32'), orig_sr=target_sr, target_sr=sr)
        out = _match_length(out.astype('float32'), len(y))
        return out, 'deepfilternet api'
    except Exception:
        pass

    return None, 'DeepFilterNet3 not available'
