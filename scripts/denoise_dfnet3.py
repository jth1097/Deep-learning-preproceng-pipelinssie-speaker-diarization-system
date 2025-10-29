from __future__ import annotations

from typing import Optional, Tuple


def denoise_dfnet3(y, sr: int, enable: bool = True) -> Tuple[Optional[object], str]:
    """
    DeepFilterNet3 denoise wrapper with graceful fallback.

    - Tries `deepfilternet` API first, then `df` API.
    - Resamples to model SR (typically 48 kHz) and back to `sr` if needed.
    - Returns (y_denoised or None, info string).
    """
    if not enable:
        return None, 'disabled'

    try:
        import numpy as np  # noqa: F401
        import librosa
    except Exception as e:
        return None, f'missing numpy/librosa ({e})'

    # 1) Try official deepfilternet package
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
        return out.astype('float32'), 'deepfilternet api'
    except Exception:
        pass

    # 2) Fallback to `df` package API
    try:
        from df.enhance import enhance, init_df
        from df.io import resample as df_resample

        model, df_state, _ = init_df()
        target_sr = 48000
        x48 = df_resample(y.astype('float32'), sr_in=sr, sr_out=target_sr)
        out48 = enhance(model, df_state, x48)
        out16 = df_resample(out48.astype('float32'), sr_in=target_sr, sr_out=sr)
        return out16.astype('float32'), 'df package api'
    except Exception:
        pass

    return None, 'DeepFilterNet3 not available'

