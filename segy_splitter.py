"""
segy_decimate.py
----------------
Reduce a SEG-Y file size by decimating traces (horizontal) and/or
time samples (vertical).

Install
-------
    pip install segyio numpy scipy tqdm

Run
---
    python segy_decimate.py
"""

import os
import numpy as np
import segyio
from tqdm import tqdm

# =============================================================================
# CONFIGURATION  — edit these
# =============================================================================

INPUT_FILE  = "Rseries_REEI16_CI_inhouse_AR1_IL5101-5400_XL6001-6398_IEEE.sgy"
OUTPUT_FILE = "decimated_output_4S.sgy"

IL_STEP          = 4      # keep every Nth inline
XL_STEP          = 4      # keep every Nth crossline
T_STEP           = 4      # keep every Nth time sample
APPLY_ANTIALIAS  = True   # low-pass before time decimation (recommended)

# =============================================================================


def decimate_segy(input_file, output_file, il_step, xl_step, t_step, antialias):

    print("=" * 60)
    print(f"Input   : {input_file}")
    print(f"IL step : {il_step}   XL step : {xl_step}   T step : {t_step}")
    print(f"Antialias filter : {antialias}")
    print("=" * 60)

    with segyio.open(input_file, "r", ignore_geometry=False) as src:

        inlines_all = src.ilines.copy()   # just the array, not the Line object
        xlines_all  = src.xlines.copy()
        twt_all     = src.samples.copy()

        inlines_dec = inlines_all[::il_step]
        xlines_dec  = xlines_all[::xl_step]
        twt_dec     = twt_all[::t_step]

        n_il     = len(inlines_dec)
        n_xl     = len(xlines_dec)
        n_t      = len(twt_dec)
        n_xl_all = len(xlines_all)

        si_orig = float(twt_all[1] - twt_all[0])
        si_new  = si_orig * t_step

        print(f"\nOriginal  : {len(inlines_all)} IL  x  {n_xl_all} XL  x  {len(twt_all)} samples")
        print(f"Decimated : {n_il} IL  x  {n_xl} XL  x  {n_t} samples")
        print(f"Sample interval : {si_orig:.3f} ms  →  {si_new:.3f} ms")
        print(f"Expected reduction : ~{il_step * xl_step * t_step}x\n")

        # build index maps  {il_value: position_in_full_array}
        il_index = {v: i for i, v in enumerate(inlines_all)}
        xl_index = {v: i for i, v in enumerate(xlines_all)}

        # anti-alias filter
        if antialias and t_step > 1:
            from scipy.signal import butter, filtfilt
            nyq    = 1.0 / (2.0 * si_orig * 1e-3)
            cutoff = (nyq / t_step) * 0.8
            b, a   = butter(4, cutoff / nyq, btype="low")
        else:
            b = a = None

        spec          = segyio.spec()
        spec.sorting  = segyio.TraceSortingFormat.INLINE_SORTING
        spec.format   = segyio.SegySampleFormat.IEEE_FLOAT_4_BYTE
        spec.ilines   = inlines_dec
        spec.xlines   = xlines_dec
        spec.samples  = twt_dec

        with segyio.create(output_file, spec) as dst:

            dst.text[0] = src.text[0]
            dst.bin.update(
                tsort = segyio.TraceSortingFormat.INLINE_SORTING,
                hdt   = int(si_new * 1000),
                dto   = int(si_new * 1000),
                hns   = n_t,
                nso   = n_t,
            )

            written = 0

            for il in tqdm(inlines_dec, desc="Writing inlines", unit="IL"):
                il_pos = il_index[il]

                for xl in xlines_dec:
                    xl_pos = xl_index[xl]

                    # flat trace index in the original file
                    src_idx = il_pos * n_xl_all + xl_pos

                    trace = src.trace[src_idx].astype(np.float32)

                    if b is not None:
                        trace = filtfilt(b, a, trace).astype(np.float32)

                    trace_dec = trace[::t_step]

                    dst.header[written] = {
                        **{k: src.header[src_idx][k] for k in src.header[src_idx]},
                        segyio.TraceField.INLINE_3D            : int(il),
                        segyio.TraceField.CROSSLINE_3D         : int(xl),
                        segyio.TraceField.TRACE_SAMPLE_INTERVAL: int(si_new * 1000),
                        segyio.TraceField.TRACE_SAMPLE_COUNT   : n_t,
                    }
                    dst.trace[written] = trace_dec
                    written += 1

    size_in  = os.path.getsize(input_file)  / 1e6
    size_out = os.path.getsize(output_file) / 1e6
    print(f"\nDone.")
    print(f"Input  : {size_in:.1f} MB")
    print(f"Output : {size_out:.1f} MB")
    print(f"Reduction : {size_in / size_out:.1f}x")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    decimate_segy(
        input_file  = INPUT_FILE,
        output_file = OUTPUT_FILE,
        il_step     = IL_STEP,
        xl_step     = XL_STEP,
        t_step      = T_STEP,
        antialias   = APPLY_ANTIALIAS,
    )