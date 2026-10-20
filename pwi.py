
import numpy as np
import random
import math
import os
import time
import xarray as xr
from scipy        import signal
from scipy        import stats
from scipy.interpolate import interp1d
from sklearn.cluster   import AgglomerativeClustering
from multiprocessing   import Pool, cpu_count
from tqdm              import tqdm
import pandas as pd
import numpy as np
import segyio

# kept for any legacy code that imports it
FIXED_OUTPUT_COLS = 260


# =============================================================================
# CLUSTER
# =============================================================================

def Cluster(impedance_matrix, smoothed_facies_matrix, nclus):
    n_simul = impedance_matrix.shape[0]
    k       = min(nclus, n_simul)

    labels    = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(
                    impedance_matrix)
    centroids = np.zeros((k, impedance_matrix.shape[1]), dtype=np.float64)
    for c in range(k):
        mask = labels == c
        if mask.any():
            centroids[c] = impedance_matrix[mask].mean(axis=0)
    return centroids, labels


# =============================================================================
# ARRAY INIT
# =============================================================================

def initialize_output_arrays(total_layers, number_of_simulations, middle_layers):
    return (
        np.zeros((total_layers,  number_of_simulations)),  # shale_volume_models
        np.zeros((total_layers,  number_of_simulations)),  # impedance_models
        np.zeros((total_layers,  number_of_simulations)),  # facies_models
        np.zeros((middle_layers, number_of_simulations)),  # smoothed_facies_models
        np.zeros(total_layers),  # impedance
        np.zeros(total_layers),  # state
        np.zeros(total_layers),  # expanded_state
        np.zeros(total_layers),  # shale_volume
        np.zeros(total_layers),  # facies
        np.zeros(total_layers),  # layer_thickness
    )


# =============================================================================
# MARKOV CHAIN HELPERS
# =============================================================================

def sample_markov_state(probability_vector):
    probability_vector    = np.atleast_2d(probability_vector)
    cumulative_probability = 0.0
    selected_index         = 0
    random_number          = np.random.uniform(0, 1)
    for state_index in range(probability_vector.shape[1]):
        lo = cumulative_probability
        hi = lo + probability_vector[0, state_index]
        if lo < random_number <= hi:
            selected_index = state_index
        cumulative_probability = hi
    return selected_index + 1


def compute_lambda(rv, lo, hi):
    return ((1 - rv) ** 3) * (hi - lo) + lo


def generate_thickness(state, multiplier,
                       ls, lss, lsh,
                       sand_range, sand_shale_range, shale_range):
    if state == 1:
        rf = random.uniform(*sand_range)
        return -multiplier * math.log(1.0 - rf) / ls
    elif state == 2:
        rf = random.uniform(*sand_shale_range)
        return -multiplier * math.log(1.0 - rf) / lss
    else:
        rf = random.uniform(*shale_range)
        return -multiplier * math.log(1.0 - rf) / lsh


def assign_petrophysical_values(current_state):
    if current_state == 1:
        fv = 1;  vsh = random.uniform(0.0, 0.2)
    elif current_state == 2:
        fv = 2;  vsh = random.uniform(0.5, 0.7)
    else:
        fv = 3;  vsh = random.uniform(0.9, 1.0)
    imp = 1.0 / (vsh / 1090 + (1 - vsh) / 780)
    return fv, vsh, imp


# =============================================================================
# MARKOV CHAIN SIMULATION
# =============================================================================

def Simulation(nsimul,
               nlayers_mclr1, nlayers_mclr2, nlayers_mclr3, totallayers,
               x1, y1, x2, y2, x3, y3, multiplier):

    (vsh_m, imp_m, fc_m, sfc_m,
     impedance, state, expanded_state,
     shale_volume, facies, layer_thickness) = initialize_output_arrays(
         totallayers, nsimul, nlayers_mclr2)

    for si in range(nsimul):
        rl_s  = random.uniform(0.1, 1.0)
        rl_ss = random.uniform(0.1, 1.0)
        rl_sh = random.uniform(0.1, 1.0)

        uzt  = nlayers_mclr3 + nlayers_mclr2
        mzt  = nlayers_mclr3
        lzt  = 0
        utm  = y1.copy();  mtm = y2.copy();  ltm = y3.copy()

        for li in range(totallayers):

            if li < nlayers_mclr3:                               # LOWER
                cs  = sample_markov_state(ltm);  ltm = ltm @ x3
                ls  = compute_lambda(rl_s,  0.25, 1.0)
                lss = compute_lambda(rl_ss, 0.25, 1.0)
                lsh = compute_lambda(rl_sh, 0.10, 0.7)
                t   = generate_thickness(cs, multiplier, ls, lss, lsh,
                                         (0.25,0.5),(0.3,0.7),(0.3,0.9))
                prev = lzt;  lzt += int(np.rint(t))
                if   nlayers_mclr3 >= lzt:          ct = t
                elif nlayers_mclr3 >  prev:         ct = nlayers_mclr3 - prev
                else:                               ct = 0

            elif li < nlayers_mclr3 + nlayers_mclr2:            # MIDDLE
                cs  = sample_markov_state(mtm);  mtm = mtm @ x2
                ls  = compute_lambda(rl_s,  0.15, 0.7)
                lss = compute_lambda(rl_ss, 0.20, 1.0)
                lsh = compute_lambda(rl_sh, 0.15, 0.7)
                t   = generate_thickness(cs, multiplier, ls, lss, lsh,
                                         (0.3,0.7),(0.3,0.7),(0.3,0.7))
                prev = mzt;  mzt += int(np.rint(t))
                lim  = nlayers_mclr3 + nlayers_mclr2
                if   lim >= mzt:   ct = t
                elif lim >  prev:  ct = lim - prev
                else:              ct = 0

            else:                                                # UPPER
                cs  = sample_markov_state(utm);  utm = utm @ x1
                ls  = compute_lambda(rl_s,  0.25, 1.0)
                lss = compute_lambda(rl_ss, 0.40, 1.3)
                lsh = compute_lambda(rl_sh, 0.10, 0.7)
                t   = generate_thickness(cs, multiplier, ls, lss, lsh,
                                         (0.25,0.5),(0.3,0.7),(0.3,0.9))
                prev = uzt;  uzt += int(np.rint(t))
                lim  = nlayers_mclr3 + nlayers_mclr2 + nlayers_mclr1
                if   lim >= uzt:   ct = t
                elif lim >  prev:  ct = lim - prev
                else:              ct = 0

            layer_thickness[totallayers - 1 - li] = int(np.rint(ct))
            state[totallayers - 1 - li]           = cs

        # expand
        ei = 0
        for l in range(totallayers):
            for _ in range(int(layer_thickness[l])):
                expanded_state[ei] = state[l]
                facies[ei], shale_volume[ei], impedance[ei] = \
                    assign_petrophysical_values(int(expanded_state[ei]))
                ei += 1

        impedance[totallayers-1]    = impedance[totallayers-2]
        facies[totallayers-1]       = facies[totallayers-2]
        shale_volume[totallayers-1] = shale_volume[totallayers-2]

        vsh_m[:, si]  = shale_volume
        imp_m[:, si]  = impedance
        fc_m[:, si]   = facies
        sfc_m[:, si]  = facies[nlayers_mclr1: nlayers_mclr1 + nlayers_mclr2]

    return vsh_m, imp_m, fc_m, sfc_m


# =============================================================================
# LEGACY HELPER (kept for back-compat)
# =============================================================================

def _fix_width(arr, fill_value=0):
    arr = np.asarray(arr).ravel()
    n   = len(arr)
    if n == FIXED_OUTPUT_COLS:
        return arr
    out = np.full(FIXED_OUTPUT_COLS, fill_value, dtype=arr.dtype)
    out[:min(n, FIXED_OUTPUT_COLS)] = arr[:min(n, FIXED_OUTPUT_COLS)]
    return out


# =============================================================================
# SINGLE-TRACE WORKER
# Returns full-length arrays at the simulation fine grid (required_si),
# plus twt_start so the caller can do the resampling.
# =============================================================================

def parallel_bmps(args, nlayers_mclr1=80, nlayers_mclr3=80):
    (seis, ht, hb,
     start_time, end_time,
     required_si, si_seis,
     total_seis_samples) = args

    seis   = np.asarray(seis, dtype=np.float32).ravel()
    dt_ms  = required_si * 1000.0

    # ---- band-pass filter ----
    si_filter  = 0.004
    nyq_filter = 1.0 / (2.0 * si_filter)
    N, Wn = signal.buttord(
        [ 9 / nyq_filter, 50 / nyq_filter],
        [ 4 / nyq_filter, 90 / nyq_filter],
        3, 60)
    b, a      = signal.butter(N, Wn, btype="band")
    impulse   = np.zeros(256);  impulse[128] = 1
    a_filt    = signal.filtfilt(b, a, impulse)

    # ---- Markov matrices ----
    x1 = np.array([[0.,0.,1.],[0.,0.,1.],[0.,1.,0.]])
    y1 = np.array([[0.,0.,1.]])
    x2 = np.array([[0.,1.,0.],[.5,0.,.5],[0.,1.,0.]])
    y2 = np.array([[0.,0.,1.]])
    x3 = np.array([[0.,0.,1.],[0.,0.,1.],[0.,1.,0.]])
    y3 = np.array([[0.,0.,1.]])

    # ---- horizon → sample indices on fine grid ----
    hts_rsamp     = int(((ht - start_time) / dt_ms) + 1)
    hbs_rsamp     = int(((hb - start_time) / dt_ms) + 1)
    nlayers_mclr2 = max(1, hbs_rsamp - hts_rsamp + 1)
    totallayers   = nlayers_mclr1 + nlayers_mclr2 + nlayers_mclr3
    multiplier    = 0.001 / required_si

    # TWT of first simulation sample (used by caller for resampling)
    twt_start = start_time + (hts_rsamp - 1 - nlayers_mclr1) * dt_ms

    # ---- simulation ----
    bmps, bmps_imp, bmps_fc, bmps_smth = Simulation(
        300, nlayers_mclr1, nlayers_mclr2, nlayers_mclr3, totallayers,
        x1, y1, x2, y2, x3, y3, multiplier)

    # ---- resample input seismic to fine grid for matching ----
    t_simp   = np.arange(start_time, end_time + si_seis*1000, si_seis*1000)
    min_len  = min(len(t_simp), len(seis))
    simp_rs, _ = signal.resample(seis[:min_len], total_seis_samples,
                                  t=t_simp[:min_len])
    simp_rs    = simp_rs.flatten()

    sw = max(0, hts_rsamp - nlayers_mclr1)
    ew = min(len(simp_rs) - 1, hbs_rsamp + nlayers_mclr3)
    simp_rs_selec = simp_rs[sw: ew + 1]
    if len(simp_rs_selec) < totallayers:
        simp_rs_selec = np.pad(simp_rs_selec,
                               (0, totallayers - len(simp_rs_selec)),
                               mode="edge")

    scalar = 0.00000000035 * (required_si / 0.00025)

    # ---- clustering + scoring ----
    ctrd, lbls = Cluster(bmps_imp.T, bmps_smth.T, 30)

    param_score = np.empty(ctrd.shape[0])
    param_corr  = np.empty(ctrd.shape[0])
    param_rmse  = np.empty(ctrd.shape[0])

    for k in range(ctrd.shape[0]):
        filt  = np.convolve(signal.detrend(ctrd[k]), a_filt, mode="same") * scalar
        n     = min(len(filt), len(simp_rs_selec))
        corr  = np.corrcoef(filt[:n], simp_rs_selec[:n])[0, 1]
        rmse  = np.sqrt(np.mean((filt[:n] - simp_rs_selec[:n])**2))
        param_corr[k]  = corr
        param_rmse[k]  = rmse
        param_score[k] = corr + (1.0 - rmse) / 2.0

    best  = int(np.argmax(param_score))
    mask  = lbls == best

    mean_vsh  = np.mean(bmps.T[mask],    axis=0)   # (totallayers,)
    std_vsh   = np.std( bmps.T[mask],    axis=0)
    _mr       = stats.mode(bmps_fc.T[mask], axis=0, keepdims=False)
    mode_fc   = _mr.mode.astype(np.float32)
    count_fc  = _mr.count.astype(np.float32)
    median_fc = np.median(bmps_fc.T[mask], axis=0).astype(np.float32)

    # Return full-length arrays + metadata for per-trace resampling
    return (
        mean_vsh.astype(np.float32),
        mode_fc,
        median_fc,
        std_vsh.astype(np.float32),
        count_fc,
        float(np.mean(param_corr)),
        float(np.mean(param_rmse)),
        int(totallayers),        # length of the arrays above
        float(twt_start),        # TWT of sample[0] in the arrays above
    )


# =============================================================================
# ARGUMENT GENERATOR
# =============================================================================

def _args_generator(seismic, htx, hbx,
                    start_time, end_time,
                    required_si, si_seis,
                    total_seis_samples, ntraces):
    for i in range(ntraces):
        yield (seismic[i], htx[i], hbx[i],
               start_time, end_time,
               required_si, si_seis, total_seis_samples)


# =============================================================================
# RESAMPLE ONE SIMULATION RESULT → OUTPUT TWT AXIS
# =============================================================================

def _resample_to_output_grid(data, twt_start, totallayers,
                              required_si, out_twt):
    """
    Interpolate a 1-D simulation result (fine grid, variable length)
    onto out_twt (any grid, fixed length).  Zeros outside the window.
    """
    dt_ms   = required_si * 1000.0
    sim_twt = twt_start + np.arange(totallayers) * dt_ms
    f       = interp1d(sim_twt, data,
                       kind="linear", bounds_error=False, fill_value=0.0)
    return f(out_twt).astype(np.float32)


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_workflow(
    seis_ds: "xr.Dataset",
    amp_var_name:  str,
    twt_axis_name: str,
    il_axis_name:  str,
    xl_axis_name:  str,
    start_time:    float,
    end_time:      float,
    topbot_file_df,
    output_name:   str,
    output_dir:    str   = "output",
    n_workers:     int   = None,
    required_si:   float = 0.00025,
) -> "xr.Dataset":
    """
    Run the full PWI workflow and return (+ save) an xarray Dataset.

    The output Dataset has the SAME iline / xline / twt coordinates as
    seis_ds so results overlay perfectly in any viewer.

    Parameters
    ----------
    topbot_file_df : DataFrame with columns il, xl, ht, hb.
                     Extra rows (outside the loaded cube) are ignored.
                     Missing traces get NaN output.
    """
    import xarray as xr

    # ---- geometry ----
    iline_vals = seis_ds[il_axis_name].values
    xline_vals = seis_ds[xl_axis_name].values
    twt_vals   = seis_ds[twt_axis_name].values.astype(np.float64)

    n_il  = len(iline_vals)
    n_xl  = len(xline_vals)
    n_twt = len(twt_vals)

    si_seis = float(twt_vals[1] - twt_vals[0]) / 1000.0   # ms → s

    # ---- validation ----
    twt_min, twt_max = float(twt_vals.min()), float(twt_vals.max())
    if start_time < twt_min or end_time > twt_max:
        raise ValueError(
            f"start_time/end_time ({start_time}–{end_time} ms) outside "
            f"TWT axis ({twt_min}–{twt_max} ms)")

    for col in ("il", "xl", "ht", "hb"):
        if col not in topbot_file_df.columns:
            raise ValueError(f"topbot_file_df missing column '{col}'")

    # ---- build per-trace horizon lookup  {(il, xl): (ht, hb)} ----
    # Extra rows are silently dropped; missing traces → NaN
    valid_il = set(iline_vals.tolist())
    valid_xl = set(xline_vals.tolist())
    horizon_lookup = {
        (int(r.il), int(r.xl)): (float(r.ht), float(r.hb))
        for r in topbot_file_df.itertuples(index=False)
        if int(r.il) in valid_il and int(r.xl) in valid_xl
    }
    print(f"Horizon entries matching cube : "
          f"{len(horizon_lookup)} / {n_il * n_xl} traces")

    # ---- build seismic array in trace order ----
    seismic = (seis_ds[amp_var_name]
               .values
               .reshape(n_il * n_xl, n_twt)
               .astype(np.float32))
    ntraces = seismic.shape[0]

    # per-trace ht / hb arrays aligned with trace order
    # traces with no horizon entry get median values (will produce NaN output)
    med_ht = float(np.median(topbot_file_df["ht"].values))
    med_hb = float(np.median(topbot_file_df["hb"].values))

    htx = np.empty(ntraces, dtype=np.float64)
    hbx = np.empty(ntraces, dtype=np.float64)
    has_horizon = np.zeros(ntraces, dtype=bool)

    for idx in range(ntraces):
        il_v = int(iline_vals[idx // n_xl])
        xl_v = int(xline_vals[idx  % n_xl])
        if (il_v, xl_v) in horizon_lookup:
            htx[idx], hbx[idx] = horizon_lookup[(il_v, xl_v)]
            has_horizon[idx]   = True
        else:
            htx[idx] = med_ht;  hbx[idx] = med_hb

    # ---- output arrays on seismic TWT grid ----
    out_twt = twt_vals.astype(np.float32)

    vsh_cube      = np.full((ntraces, n_twt), np.nan, dtype=np.float32)
    vsh_std_cube  = np.full((ntraces, n_twt), np.nan, dtype=np.float32)
    fc_mode_cube  = np.full((ntraces, n_twt), np.nan, dtype=np.float32)
    fc_med_cube   = np.full((ntraces, n_twt), np.nan, dtype=np.float32)
    fc_cnt_cube   = np.full((ntraces, n_twt), np.nan, dtype=np.float32)
    corr_map      = np.full(ntraces,          np.nan, dtype=np.float32)
    rmse_map      = np.full(ntraces,          np.nan, dtype=np.float32)

    if n_workers is None:
        n_workers = min(max(1, cpu_count() - 2), 16)

    total_seis_samples = int((end_time - start_time) / (required_si * 1000)) + 1

    print("=" * 70)
    print(f"TRACES          : {ntraces}  ({n_il} IL × {n_xl} XL)")
    print(f"WITH HORIZON    : {has_horizon.sum()}")
    print(f"OUTPUT TWT      : {n_twt} samples at {si_seis*1000:.4f} ms  "
          f"({out_twt[0]:.1f}–{out_twt[-1]:.1f} ms)")
    print(f"WORKERS         : {n_workers}")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    gen = _args_generator(seismic, htx, hbx,
                          start_time, end_time,
                          required_si, si_seis,
                          total_seis_samples, ntraces)

    t0 = time.time()
    with Pool(n_workers) as pool:
        for tr_idx, result in enumerate(tqdm(
            pool.imap(parallel_bmps, gen, chunksize=10),
            total=ntraces, desc="Processing", unit="trace", dynamic_ncols=True
        )):
            (mean_vsh, mode_fc, median_fc, std_vsh, count_fc,
             corr, rmse, totallayers, twt_start) = result

            if not has_horizon[tr_idx]:
                continue   # leave as NaN

            # Resample each output from fine simulation grid → seismic TWT axis
            vsh_cube[tr_idx]     = _resample_to_output_grid(
                mean_vsh,  twt_start, totallayers, required_si, out_twt)
            vsh_std_cube[tr_idx] = _resample_to_output_grid(
                std_vsh,   twt_start, totallayers, required_si, out_twt)
            fc_mode_cube[tr_idx] = _resample_to_output_grid(
                mode_fc,   twt_start, totallayers, required_si, out_twt)
            fc_med_cube[tr_idx]  = _resample_to_output_grid(
                median_fc, twt_start, totallayers, required_si, out_twt)
            fc_cnt_cube[tr_idx]  = _resample_to_output_grid(
                count_fc,  twt_start, totallayers, required_si, out_twt)
            corr_map[tr_idx]     = corr
            rmse_map[tr_idx]     = rmse

    elapsed = round((time.time() - t0) / 60, 2)
    tqdm.write(f"Runtime: {elapsed} min")

    # ---- reshape to (n_il, n_xl, n_twt) ----
    def _r3(arr): return arr.reshape(n_il, n_xl, n_twt)
    def _r2(arr): return arr.reshape(n_il, n_xl)

    # ---- build xarray Dataset ----
    ds_out = xr.Dataset(
        data_vars=dict(
            vsh           =(["iline","xline","twt"], _r3(vsh_cube),
                             {"long_name":"Mean Vsh",           "units":"fraction"}),
            vsh_std       =(["iline","xline","twt"], _r3(vsh_std_cube),
                             {"long_name":"Std Dev Vsh",        "units":"fraction"}),
            facies_mode   =(["iline","xline","twt"], _r3(fc_mode_cube),
                             {"long_name":"Mode Facies (1=sand,2=mixed,3=shale)"}),
            facies_median =(["iline","xline","twt"], _r3(fc_med_cube),
                             {"long_name":"Median Facies"}),
            facies_count  =(["iline","xline","twt"], _r3(fc_cnt_cube),
                             {"long_name":"Facies Count (best cluster)"}),
            correlation   =(["iline","xline"],       _r2(corr_map),
                             {"long_name":"Mean Correlation"}),
            rmse          =(["iline","xline"],       _r2(rmse_map),
                             {"long_name":"Mean RMSE"}),
        ),
        coords=dict(
            iline=(["iline"], iline_vals),
            xline=(["xline"], xline_vals),
            twt  =(["twt"],   out_twt,    {"units":"ms","long_name":"Two-Way Time"}),
        ),
        attrs=dict(
            description  = "PWI inversion output",
            output_name  = output_name,
            start_time   = start_time,
            end_time     = end_time,
            required_si  = required_si,
            si_seis_ms   = si_seis * 1000.0,
            runtime_min  = elapsed,
        ),
    )

    nc_path = os.path.join(output_dir, f"{output_name}_PWI.nc")
    ds_out.to_netcdf(nc_path)
    print(f"Saved → {nc_path}")
    return ds_out

import pandas as pd

# =============================================================================
# MAIN
# =============================================================================

def main():

    topbot_df = pd.read_csv(
        "AR1_EEI70_IL5101-5400_XL6001-6398_topbot.txt",
        sep=r"\s+",
        names=["il", "xl", "ht", "hb"]
    )

    start_twt = 4500

    segy_path = (
        r"D:\MyPythonProjects\Harsh_Samit_sir_PWI"
        r"\PWIS_Harsh\decimated_output_4S.sgy"
    )

    print("Loading SEGY cube...")

    with segyio.open(segy_path, "r") as f:

        cube = segyio.tools.cube(f)

        seis_ds = xr.Dataset(
            data_vars={
                "amp": (
                    ("iline", "xline", "twt"),
                    cube
                )
            },
            coords={
                "iline": np.asarray(f.ilines),
                "xline": np.asarray(f.xlines),
                "twt": np.asarray(f.samples) + start_twt
            }
        )

    print(seis_ds)

    run_workflow(
        seis_ds=seis_ds,
        amp_var_name="amp",
        twt_axis_name="twt",
        il_axis_name="iline",
        xl_axis_name="xline",
        start_time=4500.0,
        end_time=5500.0,
        topbot_file_df=topbot_df,
        output_name="AR1_EEI",
        output_dir="OUTPUT_AR1_1_TEST_1",
        n_workers=16,
    )


if __name__ == "__main__":

    import multiprocessing as mp

    mp.freeze_support()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    main()
