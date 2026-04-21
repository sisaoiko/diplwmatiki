#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def dilation_speed(t_s, d):
    d = np.asarray(d, float)
    

    dspeed = np.full(len(d), np.nan)
    for i in range(len(d)):
        vals = []

        if i > 0 and np.isfinite(d[i]) and np.isfinite(d[i-1]) and t_s[i] != t_s[i-1]:
            vals.append(abs(d[i] - d[i-1]) / abs(t_s[i] - t_s[i-1]))

        if i < len(d)-1 and np.isfinite(d[i]) and np.isfinite(d[i+1]) and t_s[i+1] != t_s[i]:
            vals.append(abs(d[i+1] - d[i]) / abs(t_s[i+1] - t_s[i]))

        if vals:
            dspeed[i] = max(vals)

    return dspeed

def mad_threshold(x, n=16):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return med + n * mad

def find_nan_runs(mask):
    """
    Find contiguous True runs in a boolean mask.
    Returns list of (start_idx, end_idx) inclusive.
    """
    starts =[]
    ends=[]
    for i in range(len(mask)):
        if i== 0 and mask[i]:
            starts.append(i)
        elif mask[i] and not mask[i-1]:
            starts.append(i)
        if i==len(mask)-1 and mask[i]:
            ends.append(i)
        elif i <len(mask)-1 and mask[i] and not mask[i+1]:
            ends.append(i)
    return list(zip(starts, ends))


def merge_close_gaps(mask, t_s, max_separation_s=0.100):
    """
    Merge NaN gaps if the valid chunk between them is shorter than max_separation_ms.
    """
    mask = np.asarray(mask, dtype=bool).copy()
   

    runs = find_nan_runs(mask)
    if len(runs) < 2:
        return mask

    merged_mask = mask.copy()
    current_start, current_end = runs[0]

    for next_start, next_end in runs[1:]:
        separation_s = t_s[next_start] - t_s[current_end]

        if separation_s < max_separation_s:
            merged_mask[current_start:next_end + 1] = True
            current_end = next_end
        else:
            current_start, current_end = next_start, next_end

    return merged_mask


def pad_blink_gaps(mask, t_s, min_gap_s=0.075, max_gap_s=0.500,
                    pad_before_s=0.050, pad_after_s=0.050):
    """
    For gaps with duration > min_gap_ms and < max_gap_ms,
    extend the mask by pad_before_ms before and pad_after_ms after.
    """
    mask = np.asarray(mask, dtype=bool).copy()
    
    runs = find_nan_runs(mask)
    padded_mask = mask.copy()

    for start, end in runs:
        gap_duration_s = t_s[end] - t_s[start]

        if (gap_duration_s >= min_gap_s): #and (gap_duration_ms <= max_gap_ms):
            start_time = t_s[start] - pad_before_s
            end_time = t_s[end] + pad_after_s
            extra = (t_s >= start_time) & (t_s <= end_time)
            padded_mask[extra] = True

    return padded_mask


import numpy as np
from scipy.interpolate import CubicSpline

def interpolate_short_gaps_cubic(t_s, y, max_gap_s=0.040):
    """
    Interpolate NaN gaps shorter than max_gap_ms using cubic spline,
    with 2 valid points before and 2 valid points after the gap.
    """
   
    y = np.asarray(y, dtype=float).copy()
    t_s = np.asarray(t_s, dtype=float)
    nan_mask = np.isnan(y)
    runs = find_nan_runs(nan_mask)

    for start, end in runs:
        gap_duration_s = t_s[end] - t_s[start]

        if gap_duration_s < max_gap_s:
            left1 = start - 1
            left2 = start - 2
            right1 = end + 1
            right2 = end + 2

            # check bounds
            if left2 >= 0 and right2 < len(y):
                # check support points are finite
                if (
                    np.isfinite(y[left2]) and
                    np.isfinite(y[left1]) and
                    np.isfinite(y[right1]) and
                    np.isfinite(y[right2])
                ):
                    # time points must be ordered
                    x_t = [t_s[left2], t_s[left1], t_s[right1], t_s[right2]]
                    y_t = [y[left2], y[left1], y[right1], y[right2]]

                    # extra safety: strictly increasing x
                    if np.all(np.diff(x_t) > 0):
                        cs = CubicSpline(x_t, y_t, bc_type='natural')
                        y[start:end+1] = cs(t_s[start:end+1])

    return y


def preprocess_pupil_gaps(t_s, pupil_with_blinks):
    """
    Full gap-processing stage:
    1) merge close gaps
    2) pad medium-sized gaps
    3) interpolate only short gaps
    """
    t_s = np.asarray(t_s, dtype=float)
    pupil_with_nans = np.asarray(pupil_with_blinks, dtype=float)

    initial_nan_mask = np.isnan(pupil_with_nans)

    merged_mask = merge_close_gaps(
        initial_nan_mask, t_s )

    padded_mask = pad_blink_gaps(
        merged_mask, t_s)

    pupil_after_masking = pupil_with_nans.copy()
    pupil_after_masking[padded_mask] = np.nan

    pupil_interpolated = interpolate_short_gaps_cubic( t_s, pupil_after_masking )

    return {
        "initial_nan_mask": initial_nan_mask,
        "merged_mask": merged_mask,
        "padded_mask": padded_mask,
        "pupil_after_masking": pupil_after_masking,
        "pupil_interpolated": pupil_interpolated,
    }

from scipy.interpolate import interp1d 
def resample_to_uniform_grid(eye_timestamps, signal, target_hz=120):
    """
    Resample to a uniform grid.
    NaN gaps are preserved
    """
    eye_t= 1.0 / target_hz
    t_uniform = np.arange(eye_timestamps[0], eye_timestamps[-1], eye_t)

    # Only interpolate over finite segments
    finite_mask = np.isfinite(signal)
    if finite_mask.sum() < 2:
        return t_uniform, np.full(len(t_uniform), np.nan)

    # Interpolate only within finite regions
    f = interp1d(
        eye_timestamps[finite_mask], signal[finite_mask],
        kind='linear',
        bounds_error=False,
        fill_value=np.nan  # NaN outside known data = preserves gaps
    )
    return t_uniform, f(t_uniform)

import numpy as np

def create_mean_pupil_size_offset(t_s, pupil_cleanL, pupil_cleanR):
    t_s = np.asarray(t_s, dtype=float)
    L = np.asarray(pupil_cleanL, dtype=float)
    R = np.asarray(pupil_cleanR, dtype=float)

    mean_pupil = np.full(len(L), np.nan)

    both = np.isfinite(L) & np.isfinite(R)
    only_L = np.isfinite(L) & np.isnan(R)
    only_R = np.isnan(L) & np.isfinite(R)

    # plain mean when both exist
    mean_pupil[both] = (L[both] + R[both]) / 2.0

    # estimate dynamic offset where both are present
    offset = np.full(len(L), np.nan)
    offset[both] = L[both] - R[both]

    if np.sum(both) >= 2:
        offset_interp = np.interp(t_s, t_s[both], offset[both])
    else:
        offset_interp = np.full(len(L), np.nan)

    # if only left exists, estimate mean from left and offset
    # mean = (L + R)/2, with R ≈ L - offset
    mean_pupil[only_L] = L[only_L] - offset_interp[only_L] / 2.0

    # if only right exists, estimate mean from right and offset
    # L ≈ R + offset
    mean_pupil[only_R] = R[only_R] + offset_interp[only_R] / 2.0

    return mean_pupil

from scipy.signal import butter, filtfilt
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    finite = np.isfinite(data)
    starts =[]
    ends=[]
    for i in range(len(finite)):
        if i== 0 and finite[i]:
            starts.append(i)
        elif finite[i] and not finite[i-1]:
            starts.append(i)
        if i==len(finite)-1 and finite[i]:
            ends.append(i)
        elif i <len(finite)-1 and finite[i] and not finite[i+1]:
            ends.append(i)
    for s, e in zip(starts, ends):
        segment = data[s:e]
        if len(segment) > 3 * order:
            data[s:e] = filtfilt(b, a, segment)

    return data


def rolling_zscore_causal(signal, timestamps, window_sec=120.0):
    """
    Causal rolling Z-score on a continuous pupil signal.
    At each sample t, normalizes using only the past `window_sec` seconds.
    NaNs (blinks) are ignored in mean/std computation but preserved in output.

    Parameters
    ----------
    signal     : 1D array, continuous pupil (may contain NaNs for blinks)
    timestamps : 1D array, corresponding timestamps in seconds
    window_sec : lookback window in seconds (default 2 min)

    Returns
    -------
    z : 1D array, causal Z-scored signal (NaNs preserved)
    """
    signal = np.asarray(signal, float).copy()
    z = np.full(len(signal), np.nan)

    for t in range(1, len(signal)):
        t_now = timestamps[t]
        # only look back up to window_sec
        mask = (timestamps >= t_now - window_sec) & (timestamps < t_now)
        past = signal[mask]
        finite_past = past[np.isfinite(past)]  # ignore NaNs from blinks

        if len(finite_past) < 10:             # not enough past data yet
            continue

        mean  = finite_past.mean()
        sd  = finite_past.std()

        if sd < 1e-8:
            z[t] = 0.0
        else:
            z[t] = (signal[t] - mean) / sd

    return z


import numpy as np
import numpy as np

def extract_pupil_epochs(
    pupil_signal,
    t_uniform,
    events,
    sfreq,
    tmin=-2.5,
    tmax=0.0,
    baseline=(-2.5, -2.3),
    max_nan_ratio=0.2,
    min_finite_baseline=5
):
    pupil_signal = np.asarray(pupil_signal, dtype=float)
    t_uniform = np.asarray(t_uniform, dtype=float)
    events = np.asarray(events, dtype=int)

    if len(pupil_signal) != len(t_uniform):
        raise ValueError("pupil_signal and t_uniform must have the same length")

    if len(t_uniform) < 2:
        raise ValueError("t_uniform must contain at least 2 samples")

    dt = np.median(np.diff(t_uniform))
    if not np.allclose(np.diff(t_uniform), dt, rtol=1e-3, atol=1e-6):
        raise ValueError("t_uniform must be approximately uniform")

    pupil_fs = 1.0 / dt
    n_times = int(round((tmax - tmin) / dt)) + 1
    rel_times = np.linspace(tmin, tmax, n_times)
    n_times = len(rel_times)

    event_samples = events[:, 0]
    event_times = event_samples / sfreq

    epochs = []
    epoch_info = []
    rejected = []
    kept_indices = []

    for i, event_time in enumerate(event_times):
        start_time = event_time + tmin

        start_idx = np.searchsorted(t_uniform, start_time, side="left")
        stop_idx = start_idx + n_times

        if stop_idx > len(pupil_signal):
            rejected.append({
                "event_index": i,
                "event_time": float(event_time),
                "reason": "window out of bounds"
            })
            continue

        epoch = pupil_signal[start_idx:stop_idx].copy()

        if len(epoch) != n_times:
            rejected.append({
                "event_index": i,
                "event_time": float(event_time),
                "reason": "wrong epoch length"
            })
            continue

        nan_ratio = np.mean(~np.isfinite(epoch))
        if nan_ratio > max_nan_ratio:
            rejected.append({
                "event_index": i,
                "event_time": float(event_time),
                "reason": f"NaN ratio {nan_ratio:.3f} > {max_nan_ratio}"
            })
            continue

        baseline_mean = np.nan
        if baseline is not None:
            b0, b1 = baseline
            bmask = (rel_times >= b0) & (rel_times <= b1)
            baseline_vals = epoch[bmask]
            finite_baseline = baseline_vals[np.isfinite(baseline_vals)]

            if len(finite_baseline) < min_finite_baseline:
                rejected.append({
                    "event_index": i,
                    "event_time": float(event_time),
                    "reason": "not enough finite baseline samples"
                })
                continue

            baseline_mean = np.mean(finite_baseline)
            epoch = epoch - baseline_mean

        epochs.append(epoch)
        kept_indices.append(i)
        epoch_info.append({
            "event_index": i,
            "event_sample": int(event_samples[i]),
            "event_time": float(event_time),
            "baseline_mean": float(baseline_mean) if np.isfinite(baseline_mean) else np.nan,
            "nan_ratio": float(nan_ratio),
            "n_times": int(n_times),
            "pupil_fs": float(pupil_fs),
        })

    if len(epochs) == 0:
        epochs = np.empty((0, n_times), dtype=float)
        kept_indices = np.empty((0,), dtype=int)
    else:
        epochs = np.asarray(epochs, dtype=float)
        kept_indices = np.asarray(kept_indices, dtype=int)

    return epochs, epoch_info, rejected, rel_times, kept_indices


def finalPreprocesseye(data,events):
    eye = data['Unity_ViveSREyeTracking'][0]
    eye_timestamps =data['Unity_ViveSREyeTracking'][1]
    eeg = data['BioSemi']
    arr = eeg[0]
    eeg_timestamps = eeg[1]
    pupilDiameterL = eye[0]
    pupilDiameterR= eye[1]
    openessL = eye[2]
    openessR =eye[3]
    pupilDL = np.array(pupilDiameterL, dtype=float)
    openessL = np.array(openessL, dtype=float)
    eye_timestamps = eye_timestamps - eeg_timestamps[0]
    blinkL = openessL == 0
    invalid = (pupilDL <= 0) | blinkL

    pupil_maskL = pupilDL.copy()
    pupil_maskL[invalid] = np.nan
    spdL = dilation_speed(eye_timestamps, pupil_maskL)
    thrL = mad_threshold(spdL, n=16)
    speed_outlierL = spdL > thrL
    pupil_blinkL = pupil_maskL.copy()
    pupil_blinkL[speed_outlierL] = np.nan
    resultL = preprocess_pupil_gaps(eye_timestamps,pupil_blinkL)
    pupil_cleanL = resultL['pupil_interpolated']
    
    #RIGHT EYE
    pupilDR = np.array(pupilDiameterR, dtype=float)
    openessR = np.array(openessR, dtype=float)
    #eye_timestamps = eye_timestamps - eeg_timestamps[0]
    blinkR = openessR == 0
    invalid = (pupilDR <= 0) | blinkR

    pupil_maskR = pupilDR.copy()
    pupil_maskR[invalid] = np.nan
    spdR = dilation_speed(eye_timestamps, pupil_maskR)
    thrR = mad_threshold(spdR, n=16)
    speed_outlierR = spdR > thrR
    pupil_blinkR = pupil_maskR.copy()
    pupil_blinkR[speed_outlierR] = np.nan
    resultR = preprocess_pupil_gaps(eye_timestamps,pupil_blinkR)
    pupil_cleanR = resultR['pupil_interpolated']
    
    
    #MEAN
    mean_pupil = create_mean_pupil_size_offset(eye_timestamps,pupil_cleanL,pupil_cleanR)
    t_uniform, resampled_pupilL =resample_to_uniform_grid(eye_timestamps,pupil_cleanL)
    t_uniform, resampled_pupilR =resample_to_uniform_grid(eye_timestamps,pupil_cleanR)
    t_uniform, resampled_pupil_mean =resample_to_uniform_grid(eye_timestamps,mean_pupil)
    pupil_filteredL = butter_lowpass_filter(resampled_pupilL, cutoff=4, fs=120, order=4)
    pupil_filteredR = butter_lowpass_filter(resampled_pupilR, cutoff=4, fs=120, order=4)
    pupil_filteredM = butter_lowpass_filter(resampled_pupil_mean, cutoff=4, fs=120, order=4)
    pupil_normL   = rolling_zscore_causal(pupil_filteredL, t_uniform,window_sec=120.0)
    pupil_normR   = rolling_zscore_causal(pupil_filteredR, t_uniform,window_sec=120.0)
    pupil_normM   = rolling_zscore_causal(pupil_filteredM, t_uniform,window_sec=120.0)
    pupil_epochsL,epoch_infoL, rejectedL, rel_timesL, kept_indicesL = extract_pupil_epochs(pupil_normL,t_uniform,events,sfreq=128)
    pupil_epochsR,epoch_infoR, rejectedR, rel_timesR, kept_indicesR = extract_pupil_epochs(pupil_normR,t_uniform,events,sfreq=128)
    pupil_epochsM,epoch_infoM, rejectedM, rel_timesM, kept_indicesM = extract_pupil_epochs(pupil_normM,t_uniform,events,sfreq=128)
    return (pupil_epochsL,epoch_infoL, rejectedL, rel_timesL, kept_indicesL,pupil_epochsR,epoch_infoR, rejectedR, rel_timesR, kept_indicesR,pupil_epochsM,epoch_infoM, rejectedM, rel_timesM, kept_indicesM)
    

