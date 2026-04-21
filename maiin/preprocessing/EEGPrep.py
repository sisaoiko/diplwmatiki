#!/usr/bin/env python
# coding: utf-8

# In[2]:
import pickle
import numpy as np
import mne
from mne.preprocessing import ICA
import mne_icalabel
import matplotlib.pyplot as plt
import autoreject 
from autoreject import AutoReject,get_rejection_threshold
from scipy.interpolate import interp1d 
from scipy.signal import savgol_filter, find_peaks
import os
import glob
import pandas as pd

def createRaw(arr, sfreq=2048,trigger_idx=0, eeg_start=1,eeg_end=65,extra_end =89,l_freq=1,h_freq=55):
    orig_ch_names = (
        ["Trig1"]
        + [f"A{i}" for i in range(1, 33)]
        + [f"B{i}" for i in range(1, 33)] 
        +["EX1","EX2","EX3","EX4","EX5","EX6","EX7","EX8","AUX1","AUX2","AUX3","AUX4","AUX5","AUX6","AUX7","AUX8","AUX9","AUX10","AUX11","AUX12","AUX13","AUX14","AUX15","AUX16"]
    )
    eeg_dataa = arr[eeg_start:eeg_end, :]
    trigger = arr[trigger_idx, :]
    extra_data= arr[eeg_end: extra_end, :]
# conversion of raw BioSemi ADC units (integer counts) to volts
    eeg_data_volts= eeg_dataa* 1e-6
    extra_data_volts = extra_data* 1e-6
    #print("Max after scaling:", np.max(np.abs(eeg_data_volts)))

# ---------------------------------------------------
# 3. Create MNE Raw object
# ---------------------------------------------------

    ch_types = ["stim"]+ ["eeg"] * (eeg_end - eeg_start) +["ecg"] * 2 + ["misc"] * 22
    raw_data = np.vstack([trigger[None, :], eeg_data_volts, extra_data_volts])
    info = mne.create_info(ch_names=orig_ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(raw_data, info)
    montage = mne.channels.make_standard_montage("biosemi64")
    eeg_names = montage.ch_names  

    rename_dict = {"Trig1": "TRIGGER"}
    for old, new in zip(orig_ch_names[eeg_start:eeg_end], eeg_names):
        rename_dict[old] = new

    raw.rename_channels(rename_dict)
    raw.set_channel_types({"TRIGGER": "stim"})
    raw.set_montage(montage)
    #bads = mne.preprocessing.find_bad_channels_lof(raw, n_neighbors=20, picks='eeg', metric='euclidean', threshold=5)
    #raw.info['bads'] = bads
    #raw.interpolate_bads(reset_bads=True)
    raw = raw.set_eeg_reference(ref_channels='average')
    raw.filter(l_freq=l_freq, h_freq=h_freq) 
    raw.resample(128.0, npad = "auto")
    return raw    
#raw_ica = raw.copy()
    #raw.save(
     #   r"E:\koorathota_data\pkl_data\pkl_data\pkl_data\newpkl\segmenteddata\subject_12_session_02_raw.fif",
      #  overwrite=True
    #)
def ICAProcess(raw_ica):
    artifact_classes = [
    "eye blink",
    "muscle artifact",
    "heart",
    "line noise",
    "channel noise",
    ]
    
    artifacts = []
    ica = ICA(n_components=30,  method='infomax', random_state=42, max_iter='auto')
    raw_eeg = raw_ica.copy().pick(picks='eeg')
    tstep = 1.0
    events = mne.make_fixed_length_events(raw_eeg, duration=tstep)
    epochs = mne.Epochs(raw_eeg,events,tmin=0.0,tmax=tstep,baseline=None,preload=True,reject_by_annotation=True,verbose=False,)
    reject = get_rejection_threshold(epochs)
    ica.fit(epochs, reject=reject, tstep=tstep)
    #ica.fit(raw_eeg)
    labels = mne_icalabel.label_components(raw_ica, ica, method= 'iclabel')
    #eog_inds, eog_scores = ica.find_bads_eog(raw_ica, ch_name="Fpz",threshold=3)
    #eog_inds1, eog_scores1 = ica.find_bads_eog(raw_ica, ch_name=["F7","F8"],threshold=3)
    #for i, (label, prob) in enumerate(zip(labels["labels"], labels["y_pred_proba"])):
     #   if label in artifact_classes and prob > 0.95:
      #      print(f"Excluding {label} (Prob: {prob:.2f})")
       #     artifacts.append(i)
    for i in range(len(labels["labels"])):
        label = labels["labels"][i]
        prob = labels["y_pred_proba"][i]
        if label in ("eye blink","heart","line noise") and prob>0.9:
            print(label, prob)
            if i in artifacts:
                print("nothing")
            else:
                artifacts.append(i)
    ica.exclude =artifacts
    #ica.exclude = list(set(eog_inds) | set(artifacts))
    ica.apply(raw_ica)
    n_removed = len(ica.exclude)
    n_total = ica.n_components_
    return n_removed, n_total

def get_variance_df(raw_dirty, raw_clean):
    """
    Compares variance between raw and clean EEG data and returns a Pandas DataFrame.
    """
    # 1. Pick only EEG channels to exclude triggers and other sensors
    # We work on copies to avoid altering your main variables
    dirty_eeg = raw_dirty.copy().pick_types(eeg=True)
    clean_eeg = raw_clean.copy().pick_types(eeg=True)
    
    # 2. Extract data and channel names
    data_dirty = dirty_eeg.get_data()
    data_clean = clean_eeg.get_data()
    ch_names = dirty_eeg.ch_names
    
    # 3. Calculate metrics using NumPy vectorized operations for speed
    # We calculate variance along the time axis (axis=1)
    vars_before = np.var(data_dirty, axis=1)
    vars_after = np.var(data_clean, axis=1)
    
    # Calculate % Reduction
    # Avoid division by zero if a channel is flat
    reduction = np.where(vars_before > 0, (1 - (vars_after / vars_before)) * 100, 0)
    
    # 4. Create the DataFrame
    df = pd.DataFrame({
        'Channel': ch_names,
        'Raw_Variance': vars_before,
        'Clean_Variance': vars_after,
        'Percent_Reduction': reduction
    })
    
    # 5. Sort by Percent_Reduction so blinks (highest reduction) are at the top
    df = df.sort_values(by='Percent_Reduction', ascending=False).reset_index(drop=True)
    
    return df

def move_peak_to_start(peaks, signal, flat_tol=0.001):
    snapped = []
    for p in peaks:
        val = signal[p]
        idx = p
        # Walk backwards while the value stays within flat_tol of the peak
        while idx > 0 and abs(signal[idx - 1] - val) <= flat_tol:
            idx -= 1
        snapped.append(idx)
    return np.array(snapped, dtype=int)
    
def find_onsets(peaks, direction, all_peaks, search_window, smooth):
    onsets = []
    for i, p in enumerate(peaks):
        start = max(0, p - search_window)
        prior = all_peaks[all_peaks < p]
        if len(prior) > 0:
            start = max(start, prior[-1])
        segment = smooth[start:p]
        if len(segment) == 0:
            continue
        if direction == "left":
            onset = start + np.argmin(segment)
        elif direction == "right":
            onset = start + np.argmax(segment)
        onsets.append(onset)
    return onsets
    
    
def move_peak_to_end(onsets, signal, flat_tol=0.001):
    
    snapped = []
    for p in onsets:
        val = signal[p]
        idx = p
        # Walk backwards while the value stays within flat_tol of the peak
        while idx < len(signal) - 1 and abs(signal[idx + 1] - val) <= flat_tol:
            idx += 1
        snapped.append(idx)
    return np.array(snapped, dtype=int)
def findSteeringEvents(eeg_time,eeg_timestamps,motor_timestamps,steering,FS,AMP_THRESHOLD = 0.3):

#df = pd.read_pickle(eeg_path)
#steer = np.load(motor_path)
#eeg_timestamps = data['BioSemi'][1]
    motor_timestamps_eeg = motor_timestamps-eeg_timestamps[0]
    interpolator = interp1d(
        motor_timestamps_eeg, 
        steering, 
        kind='linear',
        bounds_error=False,
        fill_value=(steering[0], steering[-1])
    )
#N = min(len(df), len(steer))

#steer = steer[:N]
#signal = steer
    signal = interpolator(eeg_time)
# =========================
# SMOOTH SIGNAL
# =========================

    smooth = savgol_filter(signal, 21, 3)

# =========================
# PEAK DETECTION
# =========================

    prom = np.percentile(np.abs(smooth), 75)

# positive steering = LEFT
    left_peaks, _ = find_peaks(
        smooth,
        prominence=prom,
        distance=FS
    )

# negative steering = RIGHT
    right_peaks, _ = find_peaks(
        -smooth,
        prominence=prom,
        distance=FS
    )

    left_peaks = left_peaks.astype(int)
    right_peaks = right_peaks.astype(int)


    left_peaks  = move_peak_to_start(left_peaks,  smooth)
    right_peaks = move_peak_to_start(right_peaks, smooth)
# =========================
# ONSET DETECTION
# =========================

#import numpy as np



    all_peaks = np.sort(np.concatenate([left_peaks, right_peaks]))

    search_window = int(5 * FS)  # fix typo
    left_onsets  = find_onsets(left_peaks,  "left",  all_peaks,search_window,smooth)
    right_onsets = find_onsets(right_peaks, "right", all_peaks,search_window,smooth)
    
    left_onsets  = move_peak_to_end(left_onsets,  smooth)
    right_onsets = move_peak_to_end(right_onsets, smooth)



# =========================
# AMPLITUDE FILTER
# =========================

    filtered_left_peaks = []
    filtered_left_onsets = []

    for onset, peak in zip(left_onsets, left_peaks):

        amplitude = abs(smooth[peak]-smooth[onset])

        if amplitude >= AMP_THRESHOLD:
            filtered_left_peaks.append(peak)
            filtered_left_onsets.append(onset)

    filtered_left_peaks = np.array(filtered_left_peaks, dtype=int)
    filtered_left_onsets = np.array(filtered_left_onsets, dtype=int)


    filtered_right_peaks = []
    filtered_right_onsets = []

    for onset, peak in zip(right_onsets, right_peaks):

        amplitude = abs(smooth[peak]-smooth[onset])

        if amplitude >= AMP_THRESHOLD:
            filtered_right_peaks.append(peak)
            filtered_right_onsets.append(onset)

    filtered_right_peaks = np.array(filtered_right_peaks, dtype=int)
    filtered_right_onsets = np.array(filtered_right_onsets, dtype=int)

    print("Left turns:", len(filtered_left_peaks))
    print("Right turns:", len(filtered_right_peaks))
    return filtered_left_peaks,filtered_right_peaks, filtered_left_onsets,filtered_right_onsets,smooth
def eventsCreation (filtered_left_peaks,filtered_right_peaks, filtered_left_onsets,filtered_right_onsets,smooth):
    events = []
    # for onsets in filtered_left_onsets:
        # events.append([int(onsets),0,1])
    # for onsets in filtered_right_onsets:
        # events.append([int(onsets),0,2])
    #for peaks in filtered_left_peaks:
     #   events.append([int(peaks),0,3])
    #for peaks in filtered_right_peaks:
     #   events.append([int(peaks),0,4])
    y_class = []
    y_reg = []

# left = 0, right = 1
    for onset, peak in zip(filtered_left_onsets, filtered_left_peaks):
        events.append([int(onset), 0, 1])
        y_class.append(0)
        y_reg.append(abs(smooth[peak] - smooth[onset]))

    for onset, peak in zip(filtered_right_onsets, filtered_right_peaks):
        events.append([int(onset), 0, 2])
        y_class.append(1)
        y_reg.append(abs(smooth[peak] - smooth[onset]))


    
    
    events = np.array(events, dtype=int)
    y_class = np.array(y_class, dtype=int)
    y_reg = np.array(y_reg, dtype=float)

    order = np.argsort(events[:, 0])
    events = events[order]
    y_class = y_class[order]
    y_reg = y_reg[order]

    return events, y_class, y_reg

event_id = {
    "left_onset": 1,
    "right_onset": 2,
    #"left_peak": 3,
    #"right_peak": 4,
}
def epochsCreation (raw, eeg_timestamps,steering,motor_timestamps):
    fs=raw.info['sfreq']
    # Ο χρόνος έναρξης του EEG
# Μετατροπή χρόνου σε δείγματα (indices)
    #event_samples = np.round( times-eeg_start_time)*fs
    #event_samples += raw.first_samp
    #limits = ((event_samples>=raw.first_samp) & (event_samples< raw.first_samp + raw.n_times))
    #event_samples = event_samples[limits].astype(int)
# Δημιουργία πίνακα events για το MNE (3 στήλες: sample, 0, event_id)
    #events = np.column_stack([
     #   event_samples,
      #  np.zeros(len(event_samples), dtype=int),
       # np.ones(len(event_samples), dtype=int)
    #])
    #event_id = dict(motor=1)
    eeg_time = raw.times
    filtered_left_peaks,filtered_right_peaks, filtered_left_onsets,filtered_right_onsets,smooth = findSteeringEvents(eeg_time,eeg_timestamps,
                                                                                                              motor_timestamps,
                                                                                                              steering,
                                                                                                              FS=fs)
    events,y_class,y_reg = eventsCreation(filtered_left_peaks,filtered_right_peaks, filtered_left_onsets,filtered_right_onsets,smooth)
    events[:, 0] += raw.first_samp
    epochs = mne.Epochs(
        raw, 
        events, 
        event_id=event_id,
        tmin=-3.0,           
        tmax=0.0,            
        baseline=(-3.0, -2.8), 
        preload=True,
        on_missing='warn',
        event_repeated='merge'
    )
    return epochs,filtered_left_peaks,filtered_right_peaks, filtered_left_onsets,filtered_right_onsets,events,y_class,y_reg 

def Autorejectf (epochs):
 
    n_interpolates = [1, 4, 8, 10]
    consensus_percs = [0.2, 0.35, 0.5, 0.8]
    ar = autoreject.AutoReject(n_interpolates, consensus_percs, picks='eeg',
                thresh_method='bayesian_optimization', random_state=42)
    
    ar.fit(epochs)
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)
    #evoked_after = AutorejectValidation(epochs_ar,epochs, reject_log)
    if hasattr(ar, 'cv_scores_'):
        print("Επιτυχές Fit! Το RMSE είναι διαθέσιμο.")
    return epochs_ar, reject_log
def AutorejectValidation (epochs_ar,epochs, reject_log):
    evoked_before = epochs.average()

# Μέσος όρος μετά τον καθαρισμό με Autoreject
    evoked_after = epochs_ar.average()

# Σχεδίαση των Butterfly plots
    res=evoked_before.plot(titles='Before Autoreject')
    res=evoked_after.plot(titles='After Autoreject')
    res=reject_log.plot('horizontal')
    return evoked_after
def finalPreprocesseeg(data):
    required = ['Unity_MotorInput', 'BioSemi', 'Unity_ViveSREyeTracking']
    if not all(k in data for k in required):
        return None

    arr = data['BioSemi'][0]
    raw = createRaw(arr)
    del arr

    raw_ica = raw.copy().pick(picks='eeg')
    n_removed, n_total = ICAProcess(raw_ica)

    #df_results = get_variance_df(raw.pick(picks='eeg'), raw_ica)

    motor = data['Unity_MotorInput']
    steering = motor[0][0]
    motor_timestamps = motor[1]
    eeg_timestamps = data['BioSemi'][1]
    info = data['Unity_TrialInfo']

    epochs, filtered_left_peaks, filtered_right_peaks, filtered_left_onsets, filtered_right_onsets, events, y_class, y_reg = epochsCreation(
        raw_ica, eeg_timestamps, steering, motor_timestamps
    )

    n_before = len(epochs)
    left_before = np.sum(y_class == 0)
    right_before = np.sum(y_class == 1)

    epochs_ar, reject_log = Autorejectf(epochs)

    keep_mask = ~reject_log.bad_epochs
    events = events[keep_mask]
    y_class = y_class[keep_mask]
    y_reg = y_reg[keep_mask]

    n_after = len(epochs_ar)
    left_after = np.sum(y_class == 0)
    right_after = np.sum(y_class == 1)
    rej_pct = 100 * (n_before - n_after) / n_before if n_before > 0 else 0.0

    stats = {
        "n_before": int(n_before),
        "n_after": int(n_after),
        "rej_pct": float(rej_pct),
        "left_before": int(left_before),
        "right_before": int(right_before),
        "left_after": int(left_after),
        "right_after": int(right_after),
    }

    return (
        raw_ica,
        epochs_ar,
        reject_log,
        filtered_left_peaks,
        filtered_right_peaks,
        filtered_left_onsets,
        filtered_right_onsets,
        events,
        y_class,
        y_reg,
        stats,
    )

# In[ ]:




