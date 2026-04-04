import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="MRI Theory Simulation", layout="wide")


# -----------------------------
# Data models
# -----------------------------
@dataclass
class Tissue:
    name: str
    t1_ms: float
    t2_ms: float
    proton_density: float


TISSUES: Dict[str, Tissue] = {
    "White Matter": Tissue("White Matter", t1_ms=850, t2_ms=80, proton_density=0.65),
    "Gray Matter": Tissue("Gray Matter", t1_ms=1350, t2_ms=100, proton_density=0.78),
    "CSF": Tissue("CSF", t1_ms=4000, t2_ms=2000, proton_density=1.00),
    "Fat": Tissue("Fat", t1_ms=260, t2_ms=70, proton_density=0.90),
    "Muscle": Tissue("Muscle", t1_ms=900, t2_ms=50, proton_density=0.75),
    "Blood": Tissue("Blood", t1_ms=1600, t2_ms=250, proton_density=0.95),
}


PATIENTS = {
    "Healthy Adult": {"heart_rate": 72, "resp_rate": 14, "spo2": 99, "motion": 0.10, "flow": 25},
    "Anxious Patient": {"heart_rate": 105, "resp_rate": 22, "spo2": 98, "motion": 0.35, "flow": 28},
    "Sedated Patient": {"heart_rate": 60, "resp_rate": 10, "spo2": 97, "motion": 0.03, "flow": 18},
    "Pediatric Patient": {"heart_rate": 110, "resp_rate": 24, "spo2": 99, "motion": 0.28, "flow": 30},
    "Cardiac Instability": {"heart_rate": 125, "resp_rate": 26, "spo2": 94, "motion": 0.25, "flow": 35},
}


SEQUENCE_INFO = {
    "Spin Echo": {"mode": "T2", "artifact_sensitivity": 0.5, "speed_factor": 1.0},
    "Gradient Echo": {"mode": "T2*", "artifact_sensitivity": 1.0, "speed_factor": 0.8},
    "EPI": {"mode": "T2*", "artifact_sensitivity": 1.4, "speed_factor": 0.15},
    "Inversion Recovery": {"mode": "T1", "artifact_sensitivity": 0.6, "speed_factor": 1.3},
}


# -----------------------------
# Physics helpers
# -----------------------------
def ernst_angle_deg(tr_ms: float, t1_ms: float) -> float:
    if tr_ms <= 0 or t1_ms <= 0:
        return 0.0
    value = math.exp(-tr_ms / t1_ms)
    value = min(max(value, -1.0), 1.0)
    return math.degrees(math.acos(value))



def compute_t2_star(t2_ms: float, field_inhomogeneity: float, sequence_name: str) -> float:
    seq_factor = SEQUENCE_INFO[sequence_name]["artifact_sensitivity"]
    # Larger inhomogeneity shortens T2*
    inhomo_component = max(5.0, 250.0 / (1 + 5 * field_inhomogeneity * seq_factor))
    inv_t2_star = 1.0 / max(t2_ms, 1e-6) + 1.0 / inhomo_component
    return 1.0 / inv_t2_star



def longitudinal_recovery(time_ms: np.ndarray, t1_ms: float, m0: float = 1.0) -> np.ndarray:
    return m0 * (1 - np.exp(-time_ms / t1_ms))



def transverse_decay(time_ms: np.ndarray, t2_ms: float, mxy0: float = 1.0) -> np.ndarray:
    return mxy0 * np.exp(-time_ms / t2_ms)



def base_signal(tr_ms: float, te_ms: float, tissue: Tissue, sequence_name: str, flip_angle_deg: float, field_inhomogeneity: float) -> float:
    mode = SEQUENCE_INFO[sequence_name]["mode"]
    alpha = math.radians(flip_angle_deg)
    e1 = math.exp(-tr_ms / tissue.t1_ms)

    # Gradient echo / steady-state-inspired signal approximation
    gre_signal = tissue.proton_density * math.sin(alpha) * (1 - e1) / max(1e-6, (1 - math.cos(alpha) * e1))

    # Spin echo / inversion recovery simplified signal approximations
    se_signal = tissue.proton_density * (1 - math.exp(-tr_ms / tissue.t1_ms))

    if sequence_name == "Inversion Recovery":
        # Use TI from session state or fallback later by caller; here assume inversion handled outside.
        signal_pre_te = se_signal
    elif sequence_name == "Gradient Echo":
        signal_pre_te = gre_signal
    elif sequence_name == "EPI":
        signal_pre_te = gre_signal
    else:
        signal_pre_te = se_signal

    if mode == "T2*":
        t2_eff = compute_t2_star(tissue.t2_ms, field_inhomogeneity, sequence_name)
        return signal_pre_te * math.exp(-te_ms / t2_eff)
    return signal_pre_te * math.exp(-te_ms / tissue.t2_ms)



def inversion_recovery_signal(tr_ms: float, te_ms: float, ti_ms: float, tissue: Tissue) -> float:
    mz = 1 - 2 * math.exp(-ti_ms / tissue.t1_ms) + math.exp(-tr_ms / tissue.t1_ms)
    return tissue.proton_density * abs(mz) * math.exp(-te_ms / tissue.t2_ms)



def scan_time_seconds(tr_ms: float, ny: int, nsa: int, nz: int = 1, sequence_name: str = "Spin Echo") -> float:
    speed_factor = SEQUENCE_INFO[sequence_name]["speed_factor"]
    return tr_ms * ny * nsa * nz * speed_factor / 1000.0



def estimate_snr(signal: float, bandwidth_khz: float, field_strength_t: float) -> float:
    # Very simplified teaching model: SNR improves with field, worsens with higher bandwidth.
    return max(0.0, signal * field_strength_t * 50 / math.sqrt(max(bandwidth_khz, 0.1)))



def artifact_scores(
    motion_level: float,
    sequence_name: str,
    field_inhomogeneity: float,
    fov_cm: float,
    anatomy_width_cm: float,
    bandwidth_khz: float,
    fat_water_shift_hz: float,
    venc_cm_s: float,
    flow_cm_s: float,
) -> Dict[str, float]:
    seq_sense = SEQUENCE_INFO[sequence_name]["artifact_sensitivity"]

    motion = min(100, 100 * motion_level * seq_sense)
    susceptibility = min(100, 80 * field_inhomogeneity * seq_sense)
    aliasing = min(100, max(0.0, 100 * (anatomy_width_cm - fov_cm) / max(anatomy_width_cm, 1)))
    chemical_shift = min(100, fat_water_shift_hz / max(bandwidth_khz * 10, 1) * 100)

    if venc_cm_s <= 0:
        venc_alias = 100.0
    else:
        venc_alias = 0.0 if abs(flow_cm_s) <= venc_cm_s else min(100, 100 * (abs(flow_cm_s) - venc_cm_s) / venc_cm_s)

    return {
        "Motion": motion,
        "Susceptibility": susceptibility,
        "Aliasing": aliasing,
        "Chemical Shift": chemical_shift,
        "Flow/VENC Aliasing": venc_alias,
    }



def simulate_time_series(
    duration_s: int,
    sample_rate_hz: float,
    tissue: Tissue,
    sequence_name: str,
    tr_ms: float,
    te_ms: float,
    flip_angle_deg: float,
    field_inhomogeneity: float,
    base_heart_rate: float,
    base_resp_rate: float,
    motion_level: float,
    spo2: float,
    flow_cm_s: float,
    ti_ms: float,
    fmri_mode: bool,
    task_onsets: List[Tuple[int, int]],
) -> pd.DataFrame:
    t = np.arange(0, duration_s, 1 / sample_rate_hz)
    heart_hz = base_heart_rate / 60.0
    resp_hz = base_resp_rate / 60.0

    hr = base_heart_rate + 3 * np.sin(2 * np.pi * 0.03 * t)
    rr = base_resp_rate + 1.5 * np.sin(2 * np.pi * 0.02 * t + 1.1)
    spo2_t = spo2 - 0.2 * np.sin(2 * np.pi * 0.015 * t)
    motion = np.clip(motion_level + 0.08 * np.sin(2 * np.pi * resp_hz * t) + 0.03 * np.random.randn(len(t)), 0, 1)
    flow = flow_cm_s + 4 * np.sin(2 * np.pi * heart_hz * t)

    if sequence_name == "Inversion Recovery":
        sig0 = inversion_recovery_signal(tr_ms, te_ms, ti_ms, tissue)
    else:
        sig0 = base_signal(tr_ms, te_ms, tissue, sequence_name, flip_angle_deg, field_inhomogeneity)

    physiological_mod = (
        1
        - 0.10 * motion
        + 0.015 * np.sin(2 * np.pi * heart_hz * t)
        + 0.02 * np.sin(2 * np.pi * resp_hz * t)
        + 0.002 * (spo2_t - 98)
    )

    bold = np.zeros_like(t)
    task = np.zeros_like(t)
    if fmri_mode:
        for start, stop in task_onsets:
            mask = (t >= start) & (t < stop)
            task[mask] = 1

        # Simple hemodynamic response shape
        tau1, tau2 = 6.0, 12.0
        hrf_t = np.arange(0, 30, 1 / sample_rate_hz)
        hrf = (hrf_t ** 8.6) * np.exp(-hrf_t / tau1) - 0.35 * (hrf_t ** 10.0) * np.exp(-hrf_t / tau2)
        hrf = hrf / np.max(np.abs(hrf))
        bold = np.convolve(task, hrf, mode="same") * 0.03

    signal = sig0 * physiological_mod * (1 + bold)
    signal = np.clip(signal, 0, None)

    return pd.DataFrame(
        {
            "time_s": t,
            "signal": signal,
            "heart_rate_bpm": hr,
            "resp_rate_bpm": rr,
            "spo2_pct": spo2_t,
            "motion_index": motion,
            "flow_cm_s": flow,
            "task": task,
            "bold_effect": bold,
        }
    )



def create_phantom_grid(size: int = 96) -> Tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    r = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x)
    tissue_map = np.zeros((size, size), dtype=float)

    tissue_map[r < 0.85] = 0.55
    tissue_map[r < 0.68] = 0.70
    tissue_map[r < 0.45] = 0.82
    ventricles = ((x + 0.18) ** 2 / 0.06**2 + y**2 / 0.12**2 < 1) | ((x - 0.18) ** 2 / 0.06**2 + y**2 / 0.12**2 < 1)
    tissue_map[ventricles] = 1.0
    fat_ring = (r > 0.85) & (r < 0.95)
    tissue_map[fat_ring] = 0.92

    # Add directional structure
    tissue_map += 0.04 * np.cos(6 * angle) * (r < 0.65)
    return tissue_map, r



def apply_artifacts_to_image(
    img: np.ndarray,
    motion_score: float,
    aliasing_score: float,
    chemical_shift_score: float,
    susceptibility_score: float,
) -> np.ndarray:
    out = img.copy()

    # Motion ghosting: shifted weighted copies
    ghost_strength = motion_score / 100.0
    if ghost_strength > 0:
        out = (1 - 0.35 * ghost_strength) * out + 0.20 * ghost_strength * np.roll(out, 6, axis=1) + 0.15 * ghost_strength * np.roll(out, 12, axis=1)

    # Aliasing / wrap-around
    alias_strength = aliasing_score / 100.0
    if alias_strength > 0:
        out += 0.35 * alias_strength * np.roll(out, out.shape[1] // 2, axis=1)

    # Chemical shift: offset bright fat-like boundary
    chem_strength = chemical_shift_score / 100.0
    if chem_strength > 0:
        out += 0.15 * chem_strength * np.roll(out, int(2 + 5 * chem_strength), axis=1)

    # Susceptibility: local dropout near sinus-like region
    sus_strength = susceptibility_score / 100.0
    if sus_strength > 0:
        h, w = out.shape
        yy, xx = np.mgrid[0:h, 0:w]
        dropout = np.exp(-((xx - w * 0.5) ** 2 / (2 * (w * 0.08) ** 2) + (yy - h * 0.22) ** 2 / (2 * (h * 0.05) ** 2)))
        distortion = np.roll(out, int(4 * sus_strength), axis=1)
        out = out * (1 - 0.7 * sus_strength * dropout) + 0.15 * sus_strength * distortion * dropout

    return np.clip(out, 0, None)



def simple_kspace_reconstruction(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    kspace = np.fft.fftshift(np.fft.fft2(image))
    recon = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace)))
    return np.log1p(np.abs(kspace)), recon


# -----------------------------
# Sidebar controls
# -----------------------------
st.title("Interactive MRI Theory Simulation and Monitoring Platform")
st.caption("An interactive platform for exploring MRI physics, physiological effects, image artifacts, and acquisition tradeoffs in real time.")

with st.expander("Why this app matters", expanded=True):
    st.markdown(
        """
This simulator connects **MRI theory** with **real-time monitoring concepts** inspired by industrial analytics platforms such as Seeq.
Users can change scan settings, patient physiology, and environment conditions, then immediately observe how signal, scan time,
artifacts, and image behavior respond.
        """
    )

DEMO_SCENARIOS = {
    "Custom": None,
    "T1 vs T2 Weighting": {
        "patient_name": "Healthy Adult", "tissue_name": "Gray Matter", "sequence_name": "Spin Echo", "field_strength_t": 3.0,
        "tr_ms": 500, "te_ms": 20, "flip_angle_deg": 90, "ti_ms": 1800, "ny": 128, "nz": 1, "nsa": 1,
        "fov_cm": 22.0, "anatomy_width_cm": 24.0, "bandwidth_khz": 40.0, "venc_cm_s": 40, "field_inhomogeneity": 0.15,
        "heart_rate": 72, "resp_rate": 14, "spo2": 99, "motion_level": 0.08, "flow_cm_s": 25, "fmri_mode": False
    },
    "Motion Artifact": {
        "patient_name": "Anxious Patient", "tissue_name": "Gray Matter", "sequence_name": "EPI", "field_strength_t": 3.0,
        "tr_ms": 1500, "te_ms": 45, "flip_angle_deg": 75, "ti_ms": 1800, "ny": 128, "nz": 1, "nsa": 1,
        "fov_cm": 22.0, "anatomy_width_cm": 24.0, "bandwidth_khz": 50.0, "venc_cm_s": 40, "field_inhomogeneity": 0.35,
        "heart_rate": 105, "resp_rate": 22, "spo2": 98, "motion_level": 0.45, "flow_cm_s": 28, "fmri_mode": False
    },
    "Aliasing from Small FOV": {
        "patient_name": "Healthy Adult", "tissue_name": "Fat", "sequence_name": "Spin Echo", "field_strength_t": 1.5,
        "tr_ms": 900, "te_ms": 18, "flip_angle_deg": 90, "ti_ms": 1800, "ny": 96, "nz": 1, "nsa": 1,
        "fov_cm": 14.0, "anatomy_width_cm": 26.0, "bandwidth_khz": 20.0, "venc_cm_s": 40, "field_inhomogeneity": 0.12,
        "heart_rate": 72, "resp_rate": 14, "spo2": 99, "motion_level": 0.05, "flow_cm_s": 20, "fmri_mode": False
    },
    "fMRI Activation": {
        "patient_name": "Healthy Adult", "tissue_name": "Gray Matter", "sequence_name": "EPI", "field_strength_t": 3.0,
        "tr_ms": 2000, "te_ms": 32, "flip_angle_deg": 75, "ti_ms": 1800, "ny": 64, "nz": 1, "nsa": 1,
        "fov_cm": 22.0, "anatomy_width_cm": 22.0, "bandwidth_khz": 55.0, "venc_cm_s": 40, "field_inhomogeneity": 0.25,
        "heart_rate": 70, "resp_rate": 13, "spo2": 99, "motion_level": 0.06, "flow_cm_s": 24, "fmri_mode": True
    },
}

with st.sidebar:
    st.header("Scenario Setup")
    demo_name = st.selectbox("Demo mode", list(DEMO_SCENARIOS.keys()))
    demo_values = DEMO_SCENARIOS[demo_name] or {}

    patient_name = st.selectbox("Patient profile", list(PATIENTS.keys()), index=list(PATIENTS.keys()).index(demo_values.get("patient_name", "Healthy Adult")))
    tissue_name = st.selectbox("Primary tissue", list(TISSUES.keys()), index=list(TISSUES.keys()).index(demo_values.get("tissue_name", "Gray Matter")))
    sequence_name = st.selectbox("Sequence", list(SEQUENCE_INFO.keys()), index=list(SEQUENCE_INFO.keys()).index(demo_values.get("sequence_name", "Spin Echo")))
    field_strength_t = st.selectbox("Field strength (T)", [1.5, 3.0, 7.0], index=[1.5, 3.0, 7.0].index(demo_values.get("field_strength_t", 3.0)))

    st.header("Acquisition Parameters")
    tr_ms = st.slider("TR (ms)", 100, 8000, int(demo_values.get("tr_ms", 1200)), step=10)
    te_ms = st.slider("TE (ms)", 5, 300, int(demo_values.get("te_ms", 35)), step=1)
    flip_angle_deg = st.slider("Flip angle (deg)", 1, 180, int(demo_values.get("flip_angle_deg", 90)), step=1)
    ti_ms = st.slider("TI (ms) [for inversion recovery]", 50, 4000, int(demo_values.get("ti_ms", 1800)), step=10)
    ny = st.slider("Phase-encoding steps (Ny)", 32, 512, int(demo_values.get("ny", 128)), step=16)
    nz = st.slider("3D phase steps (Nz)", 1, 128, int(demo_values.get("nz", 1)), step=1)
    nsa = st.slider("Number of averages (NSA)", 1, 8, int(demo_values.get("nsa", 1)), step=1)

    st.header("FOV / Bandwidth / Flow")
    fov_cm = st.slider("Field of view (cm)", 10.0, 40.0, float(demo_values.get("fov_cm", 22.0)), step=0.5)
    anatomy_width_cm = st.slider("Patient/anatomy width in phase direction (cm)", 10.0, 50.0, float(demo_values.get("anatomy_width_cm", 24.0)), step=0.5)
    bandwidth_khz = st.slider("Receiver bandwidth (kHz)", 5.0, 125.0, float(demo_values.get("bandwidth_khz", 40.0)), step=1.0)
    venc_cm_s = st.slider("VENC (cm/s)", 5, 200, int(demo_values.get("venc_cm_s", 40)), step=1)
    field_inhomogeneity = st.slider("Field inhomogeneity", 0.0, 1.0, float(demo_values.get("field_inhomogeneity", 0.2)), step=0.01)

    st.header("Patient Monitoring")
    defaults = PATIENTS[patient_name]
    heart_rate = st.slider("Heart rate (bpm)", 40, 180, int(demo_values.get("heart_rate", defaults["heart_rate"])))
    resp_rate = st.slider("Respiration rate (breaths/min)", 6, 40, int(demo_values.get("resp_rate", defaults["resp_rate"])))
    spo2 = st.slider("SpO2 (%)", 80, 100, int(demo_values.get("spo2", defaults["spo2"])))
    motion_level = st.slider("Motion level", 0.0, 1.0, float(demo_values.get("motion_level", defaults["motion"])), step=0.01)
    flow_cm_s = st.slider("Flow velocity (cm/s)", 0, 150, int(demo_values.get("flow_cm_s", defaults["flow"])))

    st.header("Simulation Options")
    duration_s = st.slider("Monitoring duration (s)", 20, 180, 60, step=10)
    sample_rate_hz = st.slider("Sampling rate (Hz)", 1, 20, 5, step=1)
    fmri_mode = st.checkbox("Enable fMRI task mode", value=bool(demo_values.get("fmri_mode", False)))
    show_kspace = st.checkbox("Show k-space", value=True)


tissue = TISSUES[tissue_name]
fat_water_shift_hz = 220 if field_strength_t == 1.5 else 440 if field_strength_t == 3.0 else 1026

if sequence_name == "Inversion Recovery":
    signal_value = inversion_recovery_signal(tr_ms, te_ms, ti_ms, tissue)
else:
    signal_value = base_signal(tr_ms, te_ms, tissue, sequence_name, flip_angle_deg, field_inhomogeneity)

snr = estimate_snr(signal_value, bandwidth_khz, field_strength_t)
scan_time_s = scan_time_seconds(tr_ms, ny, nsa, nz, sequence_name)
optimal_flip = ernst_angle_deg(tr_ms, tissue.t1_ms)

scores = artifact_scores(
    motion_level=motion_level,
    sequence_name=sequence_name,
    field_inhomogeneity=field_inhomogeneity,
    fov_cm=fov_cm,
    anatomy_width_cm=anatomy_width_cm,
    bandwidth_khz=bandwidth_khz,
    fat_water_shift_hz=fat_water_shift_hz,
    venc_cm_s=venc_cm_s,
    flow_cm_s=flow_cm_s,
)


# -----------------------------
# Top metrics
# -----------------------------
image_quality_score = max(0, min(100, 100 - 0.35 * sum(scores.values()) + 0.8 * snr - 0.02 * scan_time_s))
metric_cols = st.columns(6)
metric_cols[0].metric("Estimated Signal", f"{signal_value:.3f}")
metric_cols[1].metric("Estimated SNR", f"{snr:.1f}")
metric_cols[2].metric("Scan Time", f"{scan_time_s:.1f} s")
metric_cols[3].metric("Ernst Angle", f"{optimal_flip:.1f}°")
metric_cols[4].metric("Dominant Artifact", max(scores, key=scores.get))
metric_cols[5].metric("Image Quality Score", f"{image_quality_score:.0f}/100")


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Signal + Monitoring",
    "Image + K-space",
    "Artifacts + Tradeoffs",
    "Teaching Notes",
])

with tab1:
    st.subheader("Continuous Monitoring Dashboard")

    task_onsets = [(10, 20), (35, 45)] if fmri_mode else []
    df = simulate_time_series(
        duration_s=duration_s,
        sample_rate_hz=sample_rate_hz,
        tissue=tissue,
        sequence_name=sequence_name,
        tr_ms=tr_ms,
        te_ms=te_ms,
        flip_angle_deg=flip_angle_deg,
        field_inhomogeneity=field_inhomogeneity,
        base_heart_rate=heart_rate,
        base_resp_rate=resp_rate,
        motion_level=motion_level,
        spo2=spo2,
        flow_cm_s=flow_cm_s,
        ti_ms=ti_ms,
        fmri_mode=fmri_mode,
        task_onsets=task_onsets,
    )

    selectable = {
        "Signal": "signal",
        "Heart Rate": "heart_rate_bpm",
        "Respiration": "resp_rate_bpm",
        "SpO2": "spo2_pct",
        "Motion": "motion_index",
        "Flow": "flow_cm_s",
    }
    if fmri_mode:
        selectable["Task"] = "task"
        selectable["BOLD Effect"] = "bold_effect"

    selected_series = st.multiselect(
        "Select attributes to trend",
        list(selectable.keys()),
        default=["Signal", "Heart Rate", "Motion"],
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    for name in selected_series:
        ax.plot(df["time_s"], df[selectable[name]], label=name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.set_title("MRI Monitoring Trends")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    left, right = st.columns([1.3, 1])
    with left:
        st.dataframe(df.round(4), use_container_width=True, height=320)
    with right:
        time_ms = np.linspace(0, max(5 * tissue.t1_ms, 3000), 400)
        fig_t1, ax_t1 = plt.subplots(figsize=(6, 3))
        ax_t1.plot(time_ms, longitudinal_recovery(time_ms, tissue.t1_ms))
        ax_t1.set_title("T1 Recovery")
        ax_t1.set_xlabel("Time (ms)")
        ax_t1.set_ylabel("Mz")
        ax_t1.grid(alpha=0.3)
        st.pyplot(fig_t1)

        time_ms2 = np.linspace(0, max(5 * tissue.t2_ms, 400), 400)
        t2_eff = compute_t2_star(tissue.t2_ms, field_inhomogeneity, sequence_name)
        fig_t2, ax_t2 = plt.subplots(figsize=(6, 3))
        ax_t2.plot(time_ms2, transverse_decay(time_ms2, tissue.t2_ms), label="T2")
        ax_t2.plot(time_ms2, transverse_decay(time_ms2, t2_eff), label="T2*")
        ax_t2.set_title("Transverse Decay")
        ax_t2.set_xlabel("Time (ms)")
        ax_t2.set_ylabel("Mxy")
        ax_t2.legend()
        ax_t2.grid(alpha=0.3)
        st.pyplot(fig_t2)

with tab2:
    st.subheader("Image Formation Prototype")
    base_img, r = create_phantom_grid(112)

    # Weighting: mimic tissue contrast changes.
    if sequence_name in ["Spin Echo", "EPI"] and te_ms > 60:
        weighted = base_img**1.4
    elif sequence_name == "Gradient Echo":
        weighted = base_img * np.exp(-0.3 * field_inhomogeneity * (1 - base_img))
    elif sequence_name == "Inversion Recovery":
        # Null-like effect: darker fluid if TI near CSF null
        csf_null = math.exp(-abs(ti_ms - 0.69 * TISSUES["CSF"].t1_ms) / 600)
        weighted = base_img.copy()
        weighted[base_img > 0.95] *= (1 - 0.9 * csf_null)
    else:
        weighted = base_img

    art_img = apply_artifacts_to_image(
        weighted,
        motion_score=scores["Motion"],
        aliasing_score=scores["Aliasing"],
        chemical_shift_score=scores["Chemical Shift"],
        susceptibility_score=scores["Susceptibility"],
    )

    if show_kspace:
        kspace_view, recon = simple_kspace_reconstruction(art_img)
        c1, c2, c3 = st.columns(3)
        with c1:
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            ax1.imshow(base_img, cmap="gray")
            ax1.set_title("Ideal Phantom")
            ax1.axis("off")
            st.pyplot(fig1)
        with c2:
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.imshow(kspace_view, cmap="gray")
            ax2.set_title("K-space (log magnitude)")
            ax2.axis("off")
            st.pyplot(fig2)
        with c3:
            fig3, ax3 = plt.subplots(figsize=(4, 4))
            ax3.imshow(recon, cmap="gray")
            ax3.set_title("Reconstructed Image")
            ax3.axis("off")
            st.pyplot(fig3)
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            ax1.imshow(base_img, cmap="gray")
            ax1.set_title("Ideal Phantom")
            ax1.axis("off")
            st.pyplot(fig1)
        with c2:
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.imshow(art_img, cmap="gray")
            ax2.set_title("Artifact-Affected Image")
            ax2.axis("off")
            st.pyplot(fig2)

with tab3:
    st.subheader("Artifacts and Acquisition Tradeoffs")

    left, right = st.columns([1, 1.2])
    with left:
        st.write("### Artifact Severity")
        score_df = pd.DataFrame({"Artifact": list(scores.keys()), "Severity": list(scores.values())})
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
        ax_bar.bar(score_df["Artifact"], score_df["Severity"])
        ax_bar.set_ylim(0, 100)
        ax_bar.set_ylabel("Severity (0-100)")
        ax_bar.set_xticklabels(score_df["Artifact"], rotation=30, ha="right")
        ax_bar.grid(axis="y", alpha=0.3)
        st.pyplot(fig_bar)

    with right:
        st.write("### Parameter Interpretation")
        st.markdown(
            f"""
**Sequence:** {sequence_name}  
**Primary tissue:** {tissue.name}  
**Field strength:** {field_strength_t} T  
**TR / TE:** {tr_ms} ms / {te_ms} ms  
**Bandwidth:** {bandwidth_khz:.1f} kHz  
**FOV:** {fov_cm:.1f} cm  
**Anatomy width:** {anatomy_width_cm:.1f} cm  
**Flow / VENC:** {flow_cm_s} cm/s / {venc_cm_s} cm/s
            """
        )

        st.write(f"### Suggested talking point for this scenario: {demo_name}")
        
        scenario_message = {
            "Custom": "Use this mode to build your own scan and explain how each variable changes MRI behavior.",
            "T1 vs T2 Weighting": "Use this scenario to show how TR and TE control tissue contrast and why parameter selection matters.",
            "Motion Artifact": "Use this scenario to demonstrate how patient motion and EPI sensitivity degrade image quality and monitoring stability.",
            "Aliasing from Small FOV": "Use this scenario to show how insufficient FOV causes wrap-around artifact and why acquisition planning matters.",
            "fMRI Activation": "Use this scenario to connect physiology, task timing, and BOLD-like signal changes in a dynamic acquisition.",
        }[demo_name]
        st.info(scenario_message)

        tradeoffs = []
        tradeoffs.append("Higher bandwidth lowers chemical shift artifact but also reduces SNR." if bandwidth_khz > 50 else "Lower bandwidth improves SNR but can worsen chemical shift artifact.")
        tradeoffs.append("FOV is smaller than anatomy width, so wrap-around aliasing risk is elevated." if fov_cm < anatomy_width_cm else "FOV covers anatomy, so phase wrap-around risk is reduced.")
        tradeoffs.append("Flow exceeds VENC, so phase wrapping is expected in phase-contrast flow imaging." if flow_cm_s > venc_cm_s else "VENC is high enough to avoid obvious phase aliasing.")
        tradeoffs.append("Field inhomogeneity is high enough that T2* decay and susceptibility artifact will be noticeable." if field_inhomogeneity > 0.35 else "Field homogeneity is relatively good, limiting T2* distortion.")
        tradeoffs.append("EPI is fast, but it is especially vulnerable to susceptibility and motion." if sequence_name == "EPI" else "This sequence emphasizes stability over extreme speed.")

        for item in tradeoffs:
            st.write(f"- {item}")

with tab4:
    st.subheader("Teaching Notes for Your Final Project")
    st.markdown(
        """
### What this prototype already demonstrates
- T1 recovery and T2 / T2* decay behavior
- Dependence of signal on TR, TE, flip angle, and tissue properties
- Scan-time tradeoffs using phase encoding steps and averages
- Physiologic monitoring with trend selection, inspired by industrial process dashboards
- Artifact mechanisms including motion, aliasing, chemical shift, susceptibility, and VENC mismatch
- Simple k-space visualization and image reconstruction workflow
- Optional fMRI-style task timing with a basic BOLD-like response model

### Best way to present this to your professor
Frame the app as a **teaching simulator**, not a diagnostic tool. The goal is to let students test how MRI parameters, patient physiology, and scanner conditions interact.

### Strong next upgrades
1. Add multiple tissue regions with independent T1/T2 values and contrast comparisons.
2. Add a proper diffusion-weighted imaging module with b-values and ADC behavior.
3. Add TOF and phase-contrast MRA tabs with explicit flow-direction visualization.
4. Export monitoring data to CSV.
5. Add a scenario-comparison mode so users can compare two scans side-by-side.
        """
    )

st.download_button(
    label="Download monitoring data as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="mri_simulation_monitoring_data.csv",
    mime="text/csv",
)
