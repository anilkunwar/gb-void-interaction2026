"""
UO2 Intergranular Void – Grain Boundary Interaction Phase-Field FFT Simulator
=========================================================================
Physics: MPF (multi-phase-field) + Cahn-Hilliard (c_v, c_i) + Allen-Cahn
         (eta, phi_i) + Anisotropic mobility (bulk + surface + GB tensors)
         + Cascade irradiation source terms + recombination + GB sinks
         + Spectral elastoplasticity (plane-strain, isotropic UO2)
         + J2 plasticity flow rule (overstress power law, Hall-Petch yield)
         + Larche-Cahn stress coupling into c_v / c_i chemical potentials.

This is a conversion of an Au-assisted Cu-Cu diffusion bonding MPF simulator
(rectangular MPF edition) into a UO2 void-GB irradiation model. The
conversion:
  * Removes the upper poly-Cu substrate and the Au nanofilm layer.
  * Replaces the nc-Cu substrate with polycrystalline UO2.
  * Replaces the Au concentration field `c` with two conserved concentration
    fields `c_v` (vacancy) and `c_i` (interstitial), governed by modified
    Cahn-Hilliard equations with cascade source terms, recombination, and
    GB sinks.
  * Adds the void order parameter eta (Allen-Cahn).
  * Replaces the per-grain eta_i list with grain-orientation order parameters
    phi_i (Allen-Cahn) plus an optional artificial driving force f_d for the
    flat-GB scenarios.
  * Implements the anisotropic mobility tensor decomposition with bulk,
    surface, and GB contributions.
  * Retains the spectral elastoplasticity solver (plane-strain) and the
    J2 plasticity flow rule from the original code, re-parameterised with
    UO2 elastoplastic constants (E=220 GPa, nu=0.31, sigma_y0=150 MPa,
    H=5 GPa) and feeds the hydrostatic stress back into the vacancy /
    interstitial chemical potentials via the Larche-Cahn coupling
    (Omega_v * sigma_h on c_v, Omega_i * sigma_h on c_i).
  * Supports three simulation scenarios:
      (1) Multiple voids + flat GB
      (2) GB migration under irradiation
      (3) Polycrystalline UO2 grain growth + void formation

Outputs (saved to /home/z/my-project/download/):
  * PNG field snapshots (multi-field grid: c_v, c_i, eta, grain map,
    sigma_eq, ...)
  * CSV time-series (porosity, mean c_v, GB area, max sigma_eq, plastic
    work, ...)
  * Pickled / npz-saved full frame history for later reload
  * MP4 / GIF animation of the eta (void) field
"""

import numpy as np
import streamlit as st
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import (Colormap, LinearSegmentedColormap,
                                ListedColormap, LogNorm)
from matplotlib.ticker import AutoMinorLocator
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
import plotly.graph_objects as go
from datetime import datetime
import json, zipfile, pickle, hashlib, traceback, warnings
from io import BytesIO, StringIO
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SENTINEL FOR DECORATED FAILURES
# ============================================================================
class _Sentinel:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.reason = None
        return cls._instance
    def __bool__(self): return False
    def __repr__(self): return f"<FAILED: {self.reason}>"

FAILED = _Sentinel()


# ============================================================================
# ERROR HANDLING
# ============================================================================
def handle_errors(func=None, *, re_raise=False):
    def deco(f):
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                FAILED.reason = f"{f.__name__}: {e}"
                logger.error(f"{e}\n{traceback.format_exc()}")
                try:
                    st.error(f"❌ Error in {f.__name__}: {e}")
                except Exception:
                    pass
                if re_raise:
                    raise
                return FAILED
        wrapper.__name__ = f.__name__
        wrapper.__doc__ = f.__doc__
        return wrapper

    if func is None:
        return deco
    return deco(func)


def safe_unpack(result, n, default=None):
    if result is FAILED or result is None:
        if default is None:
            return tuple([None] * n)
        return default
    return result


# ============================================================================
# RECTANGULAR SPECTRAL OPERATORS
# ============================================================================
def make_k_vectors(Nx, Ny, dx, dy=None):
    if dy is None:
        dy = dx
    kx = 2 * np.pi * fftfreq(Nx, d=dx).reshape(1, -1)
    ky = 2 * np.pi * fftfreq(Ny, d=dy).reshape(-1, 1)
    k2 = kx**2 + ky**2
    k2[0, 0] = 1e-12
    return kx, ky, k2


def spectral_gradients(field, kx, ky):
    fh = fft2(field)
    return (np.real(ifft2(1j * kx * fh)),
            np.real(ifft2(1j * ky * fh)))


def spectral_laplacian(field, k2):
    return np.real(ifft2(-k2 * fft2(field)))


def spectral_divergence(fx, fy, kx, ky):
    return np.real(ifft2(1j * kx * fft2(fx) + 1j * ky * fft2(fy)))


# ============================================================================
# INTERFACE PROFILING HELPERS
# ============================================================================
def _extract_interface_profile(interface_2d):
    a = np.asarray(interface_2d)
    if a.ndim == 0:
        return None
    if a.ndim == 1:
        return a
    if a.shape[0] > 1 and np.allclose(a[0, :], a[-1, :], atol=1e-9):
        return a[0, :]
    return np.mean(a, axis=0)


def _plot_interface(ax, x_coords, interface_2d, **line_kwargs):
    a = np.asarray(interface_2d)
    if a.ndim == 0 or (a.ndim == 1 and a.size == 1):
        ax.axhline(float(a), **line_kwargs)
        return
    profile = _extract_interface_profile(a)
    if profile is None:
        return
    ax.plot(x_coords, profile, **line_kwargs)


# ============================================================================
# COLORMAPS (UO2-specific palettes added)
# ============================================================================
def _build_custom_colormaps():
    """UO2 phase-field colormaps. The defect concentration fields (c_v, c_i)
    span many orders of magnitude and need a perceptually-uniform sequential
    map; the void order parameter η benefits from a high-contrast 'void'
    palette that highlights the matrix↔void transition."""
    cv_conc = LinearSegmentedColormap.from_list('cv_conc', [
        (0.00, '#0d1b2a'), (0.20, '#1b4f72'), (0.45, '#2e86c1'),
        (0.70, '#85c1e9'), (0.90, '#d4e6f1'), (1.00, '#fdfefe')])

    ci_conc = LinearSegmentedColormap.from_list('ci_conc', [
        (0.00, '#1a0033'), (0.25, '#4a148c'), (0.50, '#7b1fa2'),
        (0.75, '#ba68c8'), (0.90, '#e1bee7'), (1.00, '#fce4ec')])

    void_map = LinearSegmentedColormap.from_list('void_map', [
        (0.00, '#1b1b1b'), (0.30, '#3d2b1f'), (0.55, '#a0522d'),
        (0.75, '#deb887'), (0.90, '#ffe4b5'), (1.00, '#fffaf0')])

    porosity_map = LinearSegmentedColormap.from_list('porosity_map', [
        (0.00, '#ffffff'), (0.20, '#ffe4b5'), (0.50, '#deb887'),
        (0.75, '#cd853f'), (1.00, '#8b4513')])

    uo2_grain_map = ListedColormap([
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
        '#5254a3', '#8ca252', '#bd9e39', '#ad494a', '#a55194',
        '#cedb9c', '#9c9ede', '#e7969c', '#e7ba52', '#e7cb94',
        '#de9ed6', '#adc6dc', '#bd9e39', '#c49c94', '#f7b6d2',
        '#1a9850', '#fee08b', '#f46d43', '#d73027', '#762a83',
        '#9970ab', '#c2a5cf', '#e7d4e8', '#5aae61', '#a6dba0',
        '#009999', '#99ccff', '#663399', '#ff99cc', '#33cc33',
        '#ffcc00', '#cc6600', '#996633', '#663300', '#333333',
        '#003366', '#336699', '#6699cc', '#99cc66', '#cc9966',
    ], name='uo2_grain_map')

    stress_pub = LinearSegmentedColormap.from_list('stress_pub', [
        (0.0, '#2c7bb6'), (0.25, '#abd9e9'), (0.5, '#ffffbf'),
        (0.75, '#fdae61'), (1.0, '#d7191c')])

    gb_hot = LinearSegmentedColormap.from_list('gb_hot', [
        (0.0, '#000004'), (0.3, '#420a68'), (0.6, '#932667'),
        (0.85, '#f8850f'), (1.0, '#fcffa4')])

    return {'cv_conc': cv_conc, 'ci_conc': ci_conc, 'void_map': void_map,
            'porosity_map': porosity_map, 'uo2_grain_map': uo2_grain_map,
            'stress_pub': stress_pub, 'gb_hot': gb_hot}

_CUSTOM_COLORMAPS = _build_custom_colormaps()
for _n, _c in _CUSTOM_COLORMAPS.items():
    try:
        if _n not in mpl.colormaps:
            mpl.colormaps.register(_c, name=_n, force=True)
    except Exception as _e:
        logger.warning(f"Could not register colormap '{_n}': {_e}")


class ColormapRegistry:
    BASE = [
        'viridis', 'viridis_r', 'plasma', 'plasma_r',
        'inferno', 'inferno_r', 'magma', 'magma_r',
        'cividis', 'cividis_r', 'turbo', 'turbo_r',
        'Greys', 'Greys_r', 'Blues', 'Blues_r',
        'Greens', 'Greens_r', 'Reds', 'Reds_r',
        'Oranges', 'Oranges_r', 'Purples', 'Purples_r',
        'YlOrRd', 'YlOrRd_r', 'YlOrBr', 'YlOrBr_r',
        'YlGnBu', 'YlGnBu_r', 'YlGn', 'YlGn_r',
        'BuPu', 'BuPu_r', 'PuRd', 'PuRd_r',
        'BuGn', 'BuGn_r', 'PuBu', 'PuBu_r',
        'PuBuGn', 'PuBuGn_r', 'GnBu', 'GnBu_r',
        'OrRd', 'OrRd_r',
        'RdBu', 'RdBu_r', 'coolwarm', 'coolwarm_r',
        'bwr', 'bwr_r', 'seismic', 'seismic_r',
        'Spectral', 'Spectral_r',
        'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r',
        'BrBG', 'BrBG_r', 'PiYG', 'PiYG_r',
        'PRGn', 'PRGn_r', 'PuOr', 'PuOr_r',
        'RdGy', 'RdGy_r',
        'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r',
        'hsv', 'jet', 'jet_r', 'rainbow', 'rainbow_r',
        'nipy_spectral', 'nipy_spectral_r', 'gist_rainbow', 'gist_rainbow_r',
        'binary', 'binary_r', 'gray', 'gray_r',
        'bone', 'bone_r', 'copper', 'copper_r',
        'pink', 'pink_r', 'hot', 'hot_r',
        'afmhot', 'afmhot_r', 'gist_heat', 'gist_heat_r',
        'gist_gray', 'gist_gray_r', 'gist_yarg', 'gist_yarg_r',
        'flag', 'prism', 'ocean', 'ocean_r',
        'terrain', 'terrain_r', 'gist_earth', 'gist_earth_r',
        'gist_ncar', 'gist_ncar_r', 'gist_stern', 'gist_stern_r',
        'CMRmap', 'CMRmap_r', 'cubehelix', 'cubehelix_r',
        'brg', 'brg_r', 'cool', 'cool_r', 'warm', 'warm_r',
        'Wistia', 'Wistia_r',
        'spring', 'spring_r', 'summer', 'summer_r',
        'autumn', 'autumn_r', 'winter', 'winter_r',
        'Pastel1', 'Pastel2', 'Paired', 'Accent',
        'Dark2', 'Set1', 'Set2', 'Set3',
        'tab10', 'tab20', 'tab20b', 'tab20c',
    ]
    @classmethod
    def list_all(cls):
        names = list(cls.BASE) + list(_CUSTOM_COLORMAPS.keys())
        seen, out = set(), []
        for n in names:
            if n not in seen:
                seen.add(n); out.append(n)
        return out

    @classmethod
    def resolve(cls, name, invert=False):
        if name is None:
            cmap = mpl.colormaps['viridis']
        elif isinstance(name, Colormap):
            cmap = name
        else:
            key = str(name)
            if key in mpl.colormaps:
                cmap = mpl.colormaps[key]
            elif key in _CUSTOM_COLORMAPS:
                cmap = _CUSTOM_COLORMAPS[key]
            else:
                logger.warning(f"Unknown colormap '{name}', using viridis")
                cmap = mpl.colormaps['viridis']
        if invert:
            cmap = cmap.reversed()
        return cmap

    @classmethod
    def to_plotly(cls, name, n=64, invert=False):
        cmap = cls.resolve(name, invert=invert)
        t = np.linspace(0, 1, n)
        rgb = (cmap(t)[:, :3] * 255).astype(int)
        return [[float(ti), f'rgb({r},{g},{b})'] for ti, (r, g, b) in zip(t, rgb)]


# ============================================================================
# PUBLICATION THEMES
# ============================================================================
MATPLOTLIB_THEMES = {
    'default':                None,
    'ggplot':                 'ggplot',
    'seaborn-whitegrid':      'seaborn-v0_8-whitegrid',
    'seaborn-darkgrid':       'seaborn-v0_8-darkgrid',
    'seaborn-ticks':          'seaborn-v0_8-ticks',
    'seaborn-paper':          'seaborn-v0_8-paper',
    'seaborn-talk':           'seaborn-v0_8-talk',
    'bmh':                    'bmh',
    'dark_background':        'dark_background',
    'classic':                'classic',
    'fivethirtyeight':        'fivethirtyeight',
    'Solarize_Light2':        'Solarize_Light2',
    'tableau-colorblind10':   'tableau-colorblind10',
    'nature-like (custom)':   '__nature__',
    'science-like (custom)':  '__science__',
}

_NATURE_RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'legend.frameon': False,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}

_SCIENCE_RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.linewidth': 1.0,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
}


def apply_matplotlib_theme(theme_name: str, cfg: dict):
    plt.rcdefaults()
    style = MATPLOTLIB_THEMES.get(theme_name, None)
    if style == '__nature__':
        mpl.rcParams.update(_NATURE_RC)
    elif style == '__science__':
        mpl.rcParams.update(_SCIENCE_RC)
    elif style is not None:
        try:
            plt.style.use(style)
        except Exception as e:
            logger.warning(f"Theme '{style}' unavailable: {e}")
            plt.rcdefaults()
    mpl.rcParams['axes.titlesize']  = cfg.get('title_font_size', 11)
    mpl.rcParams['axes.labelsize']  = cfg.get('label_font_size', 9)
    mpl.rcParams['xtick.labelsize'] = cfg.get('tick_font_size', 8)
    mpl.rcParams['ytick.labelsize'] = cfg.get('tick_font_size', 8)
    mpl.rcParams['legend.fontsize'] = cfg.get('legend_font_size', 8)
    mpl.rcParams['figure.dpi']      = cfg.get('dpi', 150)
    mpl.rcParams['savefig.dpi']     = cfg.get('dpi', 150)


# ============================================================================
# FIELD CONFIGURATION (UO2)
# ============================================================================
# Fields tracked by the UO2 phase-field solver and made available to the
# visualizer. Note that 'grain_map' is derived as argmax(φ_i) and 'phi_gb'
# is the GB indicator 1 - Σφ_i² (Φ = 1 inside grain, < 1 at GBs).
FIELD_DEFAULT_CMAPS = {
    'c_v':         'cv_conc',
    'c_i':         'ci_conc',
    'eta':         'void_map',
    'phi_grain':   'uo2_grain_map',
    'phi_gb':      'gb_hot',
    'porosity':    'porosity_map',
    'D_eff_v':     'plasma',
    'D_eff_i':     'magma',
    'A_recomb':    'YlOrRd',
    'sigma_eq':    'stress_pub',
    'sigma_h':     'RdBu_r',
    'h':           'gray',
    'eps_p_mag':   'YlOrRd',
    'sigma_y':     'viridis',
}

FIELD_LABELS = {
    'c_v':         'c_v (vacancy concentration)',
    'c_i':         'c_i (interstitial concentration)',
    'eta':         'η (void order parameter)',
    'phi_grain':   'Grain ID map (argmax φ_i)',
    'phi_gb':      'φ_gb (1 − Σφ_i², GB indicator)',
    'porosity':    'Porosity (η²)',
    'D_eff_v':     'D_eff_v (vacancy diffusivity)',
    'D_eff_i':     'D_eff_i (interstitial diffusivity)',
    'A_recomb':    'A_iv (recombination rate)',
    'sigma_eq':    'σ_eq (Von Mises)',
    'sigma_h':     'σ_h (Hydrostatic)',
    'h':           'Local grain size (nm)',
    'eps_p_mag':   '|ε_p| (Equivalent plastic strain)',
    'sigma_y':     'σ_y (Yield stress)',
}

FIELD_UNITS = {
    'c_v': '', 'c_i': '', 'eta': '', 'phi_grain': '', 'phi_gb': '',
    'porosity': '', 'D_eff_v': '', 'D_eff_i': '', 'A_recomb': '',
    'sigma_eq': 'GPa', 'sigma_h': 'GPa', 'h': 'nm',
    'eps_p_mag': '', 'sigma_y': 'MPa',
}

FIELD_DEFAULT_RNG = {
    'c_v':         (1e-6, 1.0),
    'c_i':         (1e-9, 1.0),
    'eta':         (0.0, 1.0),
    'phi_gb':      (0.0, 1.0),
    'porosity':    (0.0, 1.0),
    'eps_p_mag':   (0.0, 0.05),
    'h':           (0.0, 200.0),
}

DISCRETE_FIELDS = {'phi_grain', 'phi_gb'}

VECTOR_FIELD_SOURCES = ['c_v', 'c_i', 'eta', 'sigma_h', 'sigma_eq',
                        'D_eff_v', 'D_eff_i', 'phi_gb']


# ============================================================================
# TIME-UNIT HANDLING (physics-preserving, from original code)
# ============================================================================
TIME_UNITS = {
    'ns': {'label': 'ns', 'long': 'nanoseconds'},
    'μs': {'label': 'μs', 'long': 'microseconds'},
    'ms': {'label': 'ms', 'long': 'milliseconds'},
    's':  {'label': 's',  'long': 'seconds'},
    'τ (reduced)': {'label': 'τ',  'long': 'reduced units (t* = 0.01 s)'},
}

DEFAULT_TIME_UNIT = 'τ (reduced)'


def get_time_unit_label(params):
    if params is None:
        return DEFAULT_TIME_UNIT
    return params.get('time_unit', DEFAULT_TIME_UNIT)


def get_time_scale(params):
    return 1.0


def fmt_time(t_code, params, fmt='%.4f'):
    unit = get_time_unit_label(params)
    return f"{fmt % float(t_code)} {unit}"


def fmt_time_range(t_code_arr, params):
    unit = get_time_unit_label(params)
    return np.asarray(t_code_arr, dtype=float), unit


# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
def default_field_settings():
    return {
        f: {'auto': True, 'vmin': None, 'vmax': None,
            'log': False, 'invert': False,
            'cmap': FIELD_DEFAULT_CMAPS.get(f, 'viridis')}
        for f in FIELD_DEFAULT_CMAPS.keys()
    }


def default_figure_customization():
    return {
        'multi_field_width':    13.5,
        'multi_field_height':   11.0,
        'overlay_width':        11.0,
        'overlay_height':        5.5,
        'vector_width':         11.0,
        'vector_height':         5.5,
        'diagnostics_width':    13.0,
        'diagnostics_height':    8.0,
        'line_profile_width':   11.0,
        'line_profile_height':   4.5,
        'line_width':            2.0,
        'line_width_minor':      1.5,
        'marker_size':           6,
        'marker_width':          1.0,
        'contour_line_width':    1.0,
        'quiver_width':          0.0025,
        'quiver_scale':          30.0,
        'line_style':            '-',
        'line_style_minor':      '--',
        'marker_style':          'none',
        'marker_style_minor':    'none',
        'color_series_1':        '#1f77b4',
        'color_series_2':        '#d62728',
        'color_series_3':        '#2ca02c',
        'color_series_4':        '#2ca02c',
        'color_series_5':        '#000000',
        'title_override':       '',
        'xlabel_override':      '',
        'ylabel_override':      '',
        'title_font_size':      None,
        'label_font_size':      None,
        'tick_font_size':       None,
        'legend_font_size':     None,
        'show_grid':            True,
        'grid_alpha':           0.3,
        'grid_linestyle':       '-',
        'grid_color':           '#cccccc',
        'xlim':                 (None, None),
        'ylim':                 (None, None),
        'show_colorbar':        True,
        'colorbar_orientation': 'vertical',
        'spine_width':          1.0,
        'tick_direction':       'out',
        'tick_width':           1.0,
        'tick_length':          4,
        'spine_top':            True,
        'spine_right':          True,
        'spine_left':           True,
        'spine_bottom':         True,
        'show_legend':          True,
        'legend_position':      'best',
        'legend_frameon':       True,
        'legend_framealpha':    1.0,
        'axis_equal':           True,
        'tight_layout':         True,
    }


def default_viz_config():
    return {
        'theme': 'default',
        'title_font_size': 11,
        'label_font_size': 9,
        'tick_font_size': 8,
        'legend_font_size': 8,
        'dpi': 150,
        'colorbar_fraction': 0.046,
        'colorbar_pad': 0.04,
        'field_settings': default_field_settings(),
        'figure_customization': default_figure_customization(),
        'overlay': {
            'enabled': False,
            'base': 'eta',
            'overlay': 'phi_gb',
            'alpha': 0.5,
            'mode': 'contour',
            'n_contours': 10,
            'show_grain_ghost': False,
        },
        'vector': {
            'enabled': False,
            'source': 'c_v',
            'mode': 'flux',
            'density': 24,
            'color': 'white',
            'scale': 1.0,
        },
        'animation': {
            'format': 'gif',
            'fps': 5,
            'dpi': 120,
            'watermark': True,
            'watermark_text': '',
            'watermark_pos': 'lower right',
        },
    }


def get_viz_config():
    if 'viz_cfg' not in st.session_state:
        st.session_state.viz_cfg = default_viz_config()
    cfg = st.session_state.viz_cfg
    defaults = default_viz_config()
    if 'figure_customization' not in cfg:
        cfg['figure_customization'] = defaults['figure_customization']
    else:
        for k, v in defaults['figure_customization'].items():
            cfg['figure_customization'].setdefault(k, v)
    for fname, fset in cfg['field_settings'].items():
        fset.setdefault('cmap', FIELD_DEFAULT_CMAPS.get(fname, 'viridis'))
    return cfg


def apply_axis_styling(ax, fig_cfg, *, title=None, xlabel=None, ylabel=None,
                       title_fs=None, label_fs=None, grid_axes=None,
                       force_aspect=None):
    if fig_cfg is None or ax is None:
        return
    if title_fs is None:
        title_fs = fig_cfg.get('title_font_size')
    if label_fs is None:
        label_fs = fig_cfg.get('label_font_size')
    tick_fs = fig_cfg.get('tick_font_size')
    if title is not None:
        override = fig_cfg.get('title_override', '').strip()
        ax.set_title(override if override else title,
                     fontsize=title_fs)
    if xlabel is not None:
        override = fig_cfg.get('xlabel_override', '').strip()
        ax.set_xlabel(override if override else xlabel,
                      fontsize=label_fs)
    if ylabel is not None:
        override = fig_cfg.get('ylabel_override', '').strip()
        ax.set_ylabel(override if override else ylabel,
                      fontsize=label_fs)
    if tick_fs is not None:
        for label_artist in ax.get_xticklabels() + ax.get_yticklabels():
            label_artist.set_fontsize(float(tick_fs))
    target = grid_axes if grid_axes is not None else ax
    if fig_cfg.get('show_grid', True):
        target.grid(True,
                    alpha=float(fig_cfg.get('grid_alpha', 0.3)),
                    linestyle=fig_cfg.get('grid_linestyle', '-'),
                    color=fig_cfg.get('grid_color', '#cccccc'))
    else:
        target.grid(False)
    xlim = fig_cfg.get('xlim', (None, None))
    ylim = fig_cfg.get('ylim', (None, None))
    if xlim and (xlim[0] is not None or xlim[1] is not None):
        ax.set_xlim(left=xlim[0], right=xlim[1])
    if ylim and (ylim[0] is not None or ylim[1] is not None):
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    sw = float(fig_cfg.get('spine_width', 1.0))
    spine_visible = {
        'top':    fig_cfg.get('spine_top',    True),
        'right':  fig_cfg.get('spine_right',  True),
        'left':   fig_cfg.get('spine_left',   True),
        'bottom': fig_cfg.get('spine_bottom', True),
    }
    for spine_name, spine in ax.spines.items():
        spine.set_linewidth(sw)
        spine.set_visible(spine_visible.get(spine_name, True))
    ax.tick_params(width=float(fig_cfg.get('tick_width', 1.0)),
                   length=float(fig_cfg.get('tick_length', 4)),
                   direction=fig_cfg.get('tick_direction', 'out'))
    if force_aspect is not None:
        aspect_mode = force_aspect
    elif fig_cfg.get('axis_equal', True):
        aspect_mode = 'equal'
    else:
        aspect_mode = 'auto'
    try:
        if aspect_mode == 'equal':
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.set_aspect('auto')
    except Exception:
        pass


def apply_legend_styling(ax, fig_cfg):
    if fig_cfg is None or ax is None:
        return
    if not fig_cfg.get('show_legend', True):
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        return
    leg_pos = fig_cfg.get('legend_position', 'best')
    legend_fs = fig_cfg.get('legend_font_size')
    frameon = bool(fig_cfg.get('legend_frameon', True))
    framealpha = float(fig_cfg.get('legend_framealpha', 1.0))
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    legend = ax.legend(
        handles, labels,
        loc=leg_pos,
        frameon=frameon,
        framealpha=framealpha,
        fontsize=legend_fs)
    return legend


# ============================================================================
# INLINE FIGURE CUSTOMIZATION HELPERS
# ============================================================================
LINE_STYLES = ['-', '--', '-.', ':', 'none', '']
MARKER_STYLES = ['none', 'o', 's', '^', 'v', '<', '>', 'D', 'd', '+', 'x', '*', '.', ',', '1', '2', '3', '4', '_']
LEGEND_POSITIONS = ['best', 'upper right', 'upper left', 'lower right', 'lower left',
                   'center', 'center left', 'center right', 'upper center', 'lower center',
                   'outside upper', 'outside right']
TICK_DIRECTIONS = ['out', 'in', 'inout']
GRID_LINESTYLES = ['-', '--', ':', '-.', '']


def _clamped_number_input(label, value, min_value, max_value, step=0.5,
                            key=None, help=None, format=None):
    if value is None:
        value = min_value
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float(min_value)
    if value < min_value:
        value = float(min_value)
    elif value > max_value:
        value = float(max_value)
    kwargs = dict(value=value, min_value=min_value, max_value=max_value, step=step)
    if key is not None:
        kwargs['key'] = key
    if help is not None:
        kwargs['help'] = help
    if format is not None:
        kwargs['format'] = format
    return st.number_input(label, **kwargs)


def render_inline_figure_customization(viz_cfg, *,
                                        size_keys=None,
                                        key_prefix='inl',
                                        show_axis_limits=True,
                                        show_text_overrides=True,
                                        show_grid=True,
                                        show_colorbar=True,
                                        show_lines=True,
                                        label='Figure customization'):
    fc = viz_cfg.setdefault('figure_customization', default_figure_customization())
    changed = False
    default_expanded = bool(st.session_state.get(f'{key_prefix}_exp', False))

    with st.expander(f"🎨 {label}", expanded=default_expanded):
        st.session_state[f'{key_prefix}_exp'] = st.checkbox(
            "Keep expanded", value=default_expanded,
            key=f"{key_prefix}_keep_exp",
            help="If checked, this panel stays open across reruns.")

        if size_keys is not None and len(size_keys) == 2:
            w_key, h_key = size_keys
            c1, c2 = st.columns(2)
            with c1:
                new_w = _clamped_number_input(
                    "Width (in)", value=fc.get(w_key, 11.0),
                    min_value=3.0, max_value=30.0, step=0.5,
                    key=f"{key_prefix}_w",
                    help="Figure width in inches.")
            with c2:
                new_h = _clamped_number_input(
                    "Height (in)", value=fc.get(h_key, 5.5),
                    min_value=2.0, max_value=30.0, step=0.5,
                    key=f"{key_prefix}_h")
            fc[w_key] = float(new_w); changed = True
            fc[h_key] = float(new_h); changed = True

        if show_text_overrides:
            st.markdown("**Text overrides** (blank = default)")
            c1, c2, c3 = st.columns(3)
            with c1:
                t_ov = st.text_input("Title", value=fc.get('title_override', ''),
                                      key=f"{key_prefix}_ti", label_visibility="collapsed",
                                      placeholder="Title override")
            with c2:
                x_ov = st.text_input("X label", value=fc.get('xlabel_override', ''),
                                      key=f"{key_prefix}_xl", label_visibility="collapsed",
                                      placeholder="X-axis label")
            with c3:
                y_ov = st.text_input("Y label", value=fc.get('ylabel_override', ''),
                                      key=f"{key_prefix}_yl", label_visibility="collapsed",
                                      placeholder="Y-axis label")
            if t_ov != fc.get('title_override', ''):
                fc['title_override'] = t_ov; changed = True
            if x_ov != fc.get('xlabel_override', ''):
                fc['xlabel_override'] = x_ov; changed = True
            if y_ov != fc.get('ylabel_override', ''):
                fc['ylabel_override'] = y_ov; changed = True

        if show_grid:
            c1, c2 = st.columns(2)
            with c1:
                new_grid = st.checkbox("Show grid",
                                        value=bool(fc.get('show_grid', True)),
                                        key=f"{key_prefix}_grid")
            with c2:
                new_ga = st.slider("Grid α", 0.0, 1.0,
                                    float(fc.get('grid_alpha', 0.3)),
                                    0.05, key=f"{key_prefix}_ga")
            if new_grid != bool(fc.get('show_grid', True)):
                fc['show_grid'] = new_grid; changed = True
            if new_ga != float(fc.get('grid_alpha', 0.3)):
                fc['grid_alpha'] = new_ga; changed = True

        if show_axis_limits:
            xlim_cur = fc.get('xlim', (None, None))
            ylim_cur = fc.get('ylim', (None, None))
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                xmin_in = st.text_input(
                    "xmin (blank=auto)",
                    value='' if xlim_cur[0] is None else str(xlim_cur[0]),
                    key=f"{key_prefix}_xmin")
            with c2:
                xmax_in = st.text_input(
                    "xmax (blank=auto)",
                    value='' if xlim_cur[1] is None else str(xlim_cur[1]),
                    key=f"{key_prefix}_xmax")
            with c3:
                ymin_in = st.text_input(
                    "ymin (blank=auto)",
                    value='' if ylim_cur[0] is None else str(ylim_cur[0]),
                    key=f"{key_prefix}_ymin")
            with c4:
                ymax_in = st.text_input(
                    "ymax (blank=auto)",
                    value='' if ylim_cur[1] is None else str(ylim_cur[1]),
                    key=f"{key_prefix}_ymax")
            def _parse(s):
                s = s.strip() if isinstance(s, str) else s
                if s is None or s == '':
                    return None
                try:
                    return float(s)
                except Exception:
                    return None
            new_xlim = (_parse(xmin_in), _parse(xmax_in))
            new_ylim = (_parse(ymin_in), _parse(ymax_in))
            if new_xlim != tuple(xlim_cur):
                fc['xlim'] = new_xlim; changed = True
            if new_ylim != tuple(ylim_cur):
                fc['ylim'] = new_ylim; changed = True

        if show_colorbar:
            c1, c2 = st.columns(2)
            with c1:
                new_cb = st.checkbox("Show colorbar",
                                      value=bool(fc.get('show_colorbar', True)),
                                      key=f"{key_prefix}_cb")
            with c2:
                cb_o_idx = 0 if fc.get('colorbar_orientation', 'vertical') == 'vertical' else 1
                new_cbo = st.selectbox("Colorbar orient.",
                                         ['vertical', 'horizontal'],
                                         index=cb_o_idx,
                                         key=f"{key_prefix}_cbo")
            if new_cb != bool(fc.get('show_colorbar', True)):
                fc['show_colorbar'] = new_cb; changed = True
            if new_cbo != fc.get('colorbar_orientation', 'vertical'):
                fc['colorbar_orientation'] = new_cbo; changed = True

        if show_lines:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                new_lw = st.number_input(
                    "Line width", value=float(fc.get('line_width', 2.0)),
                    min_value=0.2, max_value=10.0, step=0.1,
                    key=f"{key_prefix}_lw")
            with c2:
                new_lwm = st.number_input(
                    "Minor lw", value=float(fc.get('line_width_minor', 1.5)),
                    min_value=0.1, max_value=8.0, step=0.1,
                    key=f"{key_prefix}_lwm")
            with c3:
                _ms_val = fc.get('marker_size', 6)
                try:
                    _ms_val = int(_ms_val) if _ms_val is not None else 6
                except (TypeError, ValueError):
                    _ms_val = 6
                new_ms = st.number_input(
                    "Marker size", value=_ms_val,
                    min_value=0, max_value=30, step=1,
                    key=f"{key_prefix}_ms")
            with c4:
                new_cw = st.number_input(
                    "Contour lw", value=float(fc.get('contour_line_width', 1.0)),
                    min_value=0.1, max_value=5.0, step=0.1,
                    key=f"{key_prefix}_cw")
            if new_lw != float(fc.get('line_width', 2.0)):
                fc['line_width'] = new_lw; changed = True
            if new_lwm != float(fc.get('line_width_minor', 1.5)):
                fc['line_width_minor'] = new_lwm; changed = True
            if new_ms != fc.get('marker_size', 6):
                fc['marker_size'] = new_ms; changed = True
            if new_cw != float(fc.get('contour_line_width', 1.0)):
                fc['contour_line_width'] = new_cw; changed = True

        c1, c2, c3 = st.columns(3)
        with c1:
            new_sw = st.number_input(
                "Spine w", value=float(fc.get('spine_width', 1.0)),
                min_value=0.1, max_value=5.0, step=0.1,
                key=f"{key_prefix}_sw")
        with c2:
            td_idx = ['out', 'in', 'inout'].index(
                fc.get('tick_direction', 'out'))
            new_td = st.selectbox(
                "Tick dir", ['out', 'in', 'inout'],
                index=td_idx, key=f"{key_prefix}_td")
        with c3:
            _tl_val = fc.get('tick_length', 4)
            try:
                _tl_val = int(_tl_val) if _tl_val is not None else 4
            except (TypeError, ValueError):
                _tl_val = 4
            new_tl = st.number_input(
                "Tick len", value=_tl_val,
                min_value=0, max_value=20, step=1,
                key=f"{key_prefix}_tl")
        if new_sw != float(fc.get('spine_width', 1.0)):
            fc['spine_width'] = new_sw; changed = True
        if new_td != fc.get('tick_direction', 'out'):
            fc['tick_direction'] = new_td; changed = True
        if new_tl != fc.get('tick_length', 4):
            fc['tick_length'] = new_tl; changed = True

        if changed:
            st.caption("✏️ Applied live.")
        else:
            st.caption("Tweak any value — changes apply on next figure render.")

    return changed


def render_enhanced_figure_customization(viz_cfg, *,
                                          size_keys=None,
                                          key_prefix='enh',
                                          show_line_styles=True,
                                          show_series_colors=False,
                                          show_legend=True,
                                          show_spines=True,
                                          show_font_sizes=True,
                                          show_colorbar=True,
                                          show_axis_limits=True,
                                          show_text_overrides=True,
                                          show_grid=True,
                                          n_series=5,
                                          label='Figure customization'):
    fc = viz_cfg.setdefault('figure_customization', default_figure_customization())
    default_expanded = bool(st.session_state.get(f'{key_prefix}_exp', False))

    with st.expander(f"🎨 {label}", expanded=default_expanded):
        st.session_state[f'{key_prefix}_exp'] = st.checkbox(
            "Keep expanded", value=default_expanded,
            key=f"{key_prefix}_keep_exp",
            help="If checked, this panel stays open across reruns.")

        if size_keys is not None and len(size_keys) == 2:
            w_key, h_key = size_keys
            st.markdown("**📐 Figure size (inches)**")
            c1, c2 = st.columns(2)
            with c1:
                new_w = _clamped_number_input(
                    "Width", value=fc.get(w_key, 11.0),
                    min_value=3.0, max_value=30.0, step=0.5,
                    key=f"{key_prefix}_w")
            with c2:
                new_h = _clamped_number_input(
                    "Height", value=fc.get(h_key, 5.5),
                    min_value=2.0, max_value=30.0, step=0.5,
                    key=f"{key_prefix}_h")
            fc[w_key] = float(new_w)
            fc[h_key] = float(new_h)

        if show_font_sizes:
            st.markdown("**🔤 Font sizes** (blank = theme default)")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                cur = fc.get('title_font_size')
                new_tfs = st.number_input(
                    "Title", value=float(cur) if cur is not None else 0.0,
                    min_value=0.0, max_value=30.0, step=0.5,
                    key=f"{key_prefix}_tfs",
                    help="0 = use theme default")
                fc['title_font_size'] = float(new_tfs) if (new_tfs is not None and new_tfs > 0) else None
            with c2:
                cur = fc.get('label_font_size')
                new_lfs = st.number_input(
                    "Labels", value=float(cur) if cur is not None else 0.0,
                    min_value=0.0, max_value=24.0, step=0.5,
                    key=f"{key_prefix}_lfs",
                    help="0 = use theme default")
                fc['label_font_size'] = float(new_lfs) if (new_lfs is not None and new_lfs > 0) else None
            with c3:
                cur = fc.get('tick_font_size')
                new_tkfs = st.number_input(
                    "Ticks", value=float(cur) if cur is not None else 0.0,
                    min_value=0.0, max_value=20.0, step=0.5,
                    key=f"{key_prefix}_tkfs",
                    help="0 = use theme default")
                fc['tick_font_size'] = float(new_tkfs) if (new_tkfs is not None and new_tkfs > 0) else None
            with c4:
                cur = fc.get('legend_font_size')
                new_lgfs = st.number_input(
                    "Legend", value=float(cur) if cur is not None else 0.0,
                    min_value=0.0, max_value=20.0, step=0.5,
                    key=f"{key_prefix}_lgfs",
                    help="0 = use theme default")
                fc['legend_font_size'] = float(new_lgfs) if (new_lgfs is not None and new_lgfs > 0) else None

        if show_text_overrides:
            st.markdown("**🏷️ Text overrides** (blank = default)")
            c1, c2, c3 = st.columns(3)
            with c1:
                fc['title_override'] = st.text_input(
                    "Title", value=fc.get('title_override', ''),
                    key=f"{key_prefix}_ti", placeholder="Title override")
            with c2:
                fc['xlabel_override'] = st.text_input(
                    "X label", value=fc.get('xlabel_override', ''),
                    key=f"{key_prefix}_xl", placeholder="X-axis label")
            with c3:
                fc['ylabel_override'] = st.text_input(
                    "Y label", value=fc.get('ylabel_override', ''),
                    key=f"{key_prefix}_yl", placeholder="Y-axis label")

        if show_line_styles:
            st.markdown("**✏️ Line styles & markers**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                ls_idx = LINE_STYLES.index(fc.get('line_style', '-')) \
                    if fc.get('line_style', '-') in LINE_STYLES else 0
                fc['line_style'] = st.selectbox(
                    "Line style (main)", LINE_STYLES, index=ls_idx,
                    key=f"{key_prefix}_ls",
                    format_func=lambda x: {'-': '— solid', '--': '–– dashed',
                                             '-.': '–· dash-dot', ':': '·· dotted',
                                             'none': 'none', '': 'none'}.get(x, x))
            with c2:
                lsm_idx = LINE_STYLES.index(fc.get('line_style_minor', '--')) \
                    if fc.get('line_style_minor', '--') in LINE_STYLES else 1
                fc['line_style_minor'] = st.selectbox(
                    "Line style (minor)", LINE_STYLES, index=lsm_idx,
                    key=f"{key_prefix}_lsm",
                    format_func=lambda x: {'-': '— solid', '--': '–– dashed',
                                             '-.': '–· dash-dot', ':': '·· dotted',
                                             'none': 'none', '': 'none'}.get(x, x))
            with c3:
                ms_idx = MARKER_STYLES.index(fc.get('marker_style', 'none')) \
                    if fc.get('marker_style', 'none') in MARKER_STYLES else 0
                fc['marker_style'] = st.selectbox(
                    "Marker (main)", MARKER_STYLES, index=ms_idx,
                    key=f"{key_prefix}_ms")
            with c4:
                msm_idx = MARKER_STYLES.index(fc.get('marker_style_minor', 'none')) \
                    if fc.get('marker_style_minor', 'none') in MARKER_STYLES else 0
                fc['marker_style_minor'] = st.selectbox(
                    "Marker (minor)", MARKER_STYLES, index=msm_idx,
                    key=f"{key_prefix}_msm")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                fc['line_width'] = st.number_input(
                    "Line width (main)", value=float(fc.get('line_width', 2.0)),
                    min_value=0.2, max_value=10.0, step=0.1,
                    key=f"{key_prefix}_lw")
            with c2:
                fc['line_width_minor'] = st.number_input(
                    "Line width (minor)", value=float(fc.get('line_width_minor', 1.5)),
                    min_value=0.1, max_value=8.0, step=0.1,
                    key=f"{key_prefix}_lwm")
            with c3:
                _ms_val = fc.get('marker_size', 6)
                try:
                    _ms_val = int(_ms_val) if _ms_val is not None else 6
                except (TypeError, ValueError):
                    _ms_val = 6
                fc['marker_size'] = st.number_input(
                    "Marker size", value=_ms_val,
                    min_value=0, max_value=30, step=1,
                    key=f"{key_prefix}_msz")
            with c4:
                fc['marker_width'] = st.number_input(
                    "Marker edge w", value=float(fc.get('marker_width', 1.0)),
                    min_value=0.1, max_value=5.0, step=0.1,
                    key=f"{key_prefix}_mew")

        if show_series_colors:
            st.markdown("**🎨 Per-series colors** (Void / Matrix / Avg σ / Max σ / Porosity)")
            cols = st.columns(min(n_series, 5))
            for i in range(min(n_series, 5)):
                with cols[i]:
                    fc[f'color_series_{i+1}'] = st.color_picker(
                        f"Series {i+1}", value=fc.get(f'color_series_{i+1}', '#1f77b4'),
                        key=f"{key_prefix}_cs{i+1}")

        if show_legend:
            st.markdown("**📋 Legend**")
            c1, c2, c3 = st.columns(3)
            with c1:
                fc['show_legend'] = st.checkbox(
                    "Show legend", value=bool(fc.get('show_legend', True)),
                    key=f"{key_prefix}_sleg")
            with c2:
                leg_idx = LEGEND_POSITIONS.index(fc.get('legend_position', 'best')) \
                    if fc.get('legend_position', 'best') in LEGEND_POSITIONS else 0
                fc['legend_position'] = st.selectbox(
                    "Position", LEGEND_POSITIONS, index=leg_idx,
                    key=f"{key_prefix}_legpos")
            with c3:
                fc['legend_frameon'] = st.checkbox(
                    "Frame", value=bool(fc.get('legend_frameon', True)),
                    key=f"{key_prefix}_legf")
            c1, c2 = st.columns(2)
            with c1:
                fc['legend_framealpha'] = st.slider(
                    "Frame α", 0.0, 1.0,
                    float(fc.get('legend_framealpha', 1.0)),
                    0.05, key=f"{key_prefix}_legfa")

        if show_grid:
            st.markdown("**📊 Grid**")
            c1, c2, c3 = st.columns(3)
            with c1:
                fc['show_grid'] = st.checkbox(
                    "Show grid", value=bool(fc.get('show_grid', True)),
                    key=f"{key_prefix}_grid")
            with c2:
                fc['grid_alpha'] = st.slider(
                    "Grid α", 0.0, 1.0,
                    float(fc.get('grid_alpha', 0.3)),
                    0.05, key=f"{key_prefix}_ga")
            with c3:
                gls_idx = GRID_LINESTYLES.index(fc.get('grid_linestyle', '-')) \
                    if fc.get('grid_linestyle', '-') in GRID_LINESTYLES else 0
                fc['grid_linestyle'] = st.selectbox(
                    "Grid style", GRID_LINESTYLES, index=gls_idx,
                    key=f"{key_prefix}_gls",
                    format_func=lambda x: {'-': '— solid', '--': '–– dashed',
                                             ':': '·· dotted', '-.': '–· dash-dot',
                                             '': 'none'}.get(x, x))
            fc['grid_color'] = st.color_picker(
                "Grid color", value=fc.get('grid_color', '#cccccc'),
                key=f"{key_prefix}_gcol")

        if show_axis_limits:
            st.markdown("**📏 Axis limits** (blank = auto)")
            xlim_cur = fc.get('xlim', (None, None))
            ylim_cur = fc.get('ylim', (None, None))
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                xmin_in = st.text_input(
                    "xmin", value='' if xlim_cur[0] is None else str(xlim_cur[0]),
                    key=f"{key_prefix}_xmin")
            with c2:
                xmax_in = st.text_input(
                    "xmax", value='' if xlim_cur[1] is None else str(xlim_cur[1]),
                    key=f"{key_prefix}_xmax")
            with c3:
                ymin_in = st.text_input(
                    "ymin", value='' if ylim_cur[0] is None else str(ylim_cur[0]),
                    key=f"{key_prefix}_ymin")
            with c4:
                ymax_in = st.text_input(
                    "ymax", value='' if ylim_cur[1] is None else str(ylim_cur[1]),
                    key=f"{key_prefix}_ymax")
            def _parse(s):
                s = s.strip() if isinstance(s, str) else s
                if s is None or s == '':
                    return None
                try:
                    return float(s)
                except Exception:
                    return None
            fc['xlim'] = (_parse(xmin_in), _parse(xmax_in))
            fc['ylim'] = (_parse(ymin_in), _parse(ymax_in))

        if show_colorbar:
            st.markdown("**.colorbar**")
            c1, c2 = st.columns(2)
            with c1:
                fc['show_colorbar'] = st.checkbox(
                    "Show colorbar", value=bool(fc.get('show_colorbar', True)),
                    key=f"{key_prefix}_cb")
            with c2:
                cb_o_idx = 0 if fc.get('colorbar_orientation', 'vertical') == 'vertical' else 1
                fc['colorbar_orientation'] = st.selectbox(
                    "Colorbar orient.", ['vertical', 'horizontal'],
                    index=cb_o_idx, key=f"{key_prefix}_cbo")

        if show_spines:
            st.markdown("**▦ Spines (box)**")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                fc['spine_top'] = st.checkbox(
                    "Top", value=bool(fc.get('spine_top', True)),
                    key=f"{key_prefix}_spt")
            with c2:
                fc['spine_right'] = st.checkbox(
                    "Right", value=bool(fc.get('spine_right', True)),
                    key=f"{key_prefix}_spr")
            with c3:
                fc['spine_left'] = st.checkbox(
                    "Left", value=bool(fc.get('spine_left', True)),
                    key=f"{key_prefix}_spl")
            with c4:
                fc['spine_bottom'] = st.checkbox(
                    "Bottom", value=bool(fc.get('spine_bottom', True)),
                    key=f"{key_prefix}_spb")
            with c5:
                fc['spine_width'] = st.number_input(
                    "Width", value=float(fc.get('spine_width', 1.0)),
                    min_value=0.1, max_value=5.0, step=0.1,
                    key=f"{key_prefix}_sw")

        st.markdown("**⏱️ Ticks**")
        c1, c2, c3 = st.columns(3)
        with c1:
            td_idx = TICK_DIRECTIONS.index(fc.get('tick_direction', 'out')) \
                if fc.get('tick_direction', 'out') in TICK_DIRECTIONS else 0
            fc['tick_direction'] = st.selectbox(
                "Direction", TICK_DIRECTIONS, index=td_idx,
                key=f"{key_prefix}_td")
        with c2:
            fc['tick_width'] = st.number_input(
                "Width", value=float(fc.get('tick_width', 1.0)),
                min_value=0.1, max_value=5.0, step=0.1,
                key=f"{key_prefix}_tw")
        with c3:
            _tl_val = fc.get('tick_length', 4)
            try:
                _tl_val = int(_tl_val) if _tl_val is not None else 4
            except (TypeError, ValueError):
                _tl_val = 4
            fc['tick_length'] = st.number_input(
                "Length", value=_tl_val,
                min_value=0, max_value=20, step=1,
                key=f"{key_prefix}_tl")

        st.caption("✏️ All changes apply live — no Save button needed.")

    return True


def render_live_colormap_picker(viz_cfg, *, key_prefix='cmap',
                                  label='Per-field colormap (live)'):
    field_keys = list(FIELD_DEFAULT_CMAPS.keys())
    cmap_options = ColormapRegistry.list_all()
    fc = viz_cfg.setdefault('field_settings',
                              {f: {'auto': True, 'vmin': None, 'vmax': None,
                                   'log': False, 'invert': False,
                                   'cmap': FIELD_DEFAULT_CMAPS.get(f, 'viridis')}
                               for f in field_keys})
    default_expanded = bool(st.session_state.get(f'{key_prefix}_exp', False))
    with st.expander(f"🎨 {label}", expanded=default_expanded):
        st.session_state[f'{key_prefix}_exp'] = st.checkbox(
            "Keep expanded", value=default_expanded,
            key=f"{key_prefix}_keep_exp",
            help="If checked, this panel stays open across reruns.")
        st.caption(
            f"Pick a colormap for each of the {len(field_keys)} fields. "
            f"Changes apply LIVE on the next render — no Save button needed.")
        for i in range(0, len(field_keys), 2):
            c1, c2 = st.columns(2)
            for j, col in enumerate([c1, c2]):
                if i + j < len(field_keys):
                    with col:
                        fname = field_keys[i + j]
                        fset = fc.setdefault(fname, {})
                        cur_cmap = fset.get('cmap',
                                              FIELD_DEFAULT_CMAPS.get(fname, 'viridis'))
                        cmap_idx = (cmap_options.index(cur_cmap)
                                     if cur_cmap in cmap_options else 0)
                        new_cmap = st.selectbox(
                            f"{fname}",
                            cmap_options,
                            index=cmap_idx,
                            key=f"{key_prefix}_{fname}_cmap",
                            format_func=lambda x: x)
                        if new_cmap != cur_cmap:
                            fset['cmap'] = new_cmap
                        invert = st.checkbox(
                            f"Invert {fname}",
                            value=bool(fset.get('invert', False)),
                            key=f"{key_prefix}_{fname}_inv")
                        fset['invert'] = invert
                        try:
                            swatch_cmap = ColormapRegistry.resolve(new_cmap,
                                                                     invert=invert)
                            gradient = np.linspace(0, 1, 256).reshape(1, -1)
                            fig_sw, ax_sw = plt.subplots(figsize=(3.5, 0.35),
                                                          dpi=70)
                            ax_sw.imshow(gradient, aspect='auto', cmap=swatch_cmap)
                            ax_sw.set_axis_off()
                            ax_sw.set_title(f"{fname} → {new_cmap}",
                                             fontsize=8, pad=2)
                            st.pyplot(fig_sw, clear_figure=True); plt.close(fig_sw)
                        except Exception:
                            pass
    return True


# ============================================================================
# UO2 MATERIAL PROPERTIES
# ============================================================================
# UO2 properties at 1273 K. Non-dimensionalisation: l* = 1 nm, t* = 0.01 s.
# All diffusivities are stored as the scalar D_v / D_i coefficient that
# multiplies the (D_b + D_s + D_GB) tensor sum.
class UO2Properties:
    @staticmethod
    def get_uo2_properties():
        """Returns a dict of UO2 properties in reduced units (l*=1 nm, t*=0.01 s)."""
        # ---- Lattice / thermodynamics ----
        a_lattice = 0.547           # nm
        T = 1273.0                  # K
        k_B = 1.381e-23             # J/K
        kBT_eV = 8.617e-5 * T       # eV (Boltzmann in eV/K × T)

        # ---- Defect energetics (eV) ----
        Ef_v = 5.1                  # vacancy formation energy
        Ef_i = 7.5                  # interstitial formation energy
        E_surf = 0.6                # void free-surface energy (eV)
        E_gb   = 0.3                # GB energy (eV)

        # ---- Diffusivities (scalar coefficient) ----
        # Physical: D_v = 3.346e-17 m²/s, D_i = 1.623e-9 m²/s
        # Reduced: D_phys × t* / l*² = D_phys × 0.01 / 1e-18
        D_v = 3.346e-17 * 0.01 / 1e-18   # = 0.3346 (dimensionless)
        D_i = 1.623e-9  * 0.01 / 1e-18   # = 1.623e7 (dimensionless)

        # Mobility weights (w_b = w_s = 1e-4, w_GB = 0.1)
        w_b  = 1e-4
        w_s  = 1e-4
        w_GB = 0.1

        # ---- Gradient energy coefficient ----
        # Physical: 3.38e-8 J/m. Reduced: κ × t* / (N × kBT × l*²)
        # N = 1/a³ ≈ 6.13 nm⁻³ = 6.13e27 m⁻³. N×kBT ≈ 1.08e8 J/m³.
        # κ_reduced = 3.38e-8 × 0.01 / (1.08e8 × 1e-18) ≈ 3.13
        kappa = 3.13                  # reduced units (physical: 3.38e-8 J/m)

        # ---- Allen-Cahn mobility ----
        # Physical: L = 1.56e-11 m³/(J·s). Reduced: L × t* × N × kBT / l*²
        # L_reduced ≈ 1.56e-11 × 0.01 × 1.08e8 / 1e-18 ≈ 1.68e-5 (small)
        # Use a slightly larger default for visible evolution on a 256² grid.
        L_ac = 1.0                    # Allen-Cahn mobility (reduced)

        # ---- Elastoplastic (1273 K UO2) ----
        E_young = 220.0e9             # Pa
        nu_poisson = 0.31
        sigma_y0 = 150.0e6            # Pa, initial yield stress
        H_hardening = 5.0e9           # Pa, linear hardening modulus
        # Lame parameters (isotropic, plane-strain)
        lam = E_young * nu_poisson / ((1 + nu_poisson) * (1 - 2 * nu_poisson))
        mu  = E_young / (2 * (1 + nu_poisson))

        # Plasticity flow parameters (overstress power law, Peric-type)
        gamma0_dot = 1e-3             # reference strain rate
        m_overstress = 20             # rate sensitivity exponent
        b_burgers = 0.547 * np.sqrt(3) / 4.0  # Burgers vector for fluorite UO2 ≈ 0.237 nm

        # Larché-Cahn coupling (vacancy / interstitial relaxation volumes)
        # Ω_v = 0.5 × a³ (vacancy, tension-attracting), Ω_i = 1.0 × a³
        a_si = a_lattice * 1e-9       # m
        Omega_v = 0.5 * a_si ** 3
        Omega_i = 1.0 * a_si ** 3

        # Eigenstrain from vacancy condensation (lattice contraction)
        # β_v ≈ 0.5 (relaxation volume per vacancy, dimensionless)
        beta_v = 0.5

        return {
            'lattice': {'a': a_lattice, 'T': T, 'kBT_eV': kBT_eV},
            'defects': {'Ef_v': Ef_v, 'Ef_i': Ef_i,
                        'E_surf': E_surf, 'E_gb': E_gb,
                        'c_eq_v': 1e-6, 'c_eq_i': 1e-9},
            'diffusion': {'D_v': D_v, 'D_i': D_i,
                          'w_b': w_b, 'w_s': w_s, 'w_GB': w_GB},
            'gradient': {'kappa': kappa},
            'mobility': {'L_ac': L_ac},
            'elastic': {'E': E_young, 'nu': nu_poisson,
                        'lam': lam, 'mu': mu,
                        'C11_2d': lam + 2 * mu, 'C12_2d': lam, 'C44_2d': mu},
            'plasticity': {'sigma0': sigma_y0, 'H': H_hardening,
                           'gamma0_dot': gamma0_dot, 'm': m_overstress,
                           'b': b_burgers, 'mu_p': mu, 'nu_p': nu_poisson},
            'larche_cahn': {'Omega_v': Omega_v, 'Omega_i': Omega_i,
                            'beta_v': beta_v},
        }

    @staticmethod
    def get_material(name='UO2'):
        return UO2Properties.get_uo2_properties()

    @staticmethod
    def validate_parameters(params):
        errors, warns = [], []
        if params.get('dt', 0) <= 0:  errors.append("dt must be positive")
        if params.get('dx', 0) <= 0:  errors.append("dx must be positive")
        if params.get('Nx', 0) < 32 or params.get('Ny', 0) < 32:
            warns.append("N < 32 per axis may be inaccurate")
        # Stability check for explicit cascade source / recombination
        K_dot = float(params.get('K_dot', 0.0))
        if K_dot > 0.05:
            warns.append(f"K_dot = {K_dot} is large; cascade production may "
                         f"cause numerical instabilities. Consider <0.02.")
        # J2 plasticity stability
        dt = float(params.get('dt', 1e-2))
        if dt > 0.05:
            warns.append(f"dt = {dt} > 0.05; J2 plasticity may overshoot. "
                         f"Use 0.005-0.02 for stable explicit plastic flow.")
        return errors, warns


# ============================================================================
# VORONOI GEOMETRY (re-used from original code)
# ============================================================================
def voronoi_assign(Nx, Ny, extent, seeds, mask=None, warp=None, periodic=False):
    x = np.linspace(extent[0], extent[1], Nx)
    y = np.linspace(extent[2], extent[3], Ny)
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    if len(seeds) == 0:
        return np.full((Ny, Nx), -1, dtype=int)

    if warp is not None:
        wx, wy = warp
        pts_w = pts.copy()
        ix = np.clip(((pts[:, 0] - extent[0]) / (extent[1] - extent[0]) *
                      (wx.shape[1] - 1)).astype(int), 0, wx.shape[1] - 1)
        iy = np.clip(((pts[:, 1] - extent[2]) / (extent[3] - extent[2]) *
                      (wx.shape[0] - 1)).astype(int), 0, wx.shape[0] - 1)
        pts_w[:, 0] += wx[iy, ix]
        pts_w[:, 1] += wy[iy, ix]
        seed_ix = np.clip(
            ((seeds[:, 0] - extent[0]) / (extent[1] - extent[0]) *
             (wx.shape[1] - 1)).astype(int), 0, wx.shape[1] - 1)
        seed_iy = np.clip(
            ((seeds[:, 1] - extent[2]) / (extent[3] - extent[2]) *
             (wx.shape[0] - 1)).astype(int), 0, wx.shape[0] - 1)
        warped_seeds = seeds + np.column_stack([wx[seed_iy, seed_ix],
                                                 wy[seed_iy, seed_ix]])
        tree = cKDTree(warped_seeds)
        _, idx = tree.query(pts_w, k=1)
    else:
        tree = cKDTree(seeds)
        _, idx = tree.query(pts, k=1)

    gm = idx.reshape(Ny, Nx).astype(int)
    if mask is not None:
        gm = np.where(mask, gm, -1)
    return gm


def diffuse_field(binary, sigma_px):
    if sigma_px <= 0:
        return binary.astype(float)
    return gaussian_filter(binary.astype(float), sigma=sigma_px, mode='nearest')


def make_warp_field(Nx, Ny, extent, amplitude, wavelength_nm, seed=0):
    rng = np.random.default_rng(seed)
    Lx = extent[1] - extent[0]
    Ly = extent[3] - extent[2]
    nx_modes = max(2, int(Lx / wavelength_nm))
    ny_modes = max(2, int(Ly / wavelength_nm))
    x = np.linspace(0, 1, Nx); y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y)
    wx = np.zeros_like(X); wy = np.zeros_like(Y)
    for i in range(1, nx_modes + 1):
        for j in range(1, ny_modes + 1):
            ph = rng.uniform(0, 2 * np.pi)
            amp = amplitude / (i * j)
            wx += amp * np.sin(2 * np.pi * i * X + ph) * np.cos(2 * np.pi * j * Y)
            wy += amp * np.cos(2 * np.pi * i * X) * np.sin(2 * np.pi * j * Y + ph)
    return wx, wy


# ============================================================================
# UO2 GEOMETRY — 3 SCENARIOS
# ============================================================================
class UO2Geometry:
    """Builds initial c_v, c_i, eta, phi_i fields for three scenarios:
      (1) 'multi_void_gb'  — 3 voids of different radii in front of a
                              flat GB, no irradiation.
      (2) 'gb_irradiation' — flat GB + irradiation source.
      (3) 'polycrystal'    — 64-grain periodic polycrystal with
                              irradiation-driven void formation.

    Returns
    -------
    c_v0, c_i0, eta0, phi_list0, grain_map, meta
        - c_v0     : (Ny, Nx) initial vacancy concentration
        - c_i0     : (Ny, Nx) initial interstitial concentration
        - eta0     : (Ny, Nx) initial void order parameter (1 in void, 0 matrix)
        - phi_list0: list of (Ny, Nx) grain-orientation order parameters
        - grain_map: (Ny, Nx) int grain-ID map (argmax of phi_list0)
        - meta     : dict of scenario parameters and field positions
    """

    def __init__(self, Nx, Ny, dx, dy=None):
        self.Nx, self.Ny = Nx, Ny
        self.dx = dx
        self.dy = dy if dy is not None else dx
        self.extent = [-Nx * self.dx / 2, Nx * self.dx / 2,
                       -Ny * self.dy / 2, Ny * self.dy / 2]
        self.x = np.linspace(self.extent[0], self.extent[1], Nx)
        self.y = np.linspace(self.extent[2], self.extent[3], Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    # ------------------------------------------------------------------
    @handle_errors
    def build(self,
              scenario='multi_void_gb',
              void_radii=(3.0, 4.0, 5.0),
              void_positions=None,
              n_grains_flat=2,
              gb_width_nm=3.0,
              n_poly_grains=64,
              poly_grain_size_nm=30.0,
              use_warp=True,
              warp_amp=2.0,
              c_eq_v=1e-6,
              c_eq_i=1e-9,
              gb_energy=0.3,
              surface_energy=0.6,
              kappa=3.13,
              seed=1234):
        """Build the initial fields for the chosen scenario."""
        if scenario == 'multi_void_gb':
            return self._build_multi_void_gb(
                void_radii, void_positions, n_grains_flat, gb_width_nm,
                c_eq_v, c_eq_i, gb_energy, surface_energy, kappa, seed)
        elif scenario == 'gb_irradiation':
            return self._build_gb_irradiation(
                n_grains_flat, gb_width_nm,
                c_eq_v, c_eq_i, gb_energy, surface_energy, kappa, seed)
        elif scenario == 'polycrystal':
            return self._build_polycrystal(
                n_poly_grains, poly_grain_size_nm, use_warp, warp_amp,
                c_eq_v, c_eq_i, gb_energy, surface_energy, kappa, seed)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    # ------------------------------------------------------------------
    def _make_void_disk(self, x0, y0, r):
        """Smooth circular void (η=1 inside, 0 outside, diffuse boundary)."""
        d = np.sqrt((self.X - x0) ** 2 + (self.Y - y0) ** 2)
        # Smooth transition over ~1 nm
        eta = 0.5 * (1.0 - np.tanh((d - r) / 0.8))
        return np.clip(eta, 0, 1)

    # ------------------------------------------------------------------
    def _make_flat_gb(self, gb_x=0.0, gb_width_nm=3.0, n_grains=2):
        """Diffuse flat GB at x=gb_x, separating grain 1 (x<gb_x) from grain 2.
        Returns phi_list = [phi_1, phi_2] each in [0,1]."""
        # Smooth transition: phi_1 = 0.5*(1 - tanh((x-gb_x)/w))
        w = gb_width_nm / 2.0
        phi_1 = 0.5 * (1.0 - np.tanh((self.X - gb_x) / max(w, 0.1)))
        phi_2 = 1.0 - phi_1
        return [phi_1, phi_2]

    # ------------------------------------------------------------------
    def _build_multi_void_gb(self, void_radii, void_positions,
                              n_grains, gb_width_nm,
                              c_eq_v, c_eq_i, gb_energy, surface_energy,
                              kappa, seed):
        """Scenario 1: 3 voids of different radii in front of a flat GB."""
        Nx, Ny = self.Nx, self.Ny
        # GB at x = 0, voids placed at x < 0 (grain 1 side)
        phi_list = self._make_flat_gb(gb_x=0.0, gb_width_nm=gb_width_nm,
                                       n_grains=n_grains)

        # Place voids vertically along x<0 if not given
        if void_positions is None:
            # Three voids stacked vertically in grain 1 (x<0)
            x_offset = -15.0
            y_positions = [-10.0, 0.0, 10.0]
            void_positions = list(zip([x_offset] * len(void_radii),
                                       y_positions[:len(void_radii)]))

        eta = np.zeros((Ny, Nx), dtype=float)
        for (x0, y0), r in zip(void_positions, void_radii):
            eta = np.maximum(eta, self._make_void_disk(x0, y0, r))

        # Concentration fields: void → c_v=1, c_i=0; matrix → c_eq
        c_v = np.where(eta > 0.5, 1.0, c_eq_v)
        c_i = np.where(eta > 0.5, 0.0, c_eq_i)
        # Smooth the transition
        c_v = 0.5 * (1 - np.cos(np.pi * eta)) * 1.0 + \
              0.5 * (1 + np.cos(np.pi * eta)) * c_eq_v
        c_i = 0.5 * (1 - np.cos(np.pi * eta)) * 0.0 + \
              0.5 * (1 + np.cos(np.pi * eta)) * c_eq_i

        grain_map = np.where(self.X < 0, 0, 1).astype(int)

        meta = {
            'scenario': 'multi_void_gb',
            'n_grains': n_grains,
            'gb_x': 0.0,
            'gb_width_nm': gb_width_nm,
            'void_radii': list(void_radii),
            'void_positions': list(void_positions),
            'c_eq_v': c_eq_v, 'c_eq_i': c_eq_i,
            'gb_energy': gb_energy, 'surface_energy': surface_energy,
            'kappa': kappa, 'seed': seed,
            'has_irradiation': False,
            'has_driving_force': True,
            'f_d_default': (0.2, 0.0),
        }
        return c_v, c_i, eta, phi_list, grain_map, meta

    # ------------------------------------------------------------------
    def _build_gb_irradiation(self, n_grains, gb_width_nm,
                               c_eq_v, c_eq_i, gb_energy, surface_energy,
                               kappa, seed):
        """Scenario 2: flat GB + irradiation source, no initial voids."""
        Nx, Ny = self.Nx, self.Ny
        phi_list = self._make_flat_gb(gb_x=0.0, gb_width_nm=gb_width_nm,
                                       n_grains=n_grains)
        eta = np.zeros((Ny, Nx), dtype=float)
        c_v = np.full((Ny, Nx), c_eq_v, dtype=float)
        c_i = np.full((Ny, Nx), c_eq_i, dtype=float)

        grain_map = np.where(self.X < 0, 0, 1).astype(int)

        meta = {
            'scenario': 'gb_irradiation',
            'n_grains': n_grains,
            'gb_x': 0.0,
            'gb_width_nm': gb_width_nm,
            'void_radii': [],
            'void_positions': [],
            'c_eq_v': c_eq_v, 'c_eq_i': c_eq_i,
            'gb_energy': gb_energy, 'surface_energy': surface_energy,
            'kappa': kappa, 'seed': seed,
            'has_irradiation': True,
            'has_driving_force': True,
            'f_d_default': (0.4, 0.0),
            'K_dot_default': 5e-3,
        }
        return c_v, c_i, eta, phi_list, grain_map, meta

    # ------------------------------------------------------------------
    def _build_polycrystal(self, n_grains, grain_size_nm, use_warp, warp_amp,
                            c_eq_v, c_eq_i, gb_energy, surface_energy,
                            kappa, seed):
        """Scenario 3: periodic polycrystalline UO2 with random Voronoi grains."""
        Nx, Ny = self.Nx, self.Ny
        rng = np.random.default_rng(seed)
        Lx = self.extent[1] - self.extent[0]
        Ly = self.extent[3] - self.extent[2]

        # Place grain seeds randomly (allowing some to be off-edge for periodicity)
        # Cap n_grains to fit memory: ~32 MB per field at 256² × 8 bytes
        max_grains = min(n_grains, 80)
        seeds = np.column_stack([
            rng.uniform(self.extent[0], self.extent[1], max_grains),
            rng.uniform(self.extent[2], self.extent[3], max_grains),
        ])

        # Optional warp for non-straight GBs
        if use_warp and warp_amp > 0.1:
            wx, wy = make_warp_field(Nx, Ny, self.extent,
                                      amplitude=warp_amp,
                                      wavelength_nm=grain_size_nm * 0.8,
                                      seed=seed)
            warp = (wx, wy)
        else:
            warp = None

        grain_map = voronoi_assign(Nx, Ny, self.extent, seeds,
                                    mask=None, warp=warp)

        # Build phi_i fields (one per grain)
        # diffuse GB width in pixels
        sigma_gb = max(1.5, 0.5 * 2.5 / self.dx)  # ~2.5 nm GB
        phi_list = []
        for g in range(max_grains):
            m = (grain_map == g).astype(float)
            phi = np.clip(diffuse_field(m, sigma_gb), 0, 1)
            phi_list.append(phi)
        # Normalize so sum(phi_i) = 1 in matrix
        ssum = np.sum(phi_list, axis=0)
        ssum_safe = np.where(ssum > 1e-9, ssum, 1.0)
        phi_list = [p / ssum_safe for p in phi_list]

        # No voids initially; concentrations at equilibrium
        eta = np.zeros((Ny, Nx), dtype=float)
        c_v = np.full((Ny, Nx), c_eq_v, dtype=float)
        c_i = np.full((Ny, Nx), c_eq_i, dtype=float)

        meta = {
            'scenario': 'polycrystal',
            'n_grains': max_grains,
            'grain_size_nm': grain_size_nm,
            'gb_x': None,
            'gb_width_nm': 2.5,
            'void_radii': [],
            'void_positions': [],
            'c_eq_v': c_eq_v, 'c_eq_i': c_eq_i,
            'gb_energy': gb_energy, 'surface_energy': surface_energy,
            'kappa': kappa, 'seed': seed,
            'has_irradiation': True,
            'has_driving_force': False,
            'f_d_default': (0.0, 0.0),
            'K_dot_default': 5e-3,
            'grain_seeds': seeds,
        }
        return c_v, c_i, eta, phi_list, grain_map, meta


# ============================================================================
# SPECTRAL ELASTICITY SOLVER (plane-strain, isotropic UO2)
# ============================================================================
class SpectralElasticSolver:
    """Spectral elasticity solver for plane-strain isotropic UO2.

    Takes (lam, mu) Lame parameters (derived from E=220 GPa, ν=0.31 for the
    UO2 defaults). The solver builds the periodic Green's
    function for plane-strain elasticity and returns stress + strain from
    a given eigenstrain + applied stress.
    """
    def __init__(self, Nx, Ny, dx, dy, elastic_params, kx=None, ky=None, k2=None):
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = dx, dy
        if kx is None:
            self.kx, self.ky, self.k2 = make_k_vectors(Nx, Ny, dx, dy)
        else:
            self.kx, self.ky, self.k2 = kx, ky, k2

        # Plane-strain 2D moduli (isotropic)
        # C11_2d = lam + 2*mu, C12_2d = lam, C44_2d = mu
        self.lam = float(elastic_params['lam'])
        self.mu  = float(elastic_params['mu'])
        self.C11_2d = self.lam + 2 * self.mu
        self.C12_2d = self.lam
        self.C44_2d = self.mu

        # Build Green's function in Fourier space
        k2 = self.k2
        inv_mu_k2 = 1.0 / (self.mu * k2 + 1e-30)
        ratio = (self.lam + self.mu) / (self.lam + 2 * self.mu)
        self.G11 = inv_mu_k2 * (1.0 - ratio * (self.kx**2) / k2)
        self.G12 = inv_mu_k2 * (-ratio * self.kx * self.ky / k2)
        self.G22 = inv_mu_k2 * (1.0 - ratio * (self.ky**2) / k2)
        # Zero out the mean (k=0) mode to fix translational invariance
        self.G11[0, 0] = self.G12[0, 0] = self.G22[0, 0] = 0.0

    @handle_errors
    def solve(self, exx_star, eyy_star, exy_star,
              app_xx=0.0, app_yy=0.0, app_xy=0.0):
        """Solve for stress given an eigenstrain (exx_star, eyy_star, exy_star)
        and applied far-field stress (app_xx, app_yy, app_xy).

        Returns (sigma_eq, sxx, syy, sxy, sigma_h, exx, eyy, exy).
        """
        exx_h = fft2(exx_star)
        eyy_h = fft2(eyy_star)
        exy_h = fft2(exy_star)

        eps_kk_h = exx_h + eyy_h
        fx_h = self.lam * self.kx * eps_kk_h + 2 * self.mu * (
            self.kx * exx_h + self.ky * exy_h)
        fy_h = self.lam * self.ky * eps_kk_h + 2 * self.mu * (
            self.kx * exy_h + self.ky * eyy_h)

        ux_h = self.G11 * fx_h + self.G12 * fy_h
        uy_h = self.G12 * fx_h + self.G22 * fy_h

        exx_el = np.real(ifft2(1j * self.kx * ux_h))
        eyy_el = np.real(ifft2(1j * self.ky * uy_h))
        exy_el = 0.5 * np.real(ifft2(1j * (self.kx * uy_h + self.ky * ux_h)))

        exx = exx_el + exx_star
        eyy = eyy_el + eyy_star
        exy = exy_el + exy_star

        sxx = self.C11_2d * exx_el + self.C12_2d * eyy_el + app_xx
        syy = self.C12_2d * exx_el + self.C11_2d * eyy_el + app_yy
        sxy = 2 * self.C44_2d * exy_el + app_xy

        sigma_eq = np.sqrt(0.5 * ((sxx - syy) ** 2 + syy ** 2 + sxx ** 2
                                  + 6 * sxy ** 2))
        sigma_eq = np.clip(sigma_eq, 0, 5e9)
        sigma_h = 0.5 * (sxx + syy)

        return sigma_eq, sxx, syy, sxy, sigma_h, exx, eyy, exy


# ============================================================================
# UO2 VOID-GB PHASE-FIELD SOLVER
# ============================================================================
# Implements the modified Cahn-Hilliard + Allen-Cahn + J2 elastoplasticity
# model, with the following extensions:
#   * Spectral semi-implicit time stepping for improved stability at
#     larger dt.
#   * Full J2 plasticity flow rule (overstress power law, Hall-Petch
#     yield) retained from the original Cu/Au/Cu code, re-parameterised
#     for UO2 with UO2 elastoplastic constants.
#   * Larché-Cahn stress coupling: hydrostatic stress σ_h enters the
#     vacancy/interstitial chemical potentials via Ω_v·σ_h / Ω_i·σ_h.
class UO2VoidGBSolver:
    """Solves the UO2 void-GB phase-field model.

    State variables (per grid point):
      c_v     : vacancy concentration (conserved)
      c_i     : interstitial concentration (conserved)
      eta     : void order parameter (η=1 void, η=0 matrix, non-conserved)
      phi_list: list of grain-orientation order parameters φ_i (non-conserved)
      eps_p_* : accumulated plastic strain components (xx, yy, xy)

    Time stepping (per call to step()):
      1. Compute GB indicator Φ = Σφ_i² and ∇c_v, ∇c_i, ∇φ_i.
      2. Build anisotropic mobility tensors D_v, D_i (bulk + surface + GB).
      3. Solve elastoplasticity: eigenstrain from vacancies + plastic strain
         → σ_eq, σ_h, S_ij (deviatoric).
      4. J2 plasticity update: Hall-Petch yield + overstress power-law flow.
      5. Compute chemical potentials μ_v, μ_i (with Larché-Cahn σ_h coupling).
      6. Cahn-Hilliard step for c_v, c_i (semi-implicit spectral + explicit
         cascade source P_v/P_i, recombination A_iv, GB sinks S_v^GB/S_i^GB).
      7. Allen-Cahn step for eta (with stochastic + P_v source).
      8. Allen-Cahn step for phi_i (with optional artificial driving force).
      9. Update history (porosity, mean c_v, GB area, max σ_eq, plastic work).
    """

    def __init__(self, params):
        self.params = params
        self.Nx = params['Nx']
        self.Ny = params['Ny']
        self.dx = params['dx']
        self.dy = params.get('dy', params['dx'])
        self.dt = params['dt']
        self.mat = UO2Properties.get_material(params.get('material', 'UO2'))

        # Spectral operators
        self.kx, self.ky, self.k2 = make_k_vectors(self.Nx, self.Ny,
                                                   self.dx, self.dy)

        # Geometry — scenario-driven initial state
        self.geom = UO2Geometry(self.Nx, self.Ny, self.dx, self.dy)
        result = self.geom.build(
            scenario=params.get('scenario', 'multi_void_gb'),
            void_radii=params.get('void_radii', (3.0, 4.0, 5.0)),
            void_positions=params.get('void_positions', None),
            n_grains_flat=params.get('n_grains_flat', 2),
            gb_width_nm=params.get('gb_width_nm', 3.0),
            n_poly_grains=params.get('n_poly_grains', 64),
            poly_grain_size_nm=params.get('poly_grain_size_nm', 30.0),
            use_warp=params.get('use_warp', True),
            warp_amp=params.get('warp_amp', 2.0),
            c_eq_v=self.mat['defects']['c_eq_v'],
            c_eq_i=self.mat['defects']['c_eq_i'],
            gb_energy=self.mat['defects']['E_gb'],
            surface_energy=self.mat['defects']['E_surf'],
            kappa=self.mat['gradient']['kappa'],
            seed=params.get('seed', 1234))
        if result is FAILED or result is None:
            raise RuntimeError(f"Geometry build failed: {FAILED.reason}")
        (self.c_v, self.c_i, self.eta, self.phi_list,
         self.grain_map, self.geom_meta) = result

        self.n_grains = len(self.phi_list)
        self.scenario = self.geom_meta['scenario']

        # Material constants (extracted for convenience)
        defects = self.mat['defects']
        diff = self.mat['diffusion']
        grad = self.mat['gradient']
        mob = self.mat['mobility']
        self.Ef_v = defects['Ef_v']
        self.Ef_i = defects['Ef_i']
        self.c_eq_v = defects['c_eq_v']
        self.c_eq_i = defects['c_eq_i']
        self.E_surf = defects['E_surf']
        self.E_gb = defects['E_gb']
        self.D_v_scalar = diff['D_v']
        self.D_i_scalar = diff['D_i']
        self.w_b = diff['w_b']
        self.w_s = diff['w_s']
        self.w_GB = diff['w_GB']
        self.kappa = grad['kappa']
        # Per-field gradient coefficients (uses same kappa for all fields)
        self.kappa_cv = self.kappa
        self.kappa_ci = self.kappa
        self.kappa_phi = self.kappa
        self.kappa_eta = self.kappa
        self.L_ac = float(params.get('L_ac', mob['L_ac']))
        # Polycrystal energy parameters
        self.a_GB = 1.0           # diffuse GB interface parameter
        self.a_s = self.a_GB * self.E_surf / max(self.E_gb, 1e-9)
        self.mu_poly = 1.0        # polycrystal energy coefficient

        # kBT in eV (used for entropy term in f_s)
        self.kBT = self.mat['lattice']['kBT_eV']

        # Larché-Cahn coupling (vacancy/interstitial relaxation volumes)
        lc = self.mat['larche_cahn']
        # The Ω_v·σ_h term enters the chemical potential in eV (since f_s is in eV)
        # σ_h in Pa → divide by (N·kBT in Pa/eV) to convert to eV per defect
        # N·kBT = (1/a³)·kBT (per nm³) → in SI: 6.13e27 × 1.38e-23 × 1273 ≈ 1.08e8 J/m³ = 1.08e8 Pa
        # So Ω_v·σ_h [eV] = Ω_v [m³] × σ_h [Pa] / (1.08e8 Pa/eV) ≈ Ω_v·σ_h × 9.26e-9 eV
        # For simplicity, we use a tunable coupling constant:
        self.Omega_v_eff = float(params.get('Omega_v_eff', 0.1))  # eV/GPa (tunable)
        self.Omega_i_eff = float(params.get('Omega_i_eff', 0.2))  # eV/GPa (tunable)
        self.beta_v = lc['beta_v']

        # Elastoplasticity solver
        self.elastic = SpectralElasticSolver(
            self.Nx, self.Ny, self.dx, self.dy, self.mat['elastic'],
            self.kx, self.ky, self.k2)

        # Plastic strain state
        self.eps_p_xx = np.zeros((self.Ny, self.Nx))
        self.eps_p_yy = np.zeros((self.Ny, self.Nx))
        self.eps_p_xy = np.zeros((self.Ny, self.Nx))

        # Irradiation parameters
        self.K_dot = float(params.get('K_dot', 0.0))
        self.P_casc = float(params.get('P_casc', 0.05))
        # V_G = K_dot / P_casc (max concentration increment per cascade)
        self.V_G = self.K_dot / max(self.P_casc, 1e-9)
        self.production_bias = float(params.get('production_bias', 2.0 / 3.0))
        # Recombination rates
        self.A_bulk = float(params.get('A_bulk', 1.0))
        self.A_surf = float(params.get('A_surf', 5.0))
        # GB sink strengths
        self.S_v_r = float(params.get('S_v_r', 1.0))
        self.S_i_r = float(params.get('S_i_r', 1.0))

        # Artificial driving force — only used in flat-GB scenarios
        f_d = params.get('f_d', self.geom_meta.get('f_d_default', (0.0, 0.0)))
        self.f_d_x = float(f_d[0])
        self.f_d_y = float(f_d[1])

        # Applied far-field stress
        self.applied_stress = float(params.get('applied_stress', 0.0))

        # Random generator (for cascade stochasticity)
        self.rng = np.random.default_rng(params.get('seed', 1234) + 1)

        # History tracking (for diagnostics)
        self.history = {
            'time':         [],
            'porosity':     [],
            'mean_c_v':     [],
            'mean_c_i':     [],
            'gb_area':      [],
            'max_stress':   [],
            'avg_stress':   [],
            'plastic_work': [],
            'n_voids':      [],
            'void_volume':  [],
        }

    # ==================================================================
    # FIELD DIAGNOSTICS
    # ==================================================================
    def compute_gb_indicator(self):
        """Phi = Sum phi_i^2 -- equal to 1 inside a grain, < 1 at GBs and triple
        junctions."""
        s = np.zeros((self.Ny, self.Nx))
        for phi in self.phi_list:
            s += phi ** 2
        return np.clip(s, 0, 1)

    def compute_gb_area(self):
        """Total GB area = count of cells where Φ < 0.9, weighted by dx·dy.
        This is a proxy for GB length in 2D."""
        phi = self.compute_gb_indicator()
        return float(np.sum(phi < 0.9) * self.dx * self.dy)

    def compute_local_grain_size(self):
        """Local grain size from spectral gradient of Σφ_i (analogous to original code)."""
        s = np.sum(self.phi_list, axis=0)
        gx, gy = spectral_gradients(s, self.kx, self.ky)
        gm = np.sqrt(gx ** 2 + gy ** 2 + 1e-15)
        return np.clip(2.0 / gm, 0, 300.0)

    def compute_porosity(self):
        """Porosity = mean of η² (void fraction)."""
        return float(np.mean(self.eta ** 2))

    def count_voids(self):
        """Estimate number of distinct void regions via connected components
        on (η > 0.5)."""
        from scipy.ndimage import label
        mask = self.eta > 0.5
        if not np.any(mask):
            return 0, 0.0
        labeled, n = label(mask)
        # Volume = sum of η² per labeled region (proxy)
        total_vol = float(np.sum(self.eta ** 2) * self.dx * self.dy)
        return int(n), total_vol

    # ==================================================================
    # ANISOTROPIC MOBILITY (bulk + surface + GB tensor contributions)
    # ==================================================================
    def compute_mobility_field_v(self):
        """Effective scalar mobility field for vacancies.

        Combines bulk, surface, and GB contributions.
        We collapse the tensorial form into an effective scalar mobility
        M_v(r) = D_v · (w_b + w_s · c_v²·(1-c_v²) + w_GB · (1-Φ)),
        which captures the magnitude of the surface/GB enhancement while
        dropping the directional projection tensors T_s, T_GB. This is
        numerically cheaper and works well for 2D simulations.
        """
        Phi = self.compute_gb_indicator()
        cv = self.c_v
        # Bulk + surface + GB contributions
        M_v = self.D_v_scalar * (
            self.w_b
            + self.w_s * (cv ** 2) * (1.0 - cv ** 2)
            + self.w_GB * (1.0 - Phi)
        )
        return np.maximum(M_v, 1e-12)

    def compute_mobility_field_i(self):
        """Effective scalar mobility field for interstitials (same form as vacancies)."""
        Phi = self.compute_gb_indicator()
        ci = self.c_i
        M_i = self.D_i_scalar * (
            self.w_b
            + self.w_s * (ci ** 2) * (1.0 - ci ** 2)
            + self.w_GB * (1.0 - Phi)
        )
        return np.maximum(M_i, 1e-12)

    # ==================================================================
    # FREE ENERGY DERIVATIVES
    # ==================================================================
    def compute_chemical_potential_v(self, sigma_h_Pa):
        """mu_v = dF/dc_v  (bulk free energy derivative + Larche-Cahn coupling).

        μ_v = N · [ h(η)·(Ef_v + kBT·ln(c_v/(1-c_v-c_i)))
                   + j(η)·2·(c_v - 1) ]
              - κ_cv · ∇²c_v
              - Ω_v_eff · σ_h_GPa            (Larché-Cahn)
        """
        eta = self.eta
        cv = self.c_v
        ci = self.c_i
        # Interpolating functions: h(η)=(1-η)², j(η)=η²
        h_eta = (1.0 - eta) ** 2
        j_eta = eta ** 2
        # Solid-phase free energy derivative: f_s = Ef_v·c_v + Ef_i·c_i + kBT·[...]
        # df_s/dc_v = Ef_v + kBT·(ln(c_v) - ln(1-c_v-c_i)) = Ef_v + kBT·ln(c_v/(1-c_v-c_i))
        sum_c = np.clip(cv + ci, 1e-30, 1.0 - 1e-30)
        ratio = np.log(np.maximum(cv, 1e-30) / np.maximum(1.0 - sum_c, 1e-30))
        df_s_dc_v = self.Ef_v + self.kBT * ratio
        # Void-phase free energy derivative: f_v = (c_v-1)² + c_i² → df_v/dc_v = 2(c_v-1)
        df_v_dc_v = 2.0 * (cv - 1.0)
        # Combine via interpolating functions (note N=1 in reduced units)
        mu_v = h_eta * df_s_dc_v + j_eta * df_v_dc_v
        # Gradient term: -κ_cv·∇²c_v
        mu_v -= self.kappa_cv * spectral_laplacian(cv, self.k2)
        # Larché-Cahn stress coupling (Ω_v·σ_h in eV per defect, σ_h in GPa)
        sigma_h_GPa = sigma_h_Pa / 1e9
        mu_v -= self.Omega_v_eff * sigma_h_GPa
        return mu_v

    def compute_chemical_potential_i(self, sigma_h_Pa):
        """μ_i = δF/δc_i  (analogous to μ_v, with Ω_i sign flipped)."""
        eta = self.eta
        cv = self.c_v
        ci = self.c_i
        h_eta = (1.0 - eta) ** 2
        j_eta = eta ** 2
        sum_c = np.clip(cv + ci, 1e-30, 1.0 - 1e-30)
        ratio = np.log(np.maximum(ci, 1e-30) / np.maximum(1.0 - sum_c, 1e-30))
        df_s_dc_i = self.Ef_i + self.kBT * ratio
        df_v_dc_i = 2.0 * ci
        mu_i = h_eta * df_s_dc_i + j_eta * df_v_dc_i
        mu_i -= self.kappa_ci * spectral_laplacian(ci, self.k2)
        # Larché-Cahn: interstitials prefer compressive regions (sign +)
        sigma_h_GPa = sigma_h_Pa / 1e9
        mu_i += self.Omega_i_eff * sigma_h_GPa
        return mu_i

    def compute_dF_deta(self):
        """dF/deta  (free energy derivative with respect to void OP).

        δF/δη = N · [ 2(η-1)·f_s + 2η·f_v
                     + μ_poly·(η³ - η + 2·a_s·η·Σφ_i²) ]
                - κ_η · ∇²η
        """
        eta = self.eta
        cv = self.c_v
        ci = self.c_i
        # f_s and f_v (free energy densities)
        sum_c = np.clip(cv + ci, 1e-30, 1.0 - 1e-30)
        f_s = (self.Ef_v * cv + self.Ef_i * ci
               + self.kBT * (cv * np.log(np.maximum(cv, 1e-30))
                             + ci * np.log(np.maximum(ci, 1e-30))
                             + (1 - sum_c) * np.log(np.maximum(1 - sum_c, 1e-30))))
        f_v = (cv - 1.0) ** 2 + ci ** 2
        # Σφ_i²
        sum_phi_sq = self.compute_gb_indicator()
        # Polycrystal coupling
        poly_term = self.mu_poly * (eta ** 3 - eta
                                     + 2.0 * self.a_s * eta * sum_phi_sq)
        dF_deta = 2.0 * (eta - 1.0) * f_s + 2.0 * eta * f_v + poly_term
        # Gradient term
        dF_deta -= self.kappa_eta * spectral_laplacian(eta, self.k2)
        return dF_deta

    def compute_dF_dphi(self, i):
        """dF/dphi_i  (free energy derivative with respect to grain OP).

        δF/δφ_i = N · μ_poly · [ φ_i³ - φ_i + 2·a_GB·φ_i·Σ_{j≠i}φ_j²
                                  + 2·a_s·η²·φ_i ]
                  - κ_φ · ∇²φ_i
        """
        phi_i = self.phi_list[i]
        eta = self.eta
        # Σ_{j≠i} φ_j² = (Σφ_j²) - φ_i²
        sum_phi_sq_all = self.compute_gb_indicator()
        sum_others_sq = sum_phi_sq_all - phi_i ** 2
        # Polynomial + interaction terms
        dF_dphi = self.mu_poly * (
            phi_i ** 3 - phi_i
            + 2.0 * self.a_GB * phi_i * sum_others_sq
            + 2.0 * self.a_s * (eta ** 2) * phi_i
        )
        # Gradient term
        dF_dphi -= self.kappa_phi * spectral_laplacian(phi_i, self.k2)
        return dF_dphi

    # ==================================================================
    # CASCADE SOURCE TERMS (stochastic vacancy/interstitial pair production)
    # ==================================================================
    def compute_cascade_source(self):
        """Stochastic cascade production of vacancy/interstitial pairs.

        P_v(r,t) = R2·V_G   if η<0.8 and R1≤P_casc, else 0
        P_i      = production_bias · P_v
        """
        if self.K_dot <= 0:
            return np.zeros_like(self.eta), np.zeros_like(self.eta)
        # Draw R1 (cascade trigger) and R2 (intensity) per cell
        R1 = self.rng.uniform(0.0, 1.0, size=self.eta.shape)
        R2 = self.rng.uniform(0.0, 1.0, size=self.eta.shape)
        # Cascades only in matrix (η<0.8)
        mask = (self.eta < 0.8) & (R1 <= self.P_casc)
        P_v = np.where(mask, R2 * self.V_G, 0.0)
        P_i = self.production_bias * P_v
        return P_v, P_i

    # ==================================================================
    # RECOMBINATION + GB SINKS
    # ==================================================================
    def compute_recombination(self):
        """A_iv = (A_bulk + η²·A_surf) · c_v · c_i."""
        A_r = self.A_bulk + (self.eta ** 2) * self.A_surf
        return A_r * self.c_v * self.c_i, A_r

    def compute_gb_sinks(self):
        """S_v^GB = S_v_r · (1-Φ) · (c_v - c_eq_v)
           S_i^GB = S_i_r · (1-Φ) · (c_i - c_eq_i)
        """
        Phi = self.compute_gb_indicator()
        S_v = self.S_v_r * (1.0 - Phi) * (self.c_v - self.c_eq_v)
        S_i = self.S_i_r * (1.0 - Phi) * (self.c_i - self.c_eq_i)
        return S_v, S_i

    # ==================================================================
    # ELASTOPLASTICITY (eigenstrain from vacancies + J2 plasticity)
    # ==================================================================
    def compute_eigenstrain(self):
        """Eigenstrain from vacancy condensation: ε* = β_v · c_v · I.

        This contracts the lattice in regions with high vacancy concentration
        (e.g., voids), producing a hydrostatic tensile stress around them.
        """
        beta = self.beta_v
        eps_star = beta * self.c_v
        return eps_star, eps_star, np.zeros_like(self.c_v)

    def compute_plasticity_j2(self, sigma_eq, sxx, syy, sxy, h):
        """J2 plasticity with Hall-Petch yield and overstress power-law flow.

        σ_y(r) = σ_y0 + μ·b/(2π·h·(1-ν))·ln(h/b) + H·|ε_p|
        γ̇ = γ̇_0 · ((σ_eq - σ_y)/σ_y)^m  (overstress, Peric-type)
        dε_p/dt = γ̇ · n  where n = (3/2)·S/σ_eq is the J2 flow direction.
        """
        pl = self.mat['plasticity']
        mu_p = pl['mu_p']
        nu_p = pl['nu_p']
        b = pl['b']
        sigma0 = float(self.params.get('sigma0', pl['sigma0']))
        H_hard = float(self.params.get('H', pl['H']))
        gamma0_dot = pl['gamma0_dot']
        m = int(pl['m'])

        # Hall-Petch yield stress (local grain size dependent)
        h_safe = np.maximum(h, 2.001 * b)
        sigma_y_HP = sigma0 + (mu_p * b / (2 * np.pi * h_safe * (1 - nu_p))) * \
                      np.log(h_safe / b)
        sigma_y_HP = np.where(h > 2 * b, sigma_y_HP,
                              sigma0 + mu_p / (2 * np.pi * (1 - nu_p)))
        # Add linear hardening from accumulated plastic strain
        eps_p_mag_old = np.sqrt(2.0 / 3.0 * (self.eps_p_xx ** 2 + self.eps_p_yy ** 2
                                              + 2 * self.eps_p_xy ** 2 + 1e-15))
        sigma_y = sigma_y_HP + H_hard * eps_p_mag_old

        # Overstress and flow rate
        overstress = np.clip((sigma_eq - sigma_y) / np.maximum(sigma_y, 1e-9), 0, 1)
        gamma_dot = gamma0_dot * overstress ** m
        d_inc = np.minimum(2.0 / 3.0 * gamma_dot * self.dt, 1e-3)

        # J2 associated flow direction: n_ij = (3/2)·S_ij/σ_eq
        sigma_eq_safe = np.maximum(sigma_eq, 1e-9)
        n_xx = 1.5 * (sxx - 0.5 * (sxx + syy)) / sigma_eq_safe
        n_yy = 1.5 * (syy - 0.5 * (sxx + syy)) / sigma_eq_safe
        n_xy = 1.5 * sxy / sigma_eq_safe

        self.eps_p_xx = np.clip(self.eps_p_xx + d_inc * n_xx, -0.1, 0.1)
        self.eps_p_yy = np.clip(self.eps_p_yy + d_inc * n_yy, -0.1, 0.1)
        self.eps_p_xy = np.clip(self.eps_p_xy + d_inc * n_xy, -0.1, 0.1)

        eps_p_mag = np.sqrt(2.0 / 3.0 * (self.eps_p_xx ** 2 + self.eps_p_yy ** 2
                                          + 2 * self.eps_p_xy ** 2 + 1e-15))
        return sigma_y, eps_p_mag

    # ==================================================================
    # TIME STEPPING — CAHN-HILLIARD (c_v, c_i)
    # ==================================================================
    def evolve_concentration_v(self, sigma_h_Pa, P_v, A_iv, S_v_GB):
        """Cahn-Hilliard step for vacancies with semi-implicit spectral solve.

        ∂c_v/∂t = ∇·(M_v ∇μ_v) + P_v - A_iv - S_v^GB
        """
        M_v = self.compute_mobility_field_v()
        mu_v = self.compute_chemical_potential_v(sigma_h_Pa)

        # Compute flux divergence: ∇·(M_v ∇μ_v) — done in real space for the
        # variable coefficient M_v, then assembled as the explicit RHS.
        mu_gx, mu_gy = spectral_gradients(mu_v, self.kx, self.ky)
        Jx = -M_v * mu_gx
        Jy = -M_v * mu_gy
        div_J = spectral_divergence(Jx, Jy, self.kx, self.ky)

        # Explicit RHS = flux divergence + source - recombination - GB sink
        rhs = div_J + P_v - A_iv - S_v_GB

        # Semi-implicit correction for bulk diffusion (linear stabilizer)
        # This treats the dominant bulk-diffusion part implicitly, which
        # allows much larger dt than purely explicit.
        D_bulk_v = self.D_v_scalar * self.w_b
        c_expl = self.c_v + self.dt * rhs
        denom = 1.0 + self.dt * D_bulk_v * self.k2
        c_new = np.real(ifft2(fft2(c_expl) / denom))

        # Mass conservation rescale (clip + rescale, like original code)
        c_new = np.clip(c_new, 1e-30, 1.0)
        m_old = float(np.sum(self.c_v))
        m_new = float(np.sum(c_new))
        if m_old > 1e-12 and m_new > 1e-12:
            ratio = m_new / m_old
            if 0.5 < ratio < 2.0:
                c_new *= m_old / m_new
        return np.clip(c_new, 1e-30, 1.0)

    def evolve_concentration_i(self, sigma_h_Pa, P_i, A_iv, S_i_GB):
        """Cahn-Hilliard step for interstitials (analogous to vacancies)."""
        M_i = self.compute_mobility_field_i()
        mu_i = self.compute_chemical_potential_i(sigma_h_Pa)

        mu_gx, mu_gy = spectral_gradients(mu_i, self.kx, self.ky)
        Jx = -M_i * mu_gx
        Jy = -M_i * mu_gy
        div_J = spectral_divergence(Jx, Jy, self.kx, self.ky)

        rhs = div_J + P_i - A_iv - S_i_GB

        D_bulk_i = self.D_i_scalar * self.w_b
        c_expl = self.c_i + self.dt * rhs
        denom = 1.0 + self.dt * D_bulk_i * self.k2
        c_new = np.real(ifft2(fft2(c_expl) / denom))

        c_new = np.clip(c_new, 1e-30, 1.0)
        m_old = float(np.sum(self.c_i))
        m_new = float(np.sum(c_new))
        if m_old > 1e-12 and m_new > 1e-12:
            ratio = m_new / m_old
            if 0.5 < ratio < 2.0:
                c_new *= m_old / m_new
        return np.clip(c_new, 1e-30, 1.0)

    # ==================================================================
    # TIME STEPPING — ALLEN-CAHN (η, φ_i)
    # ==================================================================
    def evolve_eta(self, P_v):
        """Allen-Cahn step for void order parameter eta.

        ∂η/∂t = -L·(δF/δη) + σ_stochastic + P_v
        """
        dF_deta = self.compute_dF_deta()
        # Stochastic thermal noise (small, for nucleation)
        noise = self.rng.normal(0.0, 0.001, size=self.eta.shape) \
                if self.params.get('use_noise', True) else 0.0
        # Explicit RHS: -L·dF/dη + noise + P_v (void-growth source)
        rhs = -self.L_ac * dF_deta + noise + P_v
        eta_expl = self.eta + self.dt * rhs
        # Semi-implicit correction for the κ_η·∇²η term
        denom = 1.0 + self.dt * self.L_ac * self.kappa_eta * self.k2
        eta_new = np.real(ifft2(fft2(eta_expl) / denom))
        return np.clip(eta_new, 0.0, 1.0)

    def evolve_phi_list(self):
        """Allen-Cahn step for all grain-orientation order parameters phi_i
        (with optional artificial driving force f_d).

        dphi_i/dt = -L * (dF/dphi_i + f_d * grad phi_i)
        """
        new_phi_list = []
        for i in range(self.n_grains):
            dF_dphi = self.compute_dF_dphi(i)
            # Artificial driving force (only on the first grain in flat-GB
            # scenarios, f_d pushes the GB)
            if self.geom_meta['has_driving_force'] and i == 0:
                phi_gx, phi_gy = spectral_gradients(self.phi_list[i],
                                                     self.kx, self.ky)
                driving_term = self.f_d_x * phi_gx + self.f_d_y * phi_gy
            else:
                driving_term = 0.0

            rhs = -self.L_ac * (dF_dphi + driving_term)
            phi_expl = self.phi_list[i] + self.dt * rhs
            denom = 1.0 + self.dt * self.L_ac * self.kappa_phi * self.k2
            phi_new = np.real(ifft2(fft2(phi_expl) / denom))
            new_phi_list.append(np.clip(phi_new, 0.0, 1.0))

        # Normalize so Σφ_i = 1 in matrix (where η ≈ 0)
        ssum = np.sum(new_phi_list, axis=0)
        ssum_safe = np.where(ssum > 1e-9, ssum, 1.0)
        new_phi_list = [p / ssum_safe for p in new_phi_list]
        return new_phi_list

    # ==================================================================
    # MAIN STEP
    # ==================================================================
    @handle_errors
    def step(self):
        """Advance the simulation by one time step."""
        # ---- 1. Compute elastoplastic state ----
        # Eigenstrain from vacancy concentration (lattice contraction)
        exx_s, eyy_s, exy_s = self.compute_eigenstrain()
        # Total eigenstrain = vacancy eigenstrain + accumulated plastic strain
        exx_t = exx_s + self.eps_p_xx
        eyy_t = eyy_s + self.eps_p_yy
        exy_t = exy_s + self.eps_p_xy

        # Applied far-field stress (uniform, optionally angle-dependent)
        sig = self.applied_stress
        theta = np.deg2rad(90.0)  # downward (compression along y)
        app_xx = sig * np.cos(theta) ** 2
        app_yy = sig * np.sin(theta) ** 2
        app_xy = sig * np.sin(theta) * np.cos(theta)

        el = self.elastic.solve(exx_t, eyy_t, exy_t, app_xx, app_yy, app_xy)
        if el is FAILED or el is None:
            raise RuntimeError("Elastic solve failed")
        (sigma_eq, sxx, syy, sxy, sigma_h,
         exx_tot, eyy_tot, exy_tot) = el

        # ---- 2. J2 plasticity update ----
        h = self.compute_local_grain_size()
        sigma_y, eps_p_mag = self.compute_plasticity_j2(
            sigma_eq, sxx, syy, sxy, h)

        # ---- 3. Cascade source terms (stochastic) ----
        P_v, P_i = self.compute_cascade_source()

        # ---- 4. Recombination + GB sinks ----
        A_iv, A_r = self.compute_recombination()
        S_v_GB, S_i_GB = self.compute_gb_sinks()

        # ---- 5. Cahn-Hilliard steps for c_v, c_i ----
        # Use σ_h from current step (operator splitting — would normally
        # iterate, but single-step is fine for these timescales).
        self.c_v = self.evolve_concentration_v(sigma_h, P_v, A_iv, S_v_GB)
        self.c_i = self.evolve_concentration_i(sigma_h, P_i, A_iv, S_i_GB)

        # ---- 6. Allen-Cahn for η (with P_v source for void growth) ----
        self.eta = self.evolve_eta(P_v)

        # ---- 7. Allen-Cahn for φ_i (with optional driving force) ----
        self.phi_list = self.evolve_phi_list()

        # ---- 8. Update history ----
        phi_gb = self.compute_gb_indicator()
        n_voids, void_vol = self.count_voids()
        t = (len(self.history['time']) + 1) * self.dt

        self.history['time'].append(t)
        self.history['porosity'].append(self.compute_porosity())
        self.history['mean_c_v'].append(float(np.mean(self.c_v)))
        self.history['mean_c_i'].append(float(np.mean(self.c_i)))
        self.history['gb_area'].append(self.compute_gb_area())
        self.history['max_stress'].append(float(np.max(sigma_eq)))
        self.history['avg_stress'].append(float(np.mean(sigma_eq)))
        self.history['plastic_work'].append(
            float(np.sum(eps_p_mag) * self.dx * self.dy))
        self.history['n_voids'].append(n_voids)
        self.history['void_volume'].append(void_vol)

        # ---- 9. Assemble results dict for visualization ----
        # Build grain map (argmax of phi_list)
        phi_stack = np.stack(self.phi_list, axis=0)
        grain_map_soft = np.argmax(phi_stack, axis=0).astype(float)
        grain_map_soft[phi_stack.max(axis=0) < 0.5] = -1.0

        # Effective diffusivity fields (for visualization)
        D_eff_v = self.compute_mobility_field_v()
        D_eff_i = self.compute_mobility_field_i()

        return {
            'c_v':         self.c_v.copy(),
            'c_i':         self.c_i.copy(),
            'eta':         self.eta.copy(),
            'phi_grain':   grain_map_soft,
            'phi_gb':      phi_gb.copy(),
            'porosity':    (self.eta ** 2).copy(),
            'D_eff_v':     D_eff_v.copy(),
            'D_eff_i':     D_eff_i.copy(),
            'A_recomb':    A_r.copy(),
            'sigma_eq':    sigma_eq.copy(),
            'sigma_h':     sigma_h.copy(),
            'h':           h.copy(),
            'eps_p_mag':   eps_p_mag.copy(),
            'sigma_y':     sigma_y.copy(),
            'sxx':         sxx.copy(),
            'syy':         syy.copy(),
            'sxy':         sxy.copy(),
            'convergence': {
                'porosity':    self.history['porosity'][-1],
                'mean_c_v':    self.history['mean_c_v'][-1],
                'mean_c_i':    self.history['mean_c_i'][-1],
                'gb_area':     self.history['gb_area'][-1],
                'max_stress':  self.history['max_stress'][-1],
                'avg_stress':  self.history['avg_stress'][-1],
                'n_voids':     n_voids,
                'void_volume': void_vol,
            }
        }


# ============================================================================
# LINE PROFILER (re-used from original)
# ============================================================================
class EnhancedLineProfiler:
    def __init__(self, Nx, Ny, dx, dy):
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = dx, dy
        self.extent = [-Nx * dx / 2, Nx * dx / 2, -Ny * dy / 2, Ny * dy / 2]

    @handle_errors
    def extract_profile(self, data, profile_type='horizontal',
                        position_ratio=0.5, angle_deg=45):
        ptype = str(profile_type).lower().replace('-', '_')
        ny, nx = data.shape
        cx, cy = nx // 2, ny // 2
        off_x = int(nx * 0.35 * position_ratio)
        off_y = int(ny * 0.35 * position_ratio)

        if ptype == 'horizontal':
            r = min(max(cy + off_y, 0), ny - 1)
            return (np.linspace(self.extent[0], self.extent[1], nx),
                    data[r, :], (self.extent[0], r * self.dy + self.extent[2],
                                 self.extent[1], r * self.dy + self.extent[2]))
        if ptype == 'vertical':
            cidx = min(max(cx + off_x, 0), nx - 1)
            return (np.linspace(self.extent[2], self.extent[3], ny),
                    data[:, cidx], (cidx * self.dx + self.extent[0], self.extent[2],
                                     cidx * self.dx + self.extent[0], self.extent[3]))
        if ptype == 'diagonal':
            L = int(min(nx, ny) * 0.8)
            s = (cx - L // 2, cy - L // 2)
            vals, ds = [], []
            for i in range(L):
                x, y = s[0] + i, s[1] + i
                if 0 <= x < nx and 0 <= y < ny:
                    vals.append(data[y, x])
                    ds.append((i - L // 2) * self.dx * np.sqrt(2))
            return np.array(ds), np.array(vals), None
        if ptype in ('anti_diagonal', 'antidiagonal'):
            L = int(min(nx, ny) * 0.8)
            s = (cx + L // 2, cy - L // 2)
            vals, ds = [], []
            for i in range(L):
                x, y = s[0] - i, s[1] + i
                if 0 <= x < nx and 0 <= y < ny:
                    vals.append(data[y, x])
                    ds.append((i - L // 2) * self.dx * np.sqrt(2))
            return np.array(ds), np.array(vals), None
        if ptype == 'custom':
            ar = np.deg2rad(angle_deg)
            L = int(min(nx, ny) * 0.8)
            vals, ds = [], []
            for t in np.linspace(-L // 2, L // 2, L):
                x = cx + t * np.cos(ar) + off_y * np.cos(ar + np.pi / 2)
                y = cy + t * np.sin(ar) + off_y * np.sin(ar + np.pi / 2)
                if 0 <= x < nx - 1 and 0 <= y < ny - 1:
                    x0, y0 = int(x), int(y)
                    x1, y1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1)
                    wx, wy = x - x0, y - y0
                    v = (data[y0, x0] * (1 - wx) * (1 - wy)
                         + data[y0, x1] * wx * (1 - wy)
                         + data[y1, x0] * (1 - wx) * wy
                         + data[y1, x1] * wx * wy)
                    vals.append(v); ds.append(t * self.dx)
            return np.array(ds), np.array(vals), None
        return None, None, None

    @handle_errors
    def extract_profile_at(self, data, x0, y0, x1, y1, n=200):
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        i = ((xs - self.extent[0]) / self.dx).astype(float)
        j = ((ys - self.extent[2]) / self.dy).astype(float)
        i = np.clip(i, 0, self.Nx - 1)
        j = np.clip(j, 0, self.Ny - 1)
        i0 = i.astype(int); j0 = j.astype(int)
        i1 = np.clip(i0 + 1, 0, self.Nx - 1)
        j1 = np.clip(j0 + 1, 0, self.Ny - 1)
        wi = i - i0
        wj = j - j0
        v = (data[j0, i0] * (1 - wi) * (1 - wj)
             + data[j0, i1] * wi * (1 - wj)
             + data[j1, i0] * (1 - wi) * wj
             + data[j1, i1] * wi * wj)
        dist = np.sqrt((xs - x0) ** 2 + (ys - y0) ** 2)
        return dist, v


# ============================================================================
# FIELD RESCALING HELPERS (UO2-specific)
# ============================================================================
def _rescale_field(field_name, data):
    """Apply unit conversion for display. Stresses are stored in Pa internally;
    convert to GPa for display. σ_y is in Pa → MPa. Other fields are dimensionless."""
    d = data.copy()
    if field_name in ('sigma_eq', 'sigma_h', 'sxx', 'syy', 'sxy'):
        d = d / 1e9     # Pa → GPa
    elif field_name == 'sigma_y':
        d = d / 1e6     # Pa → MPa
    return d, FIELD_UNITS.get(field_name, '')


# ============================================================================
# NORM + BOUNDS (re-used from original, with UO2 default ranges)
# ============================================================================
def compute_field_norm(field_name, data, fs, default_rng=None):
    if fs is None:
        fs = {}
    auto = fs.get('auto', True)
    use_log = fs.get('log', False)

    d = np.asarray(data)
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return None, None, None

    if auto:
        if default_rng is not None:
            vmin, vmax = float(default_rng[0]), float(default_rng[1])
        else:
            vmin = float(np.percentile(finite, 2))
            vmax = float(np.percentile(finite, 98))
            if field_name == 'sigma_h':
                vmax = max(abs(vmin), abs(vmax)); vmin = -vmax
    else:
        vmin = fs.get('vmin', None)
        vmax = fs.get('vmax', None)
        if vmin is None:
            vmin = (float(default_rng[0]) if default_rng is not None
                    else float(np.min(finite)))
        if vmax is None:
            vmax = (float(default_rng[1]) if default_rng is not None
                    else float(np.max(finite)))

    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-9

    norm = None
    if use_log:
        pos = finite[finite > 0]
        if pos.size > 0:
            if vmin <= 0: vmin = float(np.percentile(pos, 2))
            if vmax <= 0: vmax = float(np.percentile(pos, 98))
            if vmin >= vmax: vmin = vmax / 10.0
            try:
                norm = LogNorm(vmin=vmin, vmax=vmax)
            except Exception:
                norm = None

    return norm, vmin, vmax


def get_global_bounds(history, field_name, fs, default_rng=None):
    if fs is None:
        fs = {}

    if not fs.get('auto', True):
        vmin = fs.get('vmin', None)
        vmax = fs.get('vmax', None)
        if vmin is None:
            vmin = (float(default_rng[0]) if default_rng is not None else 0.0)
        if vmax is None:
            vmax = (float(default_rng[1]) if default_rng is not None else 1.0)
        return float(vmin), float(vmax)

    if default_rng is not None:
        return float(default_rng[0]), float(default_rng[1])

    vmin_all, vmax_all = [], []
    for h in history:
        if field_name not in h:
            continue
        dd, _ = _rescale_field(field_name, h[field_name])
        finite = dd[np.isfinite(dd)]
        if finite.size:
            vmin_all.append(np.percentile(finite, 2))
            vmax_all.append(np.percentile(finite, 98))
    if not vmin_all:
        return None, None

    vmin = float(np.min(vmin_all))
    vmax = float(np.max(vmax_all))
    if field_name == 'sigma_h':
        vmax = max(abs(vmin), abs(vmax)); vmin = -vmax
    return vmin, vmax


# ============================================================================
# VECTOR FIELD COMPUTATION
# ============================================================================
def compute_vector_field(results_dict, source, mode='gradient'):
    if mode == 'principal_stress':
        sxx = results_dict.get('sxx')
        syy = results_dict.get('syy')
        sxy = results_dict.get('sxy')
        if sxx is None or syy is None or sxy is None:
            return None, None
        theta = 0.5 * np.arctan2(2 * sxy, sxx - syy)
        vx = np.cos(theta)
        vy = np.sin(theta)
        return vx, vy

    f = results_dict.get(source)
    if f is None:
        return None, None

    gy, gx = np.gradient(f)
    if mode == 'flux':
        D = results_dict.get('D_eff_v') or results_dict.get('D_eff_i')
        if D is None:
            D = np.ones_like(f)
        vx = -D * gx
        vy = -D * gy
    else:
        vx, vy = gx, gy
    return vx, vy


# ============================================================================
# UO2 VISUALIZER
# ============================================================================
class EnhancedUO2Visualizer:
    """Visualization routines for the UO2 void-GB phase-field solver.

    Provides:
      - create_multi_field_comparison : grid of all 14 fields (c_v, c_i, η,
        grain map, GB indicator, porosity, D_eff_v, D_eff_i, A_recomb,
        σ_eq, σ_h, h, ε_p, σ_y).
      - create_overlay                 : base + contour/heatmap overlay.
      - create_vector_overlay          : quiver over base field.
      - create_plotly_heatmap          : interactive Plotly heatmap.
      - create_plotly_3d_surface       : 3D surface of one field.
      - create_plotly_animation        : time-scrubbing animation.
      - create_animation_buffer        : matplotlib GIF/MP4 writer.
    """

    def __init__(self, Nx, Ny, dx, dy=None, dt=1e-2):
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = dx, (dy if dy is not None else dx)
        self.dt = dt
        self.extent = [-Nx * self.dx / 2, Nx * self.dx / 2,
                       -Ny * self.dy / 2, Ny * self.dy / 2]
        self.line_profiler = EnhancedLineProfiler(Nx, Ny, dx, self.dy)

    def get_colormap(self, name, invert=False):
        return ColormapRegistry.resolve(name, invert=invert)

    # ------------------------------------------------------------------
    @handle_errors
    def create_multi_field_comparison(self, results_dict, style_params=None):
        if style_params is None:
            style_params = {}
        colormaps = style_params.get('colormaps', {})
        field_settings = style_params.get('field_settings', {})
        title_fs = float(style_params.get('title_font_size', 10))
        label_fs = float(style_params.get('label_font_size', 8))
        cbar_fraction = float(style_params.get('colorbar_fraction', 0.046))
        cbar_pad = float(style_params.get('colorbar_pad', 0.04))
        fig_cfg = style_params.get('figure_customization', {}) or {}

        # UO2 field list (matches FIELD_DEFAULT_CMAPS keys)
        fields = [
            ('c_v',       'c_v (vacancy concentration)',    (1e-6, 1.0)),
            ('c_i',       'c_i (interstitial concentration)', (1e-9, 1.0)),
            ('eta',       'η (void order parameter)',       (0, 1)),
            ('phi_grain', 'Grain ID map (argmax φ_i)',       None),
            ('phi_gb',    'φ_gb (GB indicator)',             (0, 1)),
            ('porosity',  'Porosity (η²)',                   (0, 1)),
            ('D_eff_v',   'D_eff_v (vacancy diffusivity)',   None),
            ('D_eff_i',   'D_eff_i (interstitial diffusivity)', None),
            ('A_recomb',  'A_iv (recombination rate)',       None),
            ('sigma_eq',  'σ_eq (GPa)',                       None),
            ('sigma_h',   'σ_h (GPa)',                        None),
            ('h',         'Local grain size (nm)',            None),
            ('eps_p_mag', '|ε_p| (plastic strain)',           None),
            ('sigma_y',   'σ_y (MPa)',                        None),
        ]
        avail = [(f, t, r) for (f, t, r) in fields if f in results_dict]
        if not avail:
            return None

        n = len(avail)
        cols = 3 if n >= 3 else n
        rows = (n + cols - 1) // cols
        fig_w = float(fig_cfg.get('multi_field_width', 4.5 * cols))
        fig_h = float(fig_cfg.get('multi_field_height', 3.8 * rows))
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
        axes = np.array(axes).flatten()

        show_cb = bool(fig_cfg.get('show_colorbar', True))
        cb_orient = fig_cfg.get('colorbar_orientation', 'vertical')

        for idx, (fname, title, rng) in enumerate(avail):
            ax = axes[idx]
            d, unit = _rescale_field(fname, results_dict[fname])

            fs = field_settings.get(fname, {})
            cmap_name = (colormaps.get(fname)
                         or fs.get('cmap')
                         or style_params.get(f'{fname}_cmap')
                         or FIELD_DEFAULT_CMAPS.get(fname, 'viridis'))
            cmap = self.get_colormap(cmap_name, invert=fs.get('invert', False))

            interp = 'nearest' if fname in DISCRETE_FIELDS else 'bilinear'

            if fname == 'phi_grain':
                im = ax.imshow(d, extent=self.extent, cmap=cmap,
                               origin='lower', aspect='equal',
                               interpolation='nearest')
                t_override = fig_cfg.get('title_override', '').strip()
                ax.set_title(t_override if t_override else title,
                             fontsize=title_fs)
                xl = fig_cfg.get('xlabel_override', '').strip() or 'x (nm)'
                yl = fig_cfg.get('ylabel_override', '').strip() or 'y (nm)'
                ax.set_xlabel(xl, fontsize=label_fs)
                ax.set_ylabel(yl, fontsize=label_fs)
                apply_axis_styling(ax, fig_cfg, grid_axes=None)
                continue

            default_rng = FIELD_DEFAULT_RNG.get(fname, rng)
            norm, vmin, vmax = compute_field_norm(fname, d, fs,
                                                   default_rng=default_rng)

            if norm is not None:
                im = ax.imshow(d, extent=self.extent, cmap=cmap, norm=norm,
                               origin='lower', aspect='equal', interpolation=interp)
            else:
                im = ax.imshow(d, extent=self.extent, cmap=cmap,
                               vmin=vmin, vmax=vmax,
                               origin='lower', aspect='equal', interpolation=interp)
            t_override = fig_cfg.get('title_override', '').strip()
            ax.set_title(t_override if t_override else title, fontsize=title_fs)
            xl = fig_cfg.get('xlabel_override', '').strip() or 'x (nm)'
            yl = fig_cfg.get('ylabel_override', '').strip() or 'y (nm)'
            ax.set_xlabel(xl, fontsize=label_fs)
            ax.set_ylabel(yl, fontsize=label_fs)
            if show_cb:
                cbar = plt.colorbar(im, ax=ax, fraction=cbar_fraction,
                                     pad=cbar_pad, orientation=cb_orient)
                if unit:
                    cbar.set_label(unit)
            apply_axis_styling(ax, fig_cfg, grid_axes=None)

        for k in range(n, len(axes)):
            axes[k].axis('off')
        if fig_cfg.get('tight_layout', True):
            plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    @handle_errors
    def create_overlay(self, results_dict, base_field, overlay_field,
                       alpha=0.5, mode='contour', n_contours=10,
                       show_grain_ghost=False, grain_map=None,
                       style_params=None):
        if style_params is None:
            style_params = {}
        field_settings = style_params.get('field_settings', {})
        colormaps = style_params.get('colormaps', {})
        title_fs = float(style_params.get('title_font_size', 11))
        label_fs = float(style_params.get('label_font_size', 9))
        fig_cfg = style_params.get('figure_customization', {}) or {}

        if base_field not in results_dict or overlay_field not in results_dict:
            return None

        base, base_unit = _rescale_field(base_field, results_dict[base_field])
        ov, ov_unit = _rescale_field(overlay_field, results_dict[overlay_field])

        base_fs = field_settings.get(base_field, {})
        ov_fs = field_settings.get(overlay_field, {})

        base_cmap_name = (colormaps.get(base_field)
                           or base_fs.get('cmap')
                           or FIELD_DEFAULT_CMAPS.get(base_field, 'viridis'))
        base_cmap = self.get_colormap(base_cmap_name,
                                       invert=base_fs.get('invert', False))

        fig_w = float(fig_cfg.get('overlay_width', 11.0))
        fig_h = float(fig_cfg.get('overlay_height', 5.5))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        if base_field in DISCRETE_FIELDS:
            im_base = ax.imshow(base, extent=self.extent, cmap=base_cmap,
                                origin='lower', aspect='equal',
                                interpolation='nearest')
        else:
            norm, vmin, vmax = compute_field_norm(
                base_field, base, base_fs,
                default_rng=FIELD_DEFAULT_RNG.get(base_field))
            if norm is not None:
                im_base = ax.imshow(base, extent=self.extent, cmap=base_cmap,
                                    norm=norm, origin='lower', aspect='equal',
                                    interpolation='bilinear')
            else:
                im_base = ax.imshow(base, extent=self.extent, cmap=base_cmap,
                                    vmin=vmin, vmax=vmax,
                                    origin='lower', aspect='equal',
                                    interpolation='bilinear')

        if show_grain_ghost and grain_map is not None:
            ghost = np.where(grain_map < 0, np.nan, grain_map)
            ax.imshow(ghost, extent=self.extent,
                      cmap=ColormapRegistry.resolve('uo2_grain_map'),
                      origin='lower', aspect='equal', interpolation='nearest',
                      alpha=0.25 * alpha)

        contour_lw = float(fig_cfg.get('contour_line_width', 1.0))
        if mode == 'contour':
            ov_cmap_name = (colormaps.get(overlay_field)
                             or ov_fs.get('cmap')
                             or FIELD_DEFAULT_CMAPS.get(overlay_field, 'viridis'))
            ov_cmap = self.get_colormap(ov_cmap_name,
                                         invert=ov_fs.get('invert', False))
            x = np.linspace(self.extent[0], self.extent[1], self.Nx)
            y = np.linspace(self.extent[2], self.extent[3], self.Ny)
            X, Y = np.meshgrid(x, y)
            ov_for_contour = ov.copy()
            ov_for_contour[~np.isfinite(ov_for_contour)] = np.nan
            _, ov_vmin, ov_vmax = compute_field_norm(
                overlay_field, ov, ov_fs,
                default_rng=FIELD_DEFAULT_RNG.get(overlay_field))
            try:
                if ov_vmin is None or ov_vmax is None or ov_vmax <= ov_vmin:
                    levels = n_contours
                else:
                    levels = np.linspace(ov_vmin, ov_vmax, n_contours)
                cs = ax.contour(X, Y, ov_for_contour, levels=levels,
                                cmap=ov_cmap, linewidths=contour_lw, alpha=alpha)
                ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f')
            except Exception as e:
                logger.warning(f"Overlay contour failed: {e}")
        else:
            ov_cmap_name = (colormaps.get(overlay_field)
                             or ov_fs.get('cmap')
                             or FIELD_DEFAULT_CMAPS.get(overlay_field, 'viridis'))
            ov_cmap = self.get_colormap(ov_cmap_name,
                                         invert=ov_fs.get('invert', False))
            _, ov_vmin, ov_vmax = compute_field_norm(
                overlay_field, ov, ov_fs,
                default_rng=FIELD_DEFAULT_RNG.get(overlay_field))
            ax.imshow(ov, extent=self.extent, cmap=ov_cmap,
                      vmin=ov_vmin, vmax=ov_vmax,
                      origin='lower', aspect='equal',
                      interpolation='bilinear', alpha=alpha)

        if fig_cfg.get('show_colorbar', True):
            cb_orient = fig_cfg.get('colorbar_orientation', 'vertical')
            cbar = plt.colorbar(im_base, ax=ax, fraction=0.046, pad=0.04,
                                  orientation=cb_orient)
            if base_unit:
                cbar.set_label(f"{base_field} ({base_unit})")

        default_title = (
            f"Overlay: {FIELD_LABELS.get(base_field, base_field)} + "
            f"{FIELD_LABELS.get(overlay_field, overlay_field)} "
            f"(mode={mode}, α={alpha:.2f})")
        apply_axis_styling(ax, fig_cfg,
                            title=default_title, title_fs=title_fs,
                            xlabel='x (nm)', ylabel='y (nm)',
                            label_fs=label_fs)
        if fig_cfg.get('tight_layout', True):
            plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    @handle_errors
    def create_vector_overlay(self, results_dict, source, mode='gradient',
                              density=24, color='white', scale=1.0,
                              base_field=None, style_params=None):
        if style_params is None:
            style_params = {}
        field_settings = style_params.get('field_settings', {})
        colormaps = style_params.get('colormaps', {})
        title_fs = float(style_params.get('title_font_size', 11))
        label_fs = float(style_params.get('label_font_size', 9))
        fig_cfg = style_params.get('figure_customization', {}) or {}

        vx, vy = compute_vector_field(results_dict, source, mode=mode)
        if vx is None:
            return None

        fig_w = float(fig_cfg.get('vector_width', 11.0))
        fig_h = float(fig_cfg.get('vector_height', 5.5))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        if base_field and base_field in results_dict:
            base, base_unit = _rescale_field(base_field, results_dict[base_field])
            base_fs = field_settings.get(base_field, {})
            base_cmap_name = (colormaps.get(base_field)
                               or base_fs.get('cmap')
                               or FIELD_DEFAULT_CMAPS.get(base_field, 'viridis'))
            base_cmap = self.get_colormap(base_cmap_name,
                                           invert=base_fs.get('invert', False))
            norm, vmin, vmax = compute_field_norm(
                base_field, base, base_fs,
                default_rng=FIELD_DEFAULT_RNG.get(base_field))
            if norm is not None:
                im = ax.imshow(base, extent=self.extent, cmap=base_cmap,
                               norm=norm, origin='lower', aspect='equal',
                               interpolation='bilinear')
            else:
                im = ax.imshow(base, extent=self.extent, cmap=base_cmap,
                               vmin=vmin, vmax=vmax,
                               origin='lower', aspect='equal',
                               interpolation='bilinear')
            if fig_cfg.get('show_colorbar', True):
                cb_orient = fig_cfg.get('colorbar_orientation', 'vertical')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                             orientation=cb_orient,
                             label=f"{base_field} ({base_unit})")
        else:
            ax.set_facecolor('#111111')

        step = max(1, self.Nx // max(density, 1))
        x = np.linspace(self.extent[0], self.extent[1], self.Nx)[::step]
        y = np.linspace(self.extent[2], self.extent[3], self.Ny)[::step]
        X, Y = np.meshgrid(x, y)
        U = vx[::step, ::step] * scale
        V = vy[::step, ::step] * scale

        mag = np.sqrt(U ** 2 + V ** 2) + 1e-12
        U_n = U / mag
        V_n = V / mag

        q_width = float(fig_cfg.get('quiver_width', 0.0025))
        q_scale = float(fig_cfg.get('quiver_scale', 30.0))
        ax.quiver(X, Y, U_n, V_n, mag, cmap='autumn', alpha=0.9,
                  scale=q_scale, width=q_width, pivot='mid')

        apply_axis_styling(ax, fig_cfg,
                            title=f"Vector field: {mode} of {source}",
                            title_fs=title_fs,
                            xlabel='x (nm)', ylabel='y (nm)',
                            label_fs=label_fs)
        if fig_cfg.get('tight_layout', True):
            plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    @handle_errors
    def create_plotly_heatmap(self, results_dict, field_name, frame_idx=0,
                              cmap_name=None, zmid=None, viz_cfg=None):
        if field_name not in results_dict:
            return None
        viz_cfg = viz_cfg or {}
        field_settings = viz_cfg.get('field_settings', {})
        fs = field_settings.get(field_name, {})

        d, unit = _rescale_field(field_name, results_dict[field_name])
        chosen = (cmap_name
                  or fs.get('cmap')
                  or FIELD_DEFAULT_CMAPS.get(field_name, 'viridis'))
        colorscale = ColormapRegistry.to_plotly(chosen, invert=fs.get('invert', False))

        norm, vmin, vmax = compute_field_norm(
            field_name, d, fs,
            default_rng=FIELD_DEFAULT_RNG.get(field_name))

        z = d
        if fs.get('log', False):
            pos = d > 0
            z = np.where(pos, np.log10(np.maximum(d, 1e-30)), np.nan)

        if zmid is None and field_name == 'sigma_h':
            zmid = 0

        fig_cfg = viz_cfg.get('figure_customization', {})
        p_width = int(fig_cfg.get('overlay_width', 11.0) * 60)
        p_height = int(fig_cfg.get('overlay_height', 5.5) * 60)
        xl = fig_cfg.get('xlabel_override', '').strip() or 'x (nm)'
        yl = fig_cfg.get('ylabel_override', '').strip() or 'y (nm)'
        title_ov = fig_cfg.get('title_override', '').strip()
        title_str = title_ov if title_ov else f"{field_name} — Frame {frame_idx} ({chosen})"

        fig = go.Figure()
        show_cb = bool(fig_cfg.get('show_colorbar', True))
        cb_dict = dict(title=f"{field_name} {unit}") if show_cb else None
        fig.add_trace(go.Heatmap(
            z=z,
            x=np.linspace(self.extent[0], self.extent[1], self.Nx),
            y=np.linspace(self.extent[2], self.extent[3], self.Ny),
            colorscale=colorscale, zmid=zmid,
            zmin=(np.log10(vmin) if fs.get('log') and vmin and vmin > 0 else vmin),
            zmax=(np.log10(vmax) if fs.get('log') and vmax and vmax > 0 else vmax),
            colorbar=cb_dict,
            hovertemplate='x: %{x:.1f} nm<br>y: %{y:.1f} nm<br>%{z:.4f}<extra></extra>'))
        fig.update_layout(title=title_str,
                          xaxis_title=xl, yaxis_title=yl,
                          yaxis=dict(scaleanchor="x", scaleratio=self.dy / self.dx),
                          width=max(400, p_width), height=max(300, p_height),
                          template="plotly_white")
        return fig

    # ------------------------------------------------------------------
    @handle_errors
    def create_plotly_3d_surface(self, results_dict, field_name, frame_idx=0,
                                 cmap_name=None, show_contours=True,
                                 viz_cfg=None):
        if field_name not in results_dict:
            return None
        viz_cfg = viz_cfg or {}
        field_settings = viz_cfg.get('field_settings', {})
        fs = field_settings.get(field_name, {})

        d, unit = _rescale_field(field_name, results_dict[field_name])
        chosen = (cmap_name
                  or fs.get('cmap')
                  or FIELD_DEFAULT_CMAPS.get(field_name, 'viridis'))
        colorscale = ColormapRegistry.to_plotly(chosen, invert=fs.get('invert', False))
        x = np.linspace(self.extent[0], self.extent[1], self.Nx)
        y = np.linspace(self.extent[2], self.extent[3], self.Ny)
        X, Y = np.meshgrid(x, y)

        norm, vmin, vmax = compute_field_norm(
            field_name, d, fs,
            default_rng=FIELD_DEFAULT_RNG.get(field_name))

        contours = None
        if show_contours:
            contours = dict(
                z=dict(show=True, usecolormap=True, project_z=True,
                       highlightcolor="limegreen"))

        fig = go.Figure(data=[go.Surface(
            z=d, x=X, y=Y, colorscale=colorscale, contours=contours,
            cmin=vmin, cmax=vmax,
            colorbar=dict(title=f"{field_name} {unit}"))])
        fig.update_layout(title=f"3D: {field_name} — Frame {frame_idx}",
                          scene=dict(xaxis_title='x (nm)', yaxis_title='y (nm)',
                                     zaxis_title=f'{field_name} {unit}',
                                     aspectmode='manual',
                                     aspectratio=dict(x=2, y=1, z=0.6)),
                          width=800, height=600, template="plotly_white")
        return fig

    # ------------------------------------------------------------------
    @handle_errors
    def create_plotly_line_profiles(self, results_dict, field_name, profile_types,
                                    position_ratio=0.5, viz_cfg=None):
        fig = go.Figure()
        d, unit = _rescale_field(field_name, results_dict[field_name])
        for pt in profile_types:
            res = self.line_profiler.extract_profile(d, pt, position_ratio)
            if res is FAILED or res is None:
                continue
            dist, prof, _ = res
            if dist is None:
                continue
            fig.add_trace(go.Scatter(x=dist, y=prof, mode='lines',
                                     name=pt.replace('_', ' ').title()))
        fig.update_layout(title=f"{field_name} Line Profiles",
                          xaxis_title="Position (nm)",
                          yaxis_title=f"{field_name} {unit}",
                          hovermode='x unified', template="plotly_white")
        return fig

    # ------------------------------------------------------------------
    @handle_errors
    def create_multi_time_profiles(self, history, field_name, ptype='horizontal',
                                   frames=None, position_ratio=0.5,
                                   use_plotly=False, viz_cfg=None):
        if frames is None:
            frames = list(range(len(history)))
        frames = [f for f in frames if 0 <= f < len(history)]
        if not frames:
            return None

        viz_cfg = viz_cfg or {}
        fig_cfg = viz_cfg.get('figure_customization', {}) or {}

        if use_plotly:
            fig = go.Figure()
            lw = float(fig_cfg.get('line_width', 2.0))
            for f in frames:
                d, unit = _rescale_field(field_name, history[f][field_name])
                res = self.line_profiler.extract_profile(d, ptype, position_ratio)
                if res is FAILED or res is None:
                    continue
                dist, prof, _ = res
                if dist is None:
                    continue
                fig.add_trace(go.Scatter(x=dist, y=prof, mode='lines',
                                         name=f"Frame {f}",
                                         line=dict(width=lw)))
            p_width = int(float(fig_cfg.get('line_profile_width', 11.0)) * 60)
            p_height = int(float(fig_cfg.get('line_profile_height', 4.5)) * 60)
            xl = fig_cfg.get('xlabel_override', '').strip() or 'Position (nm)'
            yl = fig_cfg.get('ylabel_override', '').strip() or f'{field_name}'
            title_ov = fig_cfg.get('title_override', '').strip()
            title_str = title_ov if title_ov else f"{field_name} — multi-time {ptype}"
            fig.update_layout(title=title_str,
                              xaxis_title=xl, yaxis_title=yl,
                              hovermode='x unified', template="plotly_white",
                              width=max(400, p_width), height=max(300, p_height))
            return fig
        else:
            fig_w = float(fig_cfg.get('line_profile_width', 11.0))
            fig_h = float(fig_cfg.get('line_profile_height', 4.5))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            cmap = plt.cm.viridis
            n = len(frames)
            lw = float(fig_cfg.get('line_width', 2.0))
            for k, f in enumerate(frames):
                d, unit = _rescale_field(field_name, history[f][field_name])
                res = self.line_profiler.extract_profile(d, ptype, position_ratio)
                if res is FAILED or res is None:
                    continue
                dist, prof, _ = res
                if dist is None:
                    continue
                color = cmap(k / max(n - 1, 1))
                ax.plot(dist, prof, color=color, lw=lw, label=f"Frame {f}")
            apply_axis_styling(ax, fig_cfg,
                                title=f'{field_name} — multi-time {ptype}',
                                xlabel='Position (nm)',
                                ylabel=f'{field_name}')
            ax.legend(loc='best', fontsize=8)
            if fig_cfg.get('tight_layout', True):
                plt.tight_layout()
            return fig

    # ------------------------------------------------------------------
    @handle_errors
    def create_plotly_animation(self, history, field_name, dt,
                                cmap_name=None, viz_cfg=None):
        if not history:
            return None
        viz_cfg = viz_cfg or {}
        field_settings = viz_cfg.get('field_settings', {})
        fs = field_settings.get(field_name, {})

        chosen = (cmap_name
                  or fs.get('cmap')
                  or FIELD_DEFAULT_CMAPS.get(field_name, 'viridis'))
        colorscale = ColormapRegistry.to_plotly(chosen, invert=fs.get('invert', False))

        zmin, zmax = get_global_bounds(
            history, field_name, fs,
            default_rng=FIELD_DEFAULT_RNG.get(field_name))

        if field_name == 'phi_grain':
            zmin, zmax = None, None

        x = np.linspace(self.extent[0], self.extent[1], self.Nx)
        y = np.linspace(self.extent[2], self.extent[3], self.Ny)

        frames = []
        for i, h in enumerate(history):
            d, _ = _rescale_field(field_name, h[field_name])
            frames.append(go.Frame(
                data=[go.Heatmap(z=d, x=x, y=y, colorscale=colorscale,
                                 zmin=zmin, zmax=zmax,
                                 colorbar=dict(
                                     title=f"{field_name} "
                                           f"{FIELD_UNITS.get(field_name, '')}"))],
                name=f"t={i * dt:.4f}"))

        fig = go.Figure(data=frames[0].data, frames=frames)
        fig.update_layout(
            title=f"{field_name} — interactive time scrubbing",
            xaxis_title="x (nm)", yaxis_title="y (nm)",
            yaxis=dict(scaleanchor="x", scaleratio=self.dy / self.dx),
            width=850, height=600, template="plotly_white",
            updatemenus=[dict(
                type="buttons", showactive=False,
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=150, redraw=True),
                                          fromcurrent=True,
                                          transition=dict(duration=0))]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode="immediate")]),
                ])],
            sliders=[dict(
                steps=[dict(method="animate",
                            args=[[f.name],
                                  dict(mode="immediate",
                                       frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))],
                            label=f.name)
                       for f in frames],
                x=0.05, len=0.9, xanchor="left", y=0, yanchor="top",
                currentvalue=dict(prefix="Frame: ", visible=True,
                                  xanchor="right"))],
        )
        return fig

    # ------------------------------------------------------------------
    @handle_errors
    def create_animation_buffer(self, history, field_name, dt,
                                output_format='gif', fps=5, dpi=120,
                                cmap_name=None, watermark=False,
                                watermark_text='', watermark_pos='lower right',
                                viz_cfg=None):
        if not history:
            return None
        viz_cfg = viz_cfg or {}
        field_settings = viz_cfg.get('field_settings', {})
        fs = field_settings.get(field_name, {})

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)
        ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
        cmap = ColormapRegistry.resolve(
            cmap_name
            or fs.get('cmap')
            or FIELD_DEFAULT_CMAPS.get(field_name, 'viridis'),
            invert=fs.get('invert', False))

        d0, unit = _rescale_field(field_name, history[0][field_name])

        vmin, vmax = get_global_bounds(
            history, field_name, fs,
            default_rng=FIELD_DEFAULT_RNG.get(field_name))
        if field_name == 'phi_grain':
            vmin, vmax = None, None

        interp = 'nearest' if field_name in DISCRETE_FIELDS else 'bilinear'
        im = ax.imshow(d0, extent=self.extent, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin='lower', interpolation=interp, aspect='auto')
        if field_name != 'phi_grain':
            cbar = plt.colorbar(im, ax=ax); cbar.set_label(f"{field_name} {unit}")
        t_unit = 'τ'
        title = ax.set_title(f"{field_name} — t = 0 {t_unit}")

        wm_artist = None
        if watermark and watermark_text:
            xmap = {'lower right': (0.98, 0.02, 'right', 'bottom'),
                    'lower left':  (0.02, 0.02, 'left',  'bottom'),
                    'upper right': (0.98, 0.98, 'right', 'top'),
                    'upper left':  (0.02, 0.98, 'left',  'top')}
            x, y, ha, va = xmap.get(watermark_pos, (0.98, 0.02, 'right', 'bottom'))
            wm_artist = ax.text(x, y, watermark_text, transform=ax.transAxes,
                                fontsize=8, color='white', alpha=0.75,
                                ha=ha, va=va,
                                bbox=dict(boxstyle='round,pad=0.25',
                                          facecolor='black', alpha=0.35,
                                          edgecolor='none'))

        def update(i):
            d, _ = _rescale_field(field_name, history[i][field_name])
            im.set_array(d)
            title.set_text(f"{field_name} — t = {i * dt:.4f} {t_unit}")
            return ([im, title] if wm_artist is None else [im, title, wm_artist])

        ani = animation.FuncAnimation(fig, update, frames=len(history),
                                       interval=1000 / max(fps, 1), blit=True)
        buf = BytesIO()
        try:
            if output_format == 'gif':
                # Newer matplotlib PillowWriter requires a file path, not a
                # file-like object. Use a NamedTemporaryFile as an intermediary.
                import tempfile, os
                with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tf:
                    tmp_path = tf.name
                try:
                    ani.save(tmp_path, writer='pillow', fps=fps, dpi=dpi)
                    with open(tmp_path, 'rb') as f:
                        buf.write(f.read())
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
            else:
                # ffmpeg writer accepts file-like objects
                ani.save(buf, writer='ffmpeg', fps=fps, dpi=dpi)
        except Exception as e:
            logger.error(f"Animation export failed: {e}")
            plt.close(fig)
            return None
        plt.close(fig)
        buf.seek(0)
        return buf


# ============================================================================
# ANIMATION HELPER (backward compat)
# ============================================================================
@handle_errors
def make_animation(history, field_name, dt, extent, Nx, Ny, output_format='gif',
                   fps=5, dpi=120, cmap_name=None, watermark=False,
                   watermark_text='', watermark_pos='lower right'):
    vis = EnhancedUO2Visualizer(Nx, Ny,
                                 (extent[1] - extent[0]) / max(Nx - 1, 1),
                                 (extent[3] - extent[2]) / max(Ny - 1, 1),
                                 dt=dt)
    return vis.create_animation_buffer(
        history, field_name, dt, output_format=output_format, fps=fps, dpi=dpi,
        cmap_name=cmap, watermark=watermark,
        watermark_text=watermark_text, watermark_pos=watermark_pos)


# ============================================================================
# EXPORTERS
# ============================================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


class DataExporter:
    """Export simulation results in various formats:
      - PKL  : full pickled state (params + history + metadata)
      - NPZ  : NumPy archive of all fields per frame
      - JSON : human-readable (large but portable)
      - CSV  : per-frame CSV zipped together
      - HDF5 : hierarchical binary (if h5py available)
      - PT   : PyTorch tensors (if torch available)
    """

    @staticmethod
    @handle_errors
    def export_pkl(sim_data, params, history, name):
        buf = BytesIO()
        pickle.dump({'params': params, 'history': history,
                     'metadata': sim_data.get('metadata', {})}, buf)
        buf.seek(0)
        return buf, f"{name}.pkl"

    @staticmethod
    @handle_errors
    def export_npz(sim_data, params, history, name, Nx, Ny, dx, dy):
        """Save all frames as a NumPy .npz archive. Each frame becomes a
        sub-array keyed by f"{field}_{frame_idx:04d}"."""
        buf = BytesIO()
        out = {
            'params': dict(params),
            'x': np.linspace(-Nx * dx / 2, Nx * dx / 2, Nx),
            'y': np.linspace(-Ny * dy / 2, Ny * dy / 2, Ny),
        }
        keys = ('c_v', 'c_i', 'eta', 'phi_grain', 'phi_gb', 'porosity',
                'D_eff_v', 'D_eff_i', 'A_recomb', 'sigma_eq', 'sigma_h',
                'h', 'eps_p_mag', 'sigma_y')
        for i, frame in enumerate(history):
            for k in keys:
                if k in frame:
                    v = frame[k]
                    if isinstance(v, np.ndarray):
                        # Apply same unit scaling as visualization
                        if k in ('sigma_eq', 'sigma_h', 'sxx', 'syy', 'sxy'):
                            v = v / 1e9
                        elif k == 'sigma_y':
                            v = v / 1e6
                        out[f"{k}_{i:04d}"] = v
        np.savez_compressed(buf, **out)
        buf.seek(0)
        return buf, f"{name}.npz"

    @staticmethod
    @handle_errors
    def export_pt(sim_data, params, history, name):
        if not TORCH_AVAILABLE:
            st.error("torch not installed — pip install torch")
            return None, None
        buf = BytesIO()
        out = {'params': params, 'metadata': sim_data.get('metadata', {}),
               'history': []}
        for frame in history:
            frame_out = {}
            for k, v in frame.items():
                if isinstance(v, np.ndarray):
                    frame_out[k] = torch.from_numpy(np.asarray(v))
                elif isinstance(v, dict):
                    frame_out[k] = {
                        kk: (torch.from_numpy(np.asarray(vv))
                             if isinstance(vv, np.ndarray) else vv)
                        for kk, vv in v.items()
                    }
            out['history'].append(frame_out)
        torch.save(out, buf)
        buf.seek(0)
        return buf, f"{name}.pt"

    @staticmethod
    @handle_errors
    def export_json(sim_data, params, history, name):
        def conv(o):
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, np.generic): return o.item()
            if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
            if isinstance(o, list): return [conv(i) for i in o]
            return o
        s = json.dumps({'params': params,
                        'metadata': sim_data.get('metadata', {}),
                        'history': conv(history)},
                       indent=2, cls=NumpyEncoder)
        return BytesIO(s.encode()), f"{name}.json"

    @staticmethod
    @handle_errors
    def export_csv(history, name, extent, Nx, Ny, dx, dy):
        """Export per-frame CSVs zipped together. Each CSV has columns
        x, y, c_v, c_i, eta, phi_grain, phi_gb, sigma_eq, sigma_h, ..."""
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            x = np.linspace(extent[0], extent[1], Nx)
            y = np.linspace(extent[2], extent[3], Ny)
            X, Y = np.meshgrid(x, y)
            keys = ('c_v', 'c_i', 'eta', 'phi_grain', 'phi_gb', 'porosity',
                    'sigma_eq', 'sigma_h', 'h', 'eps_p_mag', 'sigma_y',
                    'D_eff_v', 'D_eff_i', 'A_recomb')
            for i, frame in enumerate(history):
                d = {'x': X.flatten(), 'y': Y.flatten()}
                for k in keys:
                    if k in frame:
                        v = frame[k]
                        if k in ('sigma_eq', 'sigma_h'):
                            v = v / 1e9
                        elif k == 'sigma_y':
                            v = v / 1e6
                        d[k] = np.asarray(v).flatten()
                zf.writestr(f"{name}_frame_{i:04d}.csv",
                            pd.DataFrame(d).to_csv(index=False))
        buf.seek(0)
        return buf, f"{name}_csv.zip"

    @staticmethod
    @handle_errors
    def export_timeseries_csv(solver_history, name):
        """Export the diagnostic time-series (porosity, mean c_v, GB area,
        max σ_eq, plastic work, n_voids) as a single CSV."""
        h = solver_history
        df = pd.DataFrame({
            'time':         h['time'],
            'porosity':     h['porosity'],
            'mean_c_v':     h['mean_c_v'],
            'mean_c_i':     h['mean_c_i'],
            'gb_area':      h['gb_area'],
            'max_stress_Pa': h['max_stress'],
            'avg_stress_Pa': h['avg_stress'],
            'max_stress_GPa': np.array(h['max_stress']) / 1e9,
            'avg_stress_GPa': np.array(h['avg_stress']) / 1e9,
            'plastic_work': h['plastic_work'],
            'n_voids':      h['n_voids'],
            'void_volume':  h['void_volume'],
        })
        csv_str = df.to_csv(index=False)
        return BytesIO(csv_str.encode()), f"{name}_timeseries.csv"

    @staticmethod
    @handle_errors
    def export_hdf5(sim_data, params, history, name, Nx, Ny, dx, dy):
        if not H5PY_AVAILABLE:
            st.error("h5py not installed — pip install h5py")
            return None, None
        buf = BytesIO()
        with h5py.File(buf, 'w') as f:
            pg = f.create_group('parameters')
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool)):
                    pg.attrs[k] = v
            f.create_dataset('x', data=np.linspace(-Nx * dx / 2, Nx * dx / 2, Nx))
            f.create_dataset('y', data=np.linspace(-Ny * dy / 2, Ny * dy / 2, Ny))
            fg = f.create_group('frames')
            for i, frame in enumerate(history):
                g = fg.create_group(f'frame_{i:04d}')
                for k, v in frame.items():
                    if isinstance(v, np.ndarray):
                        g.create_dataset(k, data=v)
        buf.seek(0)
        return buf, f"{name}.h5"


# ============================================================================
# SIMULATION DATABASE
# ============================================================================
class SimulationDatabase:
    MAX_SIMS = 32

    @staticmethod
    def _hash(params):
        s = json.dumps({k: v for k, v in params.items()
                        if k not in ('history', 'results', 'geom_viz',
                                     'n_steps', 'save_frequency')},
                       sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()[:8]

    @staticmethod
    @handle_errors
    def save(params, history, geometry_data, metadata=None, run_time=0.0):
        if 'sims' not in st.session_state:
            st.session_state.sims = {}
        sid = SimulationDatabase._hash(params)
        st.session_state.sims[sid] = {
            'id': sid, 'params': params, 'results_history': history,
            'geometry_data': geometry_data,
            'metadata': metadata or {'created_at': datetime.now().isoformat(),
                                      'run_time': run_time},
            'created_at': datetime.now().isoformat(),
        }
        if len(st.session_state.sims) > SimulationDatabase.MAX_SIMS:
            oldest = min(st.session_state.sims.items(),
                         key=lambda kv: kv[1].get('created_at', ''))
            del st.session_state.sims[oldest[0]]
        return sid

    @staticmethod
    def get(sid):
        return st.session_state.get('sims', {}).get(sid)

    @staticmethod
    def delete(sid):
        if sid in st.session_state.get('sims', {}):
            del st.session_state.sims[sid]
            return True
        return False

    @staticmethod
    def clear_all():
        st.session_state.sims = {}

    @staticmethod
    def all():
        return st.session_state.get('sims', {})

    @staticmethod
    def list_for_ui():
        out = []
        for sid, sim in SimulationDatabase.all().items():
            p = sim.get('params', {})
            scenario = p.get('scenario', '?')
            out.append({
                'id': sid,
                'name': (f"UO2 | {scenario} | "
                         f"Nx={p.get('Nx', '?')} | "
                         f"K_dot={p.get('K_dot', 0):.1e} | "
                         f"σ_app={p.get('applied_stress', 0) / 1e6:.0f}MPa"),
                'params': p,
            })
        return out


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================
def main():
    st.set_page_config(
        page_title="UO2 Void-GB Phase-Field (J2 Plasticity)",
        layout="wide", initial_sidebar_state="expanded")

    viz_cfg = get_viz_config()
    apply_matplotlib_theme(viz_cfg['theme'], viz_cfg)

    st.markdown("""
    <style>
    .main-header { font-size: 2.1rem; text-align: center; margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #1b4f72, #a0522d);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .viz-badge { background:#eef; padding:2px 6px; border-radius:4px;
                 font-size:0.8rem; color:#335; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">⚛️ UO2 Void–GB Phase-Field with J2 '
                'Plasticity</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color:#FFF8E7; padding:1rem; border-radius:8px;
                border-left:5px solid #a0522d;">
    <b>Physics:</b> Modified Cahn-Hilliard (c_v, c_i) + Allen-Cahn (η, φ_i)
    + Anisotropic mobility (bulk + surface + GB) + Cascade irradiation sources
    + Recombination + GB sinks + Spectral elastoplasticity (J2 flow rule,
    Hall-Petch yield) + Larché-Cahn σ_h coupling.<br>
        <b>Material:</b> UO2 at 1273 K (E=220 GPa, nu=0.31, sigma_y0=150 MPa).<br>
    <span class="viz-badge">3 scenarios: multi-void+GB · GB migration under irradiation · polycrystal</span>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuration")
        mode = st.radio("Mode",
                        ["Run New Simulation", "Single Simulation View",
                         "Compare Saved"],
                        index=0, key="mode")

        if mode == "Run New Simulation":
            st.subheader("🧪 Scenario")
            scenario = st.selectbox(
                "Simulation scenario",
                ['multi_void_gb', 'gb_irradiation', 'polycrystal'],
                index=0,
                format_func=lambda x: {
                    'multi_void_gb':  'Multi-void + flat GB',
                    'gb_irradiation': 'GB migration under irradiation',
                    'polycrystal':    'Polycrystalline UO2',
                }[x],
                help="Selects the initial geometry and which physics modules "
                     "are active (irradiation source terms, artificial GB "
                     "driving force, etc.).")

            st.subheader("🧱 Geometry")
            if scenario == 'multi_void_gb':
                st.caption("Three voids of different radii placed in front "
                           "of a flat GB.")
                r1 = st.slider("Void 1 radius (nm)", 1.0, 10.0, 3.0, 0.5)
                r2 = st.slider("Void 2 radius (nm)", 1.0, 10.0, 4.0, 0.5)
                r3 = st.slider("Void 3 radius (nm)", 1.0, 10.0, 5.0, 0.5)
                void_radii = (r1, r2, r3)
                void_positions = None  # auto-placed
                n_grains_flat = 2
                gb_width_nm = st.slider("GB width (nm)", 1.0, 6.0, 3.0, 0.5)
                n_poly_grains = 2
                poly_grain_size_nm = 30.0
                use_warp = False
                warp_amp = 0.0
            elif scenario == 'gb_irradiation':
                st.caption("Flat GB + irradiation source, no initial voids. "
                           "Watch VDZ formation as the GB sweeps through.")
                void_radii = ()
                void_positions = None
                n_grains_flat = 2
                gb_width_nm = st.slider("GB width (nm)", 1.0, 6.0, 3.0, 0.5)
                n_poly_grains = 2
                poly_grain_size_nm = 30.0
                use_warp = False
                warp_amp = 0.0
            else:  # polycrystal
                st.caption("Periodic polycrystalline UO2 with random Voronoi "
                           "grains. Memory scales linearly with n_grains.")
                void_radii = ()
                void_positions = None
                n_grains_flat = 2
                gb_width_nm = 2.5
                n_poly_grains = st.slider(
                    "Number of grains", 4, 64, 32, 4,
                    help="Default 64. Higher = more memory "
                         "(~0.5 MB per grain at 128x128).")
                poly_grain_size_nm = st.slider(
                    "Target grain size (nm)", 10.0, 80.0, 30.0, 5.0)
                use_warp = st.checkbox(
                    "Apply warp to Voronoi boundaries", value=True)
                warp_amp = st.slider("GB waviness (nm)", 0.0, 8.0, 2.0, 0.5)

            seed_in = st.number_input("Random seed", value=1234, step=1)

            st.subheader("🔲 Grid")
            st.markdown("**📐 Physical domain size**")
            c1, c2 = st.columns(2)
            with c1:
                Lx_nm = st.slider("Lx (domain width, nm)", 30.0, 400.0, 80.0, 5.0,
                                   format="%.1f", key="Lx_nm")
            with c2:
                Ly_nm = st.slider("Ly (domain height, nm)", 30.0, 400.0, 80.0, 5.0,
                                   format="%.1f", key="Ly_nm")
            dx = st.slider("dx (nm)", 0.2, 2.0, 0.5, 0.1)
            dy = st.slider("dy (nm)", 0.2, 2.0, 0.5, 0.1)

            Nx = int(round(Lx_nm / dx))
            Ny = int(round(Ly_nm / dy))
            Nx = max(32, min(512, Nx))
            Ny = max(32, min(512, Ny))
            Lx_actual = Nx * dx
            Ly_actual = Ny * dy
            st.caption(
                f"📐 Actual domain: **{Lx_actual:.1f} nm × {Ly_actual:.1f} nm** "
                f"(Nx={Nx}, Ny={Ny}, {Nx * Ny:,} cells)")

            ncells = Nx * Ny
            mem_per_frame_mb = ncells * 8 / 1e6
            mem_estimate_mb = mem_per_frame_mb * (20 + n_poly_grains * 0.5)
            if mem_estimate_mb > 1500:
                st.warning(f"⚠️ Estimated memory ~{mem_estimate_mb / 1000:.1f} GB "
                           f"— may OOM. Reduce n_grains or domain size.")
            elif mem_estimate_mb > 500:
                st.caption(f"💾 Estimated memory ~{mem_estimate_mb:.0f} MB — OK "
                           f"but watch history length.")

            st.subheader("⏱️ Time")
            time_unit_keys = list(TIME_UNITS.keys())
            time_unit = st.selectbox(
                "Time unit", time_unit_keys,
                index=time_unit_keys.index(
                    st.session_state.get('time_unit_sel', DEFAULT_TIME_UNIT)),
                key="time_unit_sel",
                help="Pick the physical time unit for display. 'τ' is the reduced "
                     "unit (t* = 0.01 s).")
            dt = st.slider(f"dt ({time_unit})", 1e-4, 5e-2, 1e-2, 1e-4,
                            format="%.5f")
            n_steps = st.slider("Steps", 20, 2000, 200, 20)
            st.session_state['n_steps_cached'] = int(n_steps)
            save_freq = st.slider("Save every N steps", 1, 100, 10, 1)
            st.caption(f"Total simulated time = {dt * n_steps:.4g} {time_unit}")

            st.subheader("🌡️ UO2 Thermodynamics")
            kappa = st.slider("κ (gradient energy coefficient)", 0.5, 10.0, 3.13, 0.1)
            L_ac = st.slider("L (Allen-Cahn mobility)", 0.1, 5.0, 1.0, 0.1)
            a_GB = st.slider("a_GB (diffuse GB parameter)", 0.1, 5.0, 1.0, 0.1)

            st.subheader("🕳️ Diffusion & Mobility")
            D_v_scalar = st.slider("D_v (vacancy, scalar)", 1e-3, 5.0, 0.3346, 1e-3,
                                    format="%.4f")
            D_i_scalar = st.slider("D_i (interstitial, scalar)", 1e3, 1e8, 1.623e7, 1e3,
                                    format="%.2e")
            w_b = st.slider("w_b (bulk weight)", 1e-6, 1e-2, 1e-4, 1e-5, format="%.1e")
            w_s = st.slider("w_s (surface weight)", 1e-6, 1e-2, 1e-4, 1e-5, format="%.1e")
            w_GB = st.slider("w_GB (GB weight)", 1e-3, 1.0, 0.1, 1e-3, format="%.3f")

            st.subheader("☢️ Irradiation")
            has_irradiation = (scenario in ('gb_irradiation', 'polycrystal'))
            if has_irradiation:
                K_dot_default = 5e-3 if scenario == 'gb_irradiation' else 5e-3
                K_dot = st.slider(
                    "K_dot (dpa per τ)", 1e-4, 5e-2, K_dot_default, 1e-4,
                    format="%.4f",
                    help="Displacement rate (default 5e-3 dpa/τ).")
                P_casc = st.slider(
                    "P_casc (cascade probability)", 1e-3, 0.2, 0.05, 1e-3,
                    format="%.3f",
                    help="Probability of cascade per cell per timestep.")
                production_bias = st.slider(
                    "Production bias (P_i / P_v)", 0.1, 1.0, 2.0 / 3.0, 0.05)
                A_bulk = st.slider("A_bulk (bulk recombination)", 0.0, 5.0, 1.0, 0.1)
                A_surf = st.slider("A_surf (surface recombination)", 0.0, 20.0, 5.0, 0.5)
                S_v_r = st.slider("S_v_r (vacancy GB sink)", 0.0, 5.0, 1.0, 0.1)
                S_i_r = st.slider("S_i_r (interstitial GB sink)", 0.0, 5.0, 1.0, 0.1)
            else:
                K_dot = 0.0
                P_casc = 0.05
                production_bias = 2.0 / 3.0
                A_bulk = 1.0
                A_surf = 5.0
                S_v_r = 1.0
                S_i_r = 1.0

            st.subheader("🔗 Mechanics (J2 plasticity)")
            applied_mpa = st.slider(
                "Applied stress (MPa)", 0.0, 500.0, 50.0, 10.0,
                help="Far-field compressive stress along y (downward).")
            sigma0_mpa = st.slider("σ_y0 (initial yield, MPa)", 50.0, 500.0, 150.0, 10.0)
            H_GPa = st.slider("H (hardening modulus, GPa)", 0.0, 20.0, 5.0, 0.5)
            m_overstress = st.slider("m (overstress exponent)", 5, 40, 20, 1)
            Omega_v_eff = st.slider(
                "Ω_v (Larché-Cahn, eV/GPa)", 0.0, 1.0, 0.1, 0.01,
                help="Hydrostatic-stress coupling strength for vacancies. "
                     "Vacancies are attracted to tensile regions.")
            Omega_i_eff = st.slider(
                "Ω_i (Larché-Cahn, eV/GPa)", 0.0, 2.0, 0.2, 0.01)
            beta_v = st.slider(
                "β_v (vacancy eigenstrain)", 0.0, 1.0, 0.5, 0.05,
                help="Lattice contraction per unit vacancy concentration.")

            st.subheader("↔️ GB Driving Force")
            if scenario in ('multi_void_gb', 'gb_irradiation'):
                f_d_default = (0.2 if scenario == 'multi_void_gb' else 0.4)
                f_d_x = st.slider(
                    "f_d_x (artificial GB driving force)", 0.0, 1.0,
                    f_d_default, 0.05,
                    help="Artificial driving force — pushes the GB in the +x direction. "
                         "0.2 for multi-void, 0.4 for irradiation scenario.")
                f_d_y = 0.0
            else:
                f_d_x = 0.0
                f_d_y = 0.0
                st.caption("Not applicable for polycrystal scenario.")

            # ===== Build params dict and initialize =====
            GEOM_TRIGGER_KEYS = [
                ('scenario', scenario),
                ('Lx_nm', Lx_nm), ('Ly_nm', Ly_nm),
                ('dx', dx), ('dy', dy),
                ('void_radii', tuple(void_radii)),
                ('gb_width_nm', gb_width_nm),
                ('n_poly_grains', int(n_poly_grains)),
                ('poly_grain_size_nm', poly_grain_size_nm),
                ('use_warp', bool(use_warp)),
                ('warp_amp', warp_amp),
                ('seed_in', int(seed_in)),
            ]

            def _do_initialize(silent=False):
                params = {
                    'scenario': scenario,
                    'Nx': Nx, 'Ny': Ny, 'dx': dx, 'dy': dy, 'dt': dt,
                    'time_unit': time_unit,
                    'void_radii': tuple(void_radii),
                    'void_positions': void_positions,
                    'n_grains_flat': n_grains_flat,
                    'gb_width_nm': gb_width_nm,
                    'n_poly_grains': int(n_poly_grains),
                    'poly_grain_size_nm': poly_grain_size_nm,
                    'use_warp': bool(use_warp),
                    'warp_amp': warp_amp,
                    'seed': int(seed_in),
                    'kappa': kappa, 'L_ac': L_ac, 'a_GB': a_GB,
                    'D_v_scalar': D_v_scalar, 'D_i_scalar': D_i_scalar,
                    'w_b': w_b, 'w_s': w_s, 'w_GB': w_GB,
                    'K_dot': K_dot, 'P_casc': P_casc,
                    'production_bias': production_bias,
                    'A_bulk': A_bulk, 'A_surf': A_surf,
                    'S_v_r': S_v_r, 'S_i_r': S_i_r,
                    'applied_stress': applied_mpa * 1e6,
                    'sigma0': sigma0_mpa * 1e6,
                    'H': H_GPa * 1e9,
                    'Omega_v_eff': Omega_v_eff,
                    'Omega_i_eff': Omega_i_eff,
                    'beta_v': beta_v,
                    'f_d': (f_d_x, f_d_y),
                    'n_steps': n_steps, 'save_frequency': save_freq,
                    'material': 'UO2',
                    'use_noise': True,
                }
                errors, warns = UO2Properties.validate_parameters(params)
                if errors:
                    st.error("; ".join(errors))
                    return False
                for w in warns:
                    st.warning(w)
                st.session_state.params = params
                st.session_state.initialized = True

                geom = UO2Geometry(Nx, Ny, dx, dy)
                res = geom.build(
                    scenario=scenario,
                    void_radii=tuple(void_radii),
                    void_positions=void_positions,
                    n_grains_flat=n_grains_flat,
                    gb_width_nm=gb_width_nm,
                    n_poly_grains=int(n_poly_grains),
                    poly_grain_size_nm=poly_grain_size_nm,
                    use_warp=bool(use_warp),
                    warp_amp=warp_amp,
                    c_eq_v=1e-6, c_eq_i=1e-9,
                    gb_energy=0.3, surface_energy=0.6,
                    kappa=kappa, seed=int(seed_in))
                if res is FAILED or res is None:
                    st.error(f"Geometry build failed: {FAILED.reason}")
                    return False
                (c_v0, c_i0, eta0, phi_list0, gm0, meta0) = res
                st.session_state.geom = geom
                st.session_state.c_v0 = c_v0
                st.session_state.c_i0 = c_i0
                st.session_state.eta0 = eta0
                st.session_state.phi_list0 = phi_list0
                st.session_state.gm0 = gm0
                st.session_state.meta0 = meta0
                for _k in ('history', 'timesteps', 'solver'):
                    if _k in st.session_state:
                        del st.session_state[_k]
                if not silent:
                    st.success(
                        f"✅ Initialized: scenario={scenario}, "
                        f"{meta0['n_grains']} grains, "
                        f"{len(void_radii)} voids, "
                        f"domain {Nx * dx:.1f} × {Ny * dy:.1f} nm")
                return True

            # Auto-rebuild on geometry change
            prev_geom_params = st.session_state.get('_prev_geom_params', {})
            geom_changed = False
            for k, v in GEOM_TRIGGER_KEYS:
                if prev_geom_params.get(k) != v:
                    geom_changed = True
                    prev_geom_params[k] = v
            st.session_state['_prev_geom_params'] = prev_geom_params

            if geom_changed and st.session_state.get('initialized', False):
                with st.spinner("🔄 Rebuilding geometry…"):
                    _do_initialize(silent=True)
                st.rerun()

            if st.button("🚀 Initialize", type="primary",
                         use_container_width=True):
                _do_initialize(silent=False)

        elif mode == "Single Simulation View":
            sims = SimulationDatabase.list_for_ui()
            if not sims:
                st.info("No saved simulations.")
            else:
                names = [s['name'] for s in sims]
                chosen = st.selectbox("Select", names)
                st.session_state.view_id = sims[names.index(chosen)]['id']

        elif mode == "Compare Saved":
            sims = SimulationDatabase.list_for_ui()
            if not sims:
                st.info("No saved simulations.")
            else:
                names = [s['name'] for s in sims]
                picked = st.multiselect("Select", names,
                                         default=names[:min(2, len(names))])
                st.session_state.compare_ids = [sims[names.index(n)]['id']
                                                 for n in picked]

        st.markdown("---")
        with st.expander("🗄️ Simulation database", expanded=False):
            n_sims = len(SimulationDatabase.all())
            st.caption(f"{n_sims} saved simulations (max {SimulationDatabase.MAX_SIMS})")
            if st.button("🧹 Clear all saved simulations"):
                SimulationDatabase.clear_all()
                st.success("Cleared.")
                st.rerun()

    # ==================================================================
    # MAIN AREA
    # ==================================================================
    if mode == "Run New Simulation" and st.session_state.get('initialized'):
        params = st.session_state.params
        Nx, Ny = params['Nx'], params['Ny']
        dx, dy = params['dx'], params['dy']
        meta = st.session_state.meta0

        tabs = st.tabs(["📐 Geometry", "▶️ Run", "🎨 Visualization Settings",
                        "📊 Results", "🔬 Line Profiles", "🌐 Plotly",
                        "📤 Export"])

        # ---- Geometry ----------------------------------------------
        with tabs[0]:
            st.header(f"Initial Geometry — {meta['scenario']}")
            st.caption(
                f"Scenario: {meta['scenario']} · "
                f"{meta['n_grains']} grains · "
                f"{len(meta.get('void_radii', []))} voids · "
                f"Domain: {Nx * dx:.1f} × {Ny * dy:.1f} nm")

            extent = [-Nx * dx / 2, Nx * dx / 2, -Ny * dy / 2, Ny * dy / 2]
            gm = st.session_state.gm0

            fig, ax = plt.subplots(figsize=(10, 4.5))
            cmap = ColormapRegistry.resolve('uo2_grain_map')
            im = ax.imshow(gm, extent=extent, origin='lower', cmap=cmap,
                           interpolation='nearest', aspect='equal')

            # Mark void positions if present
            if meta.get('void_positions'):
                for (x0, y0), r in zip(meta['void_positions'],
                                         meta['void_radii']):
                    circle = plt.Circle((x0, y0), r, color='red',
                                         fill=False, lw=2)
                    ax.add_patch(circle)
                ax.plot([], [], 'o', color='red', mfc='none', mec='red',
                         lw=2, label=f'Voids ({len(meta["void_radii"])})')
                ax.legend(loc='upper right')

            # Mark flat GB if applicable
            if meta.get('gb_x') is not None:
                ax.axvline(meta['gb_x'], color='cyan', lw=2, ls='--',
                            label=f'Flat GB at x={meta["gb_x"]:.1f} nm')
                ax.legend(loc='upper right')

            ax.set_title(f"Initial grain map — {meta['scenario']}")
            ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
            st.pyplot(fig); plt.close(fig)

            # Show initial void order parameter
            eta0 = st.session_state.eta0
            if eta0.max() > 0.01:
                fig, ax = plt.subplots(figsize=(10, 4.5))
                cmap_eta = ColormapRegistry.resolve('void_map')
                im = ax.imshow(eta0, extent=extent, origin='lower',
                                cmap=cmap_eta, vmin=0, vmax=1, aspect='equal')
                plt.colorbar(im, ax=ax, label='η (void OP)')
                ax.set_title('Initial void order parameter η')
                ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
                st.pyplot(fig); plt.close(fig)

            # Show initial c_v
            c_v0 = st.session_state.c_v0
            fig, ax = plt.subplots(figsize=(10, 4.5))
            fs_cv = viz_cfg['field_settings'].get('c_v', {})
            cmap_cv = ColormapRegistry.resolve(
                FIELD_DEFAULT_CMAPS['c_v'],
                invert=fs_cv.get('invert', False))
            im = ax.imshow(c_v0, extent=extent, origin='lower', cmap=cmap_cv,
                            vmin=0, vmax=1, aspect='equal')
            plt.colorbar(im, ax=ax, label='c_v')
            ax.set_title('Initial vacancy concentration c_v')
            ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
            st.pyplot(fig); plt.close(fig)

            with st.expander("📋 Scenario parameters", expanded=False):
                st.json({
                    'scenario': meta['scenario'],
                    'n_grains': meta['n_grains'],
                    'gb_width_nm': meta.get('gb_width_nm'),
                    'void_radii': meta.get('void_radii', []),
                    'void_positions': meta.get('void_positions', []),
                    'has_irradiation': meta.get('has_irradiation'),
                    'has_driving_force': meta.get('has_driving_force'),
                    'f_d_default': meta.get('f_d_default'),
                    'K_dot_default': meta.get('K_dot_default'),
                })

        # ---- Run ---------------------------------------------------
        with tabs[1]:
            st.header("Run the UO2 Phase-Field Simulation")
            col1, col2 = st.columns([2, 1])
            with col1:
                t_unit = get_time_unit_label(params)
                irr_status = (f"K_dot = {params['K_dot']:.2e} dpa/{t_unit}"
                              if params.get('K_dot', 0) > 0 else "OFF")
                st.markdown(
                    f"**Expected physics with current settings:**\n"
                    f"- Scenario: `{params['scenario']}`\n"
                    f"- Domain: `{Nx * dx:.1f} × {Ny * dy:.1f} nm` "
                    f"({Nx} × {Ny} cells)\n"
                    f"- Time step: `{params['dt']:.4g} {t_unit}` × "
                    f"`{params['n_steps']}` steps = "
                    f"`{params['n_steps'] * params['dt']:.3g} {t_unit}` total\n"
                    f"- Irradiation: `{irr_status}`\n"
                    f"- Applied stress: `{params['applied_stress'] / 1e6:.0f} MPa`\n"
                    f"- J2 yield σ_y0: `{params['sigma0'] / 1e6:.0f} MPa`, "
                    f"hardening H: `{params['H'] / 1e9:.1f} GPa`\n"
                    f"- Larché-Cahn Ω_v: `{params['Omega_v_eff']:.2f} eV/GPa`\n"
                    f"- GB driving force f_d: `({params['f_d'][0]:.2f}, "
                    f"{params['f_d'][1]:.2f})`")
            if st.button("▶️ Start Evolution", type="primary",
                         use_container_width=True):
                with st.spinner("Simulating UO2 void-GB phase-field + J2..."):
                    solver = UO2VoidGBSolver(params)
                    # Override initial state with cached geometry
                    solver.c_v = st.session_state.c_v0.copy()
                    solver.c_i = st.session_state.c_i0.copy()
                    solver.eta = st.session_state.eta0.copy()
                    solver.phi_list = [p.copy() for p in st.session_state.phi_list0]

                    n_steps = params['n_steps']
                    save_freq = params['save_frequency']
                    history, timesteps = [], []
                    progress = st.progress(0)
                    status = st.empty()

                    for step in range(n_steps):
                        res = solver.step()
                        if res is FAILED or res is None:
                            st.error(f"Solver step failed: {FAILED.reason}")
                            break
                        if step % save_freq == 0 or step == n_steps - 1:
                            history.append({
                                k: (v.copy() if isinstance(v, np.ndarray)
                                    else (dict(v) if isinstance(v, dict) else v))
                                for k, v in res.items()
                            })
                            timesteps.append(step * params['dt'])
                        progress.progress((step + 1) / n_steps)
                        if step % max(1, n_steps // 20) == 0:
                            por = solver.history['porosity'][-1]
                            cv = solver.history['mean_c_v'][-1]
                            smax = solver.history['max_stress'][-1] / 1e9
                            nv = solver.history['n_voids'][-1]
                            status.text(
                                f"Step {step + 1}/{n_steps} · "
                                f"porosity={por:.4f} · "
                                f"mean_c_v={cv:.2e} · "
                                f"max_σ_eq={smax:.3f} GPa · "
                                f"n_voids={nv}")

                    st.success(f"✅ Done. {len(history)} frames saved.")
                    st.session_state.history = history
                    st.session_state.timesteps = timesteps
                    st.session_state.solver = solver
                    sid = SimulationDatabase.save(
                        params, history, None,
                        run_time=n_steps * params['dt'])
                    st.session_state.last_sid = sid
                    st.balloons()

        # ---- Visualization Settings --------------------------------
        with tabs[2]:
            st.header("🎨 Visualization Settings")
            st.caption("These settings apply to the Results, Line Profiles, "
                       "Plotly, and Export tabs.")

            sub1, sub2, sub3 = st.tabs(
                ["Global & Field Scaling",
                 "Overlay & Vector & Animation",
                 "Figure Customization"])

            with sub1:
                st.subheader("🖌️ Global matplotlib theme")
                theme_keys = list(MATPLOTLIB_THEMES.keys())
                default_theme_idx = (theme_keys.index(viz_cfg['theme'])
                                     if viz_cfg['theme'] in theme_keys else 0)
                c1, c2, c3 = st.columns(3)
                with c1:
                    theme = st.selectbox("Theme", theme_keys,
                                          index=default_theme_idx, key="viz_theme")
                with c2:
                    title_fs = st.slider("Title font size", 6, 24,
                                          int(viz_cfg['title_font_size']),
                                          key="viz_title_fs")
                    label_fs = st.slider("Axis label font size", 6, 20,
                                          int(viz_cfg['label_font_size']),
                                          key="viz_label_fs")
                with c3:
                    tick_fs = st.slider("Tick font size", 4, 18,
                                         int(viz_cfg['tick_font_size']),
                                         key="viz_tick_fs")
                    legend_fs = st.slider("Legend font size", 4, 18,
                                           int(viz_cfg['legend_font_size']),
                                           key="viz_legend_fs")
                dpi = st.slider("DPI", 72, 600, int(viz_cfg['dpi']), 12,
                                 key="viz_dpi")

                if st.button("♻️ Apply theme now", key="viz_apply_theme"):
                    viz_cfg['theme'] = theme
                    viz_cfg['title_font_size'] = title_fs
                    viz_cfg['label_font_size'] = label_fs
                    viz_cfg['tick_font_size'] = tick_fs
                    viz_cfg['legend_font_size'] = legend_fs
                    viz_cfg['dpi'] = dpi
                    apply_matplotlib_theme(theme, viz_cfg)
                    st.success("Theme applied.")
                    st.rerun()

                st.markdown("---")
                st.subheader("📏 Per-field dynamic range & scaling")
                field_keys = list(FIELD_DEFAULT_CMAPS.keys())
                field_choice = st.selectbox("Field to configure", field_keys,
                                              key="viz_field_choice")
                fs = viz_cfg['field_settings'][field_choice]

                auto = st.checkbox("Auto range (2–98 percentile)",
                                    value=fs.get('auto', True),
                                    key=f"viz_auto_{field_choice}")
                if not auto:
                    c1, c2 = st.columns(2)
                    drng = FIELD_DEFAULT_RNG.get(field_choice, (0.0, 1.0))
                    current_vmin = (fs.get('vmin') if fs.get('vmin') is not None
                                    else drng[0])
                    current_vmax = (fs.get('vmax') if fs.get('vmax') is not None
                                    else drng[1])
                    vmin_in = c1.number_input("Min value (vmin)",
                                               value=float(current_vmin),
                                               key=f"viz_vmin_{field_choice}",
                                               format="%.6g")
                    vmax_in = c2.number_input("Max value (vmax)",
                                               value=float(current_vmax),
                                               key=f"viz_vmax_{field_choice}",
                                               format="%.6g")
                else:
                    vmin_in, vmax_in = None, None

                col_log, col_inv = st.columns(2)
                with col_log:
                    log_scale = st.checkbox("Logarithmic scale",
                                             value=fs.get('log', False),
                                             key=f"viz_log_{field_choice}",
                                             help="Best for c_v, c_i, D_eff_v, D_eff_i.")
                with col_inv:
                    invert = st.checkbox("Invert colormap",
                                          value=fs.get('invert', False),
                                          key=f"viz_inv_{field_choice}")

                cmap_options = ColormapRegistry.list_all()
                current_cmap = fs.get('cmap',
                                       FIELD_DEFAULT_CMAPS.get(field_choice, 'viridis'))
                cmap_idx = (cmap_options.index(current_cmap)
                            if current_cmap in cmap_options else 0)
                cmap_choice = st.selectbox(
                    f"Colormap for '{field_choice}'",
                    cmap_options, index=cmap_idx,
                    key=f"viz_cmap_{field_choice}")

                if st.button("💾 Save field settings",
                              key=f"viz_save_{field_choice}"):
                    viz_cfg['field_settings'][field_choice] = {
                        'auto': auto, 'vmin': vmin_in, 'vmax': vmax_in,
                        'log': log_scale, 'invert': invert,
                        'cmap': cmap_choice,
                    }
                    st.success(f"Saved settings for '{field_choice}'.")
                    st.rerun()

            with sub2:
                st.subheader("🪟 Multi-field overlay builder")
                ov = viz_cfg['overlay']
                ov_enabled = st.checkbox("Enable overlay mode in Results",
                                          value=ov['enabled'], key="ov_en")
                field_keys2 = list(FIELD_DEFAULT_CMAPS.keys())
                c1, c2 = st.columns(2)
                with c1:
                    base_idx = (field_keys2.index(ov['base'])
                                if ov['base'] in field_keys2 else 2)
                    ov_base = st.selectbox("Base field", field_keys2,
                                            index=base_idx, key="ov_base")
                with c2:
                    over_idx = (field_keys2.index(ov['overlay'])
                                if ov['overlay'] in field_keys2 else 4)
                    ov_over = st.selectbox("Overlay field", field_keys2,
                                            index=over_idx, key="ov_over")
                c1, c2 = st.columns(2)
                with c1:
                    ov_mode = st.radio("Overlay mode", ['contour', 'heatmap'],
                                        index=0 if ov['mode'] == 'contour' else 1,
                                        key="ov_mode")
                with c2:
                    ov_alpha = st.slider("Alpha", 0.05, 1.0,
                                          float(ov['alpha']), 0.05, key="ov_alpha")
                ov_nc = st.slider("Contour levels", 3, 30, int(ov['n_contours']),
                                   1, key="ov_nc")
                if st.button("💾 Save overlay settings", key="ov_save"):
                    viz_cfg['overlay'] = {
                        'enabled': ov_enabled, 'base': ov_base,
                        'overlay': ov_over, 'alpha': ov_alpha,
                        'mode': ov_mode, 'n_contours': ov_nc,
                    }
                    st.success("Overlay settings saved.")
                    st.rerun()

                st.markdown("---")
                st.subheader("🧭 Vector field overlay")
                vv = viz_cfg['vector']
                vv_enabled = st.checkbox("Enable vector-field mode",
                                          value=vv['enabled'], key="vv_en")
                c1, c2 = st.columns(2)
                with c1:
                    vv_src_idx = (VECTOR_FIELD_SOURCES.index(vv['source'])
                                  if vv['source'] in VECTOR_FIELD_SOURCES else 0)
                    vv_source = st.selectbox("Source field", VECTOR_FIELD_SOURCES,
                                              index=vv_src_idx, key="vv_src")
                with c2:
                    vv_mode_list = ['gradient', 'flux', 'principal_stress']
                    vv_m_idx = (vv_mode_list.index(vv['mode'])
                                if vv['mode'] in vv_mode_list else 0)
                    vv_mode = st.selectbox("Mode", vv_mode_list,
                                            index=vv_m_idx, key="vv_mode")
                c1, c2 = st.columns(2)
                with c1:
                    vv_density = st.slider("Arrow density", 6, 80,
                                            int(vv['density']), 2, key="vv_den")
                with c2:
                    vv_scale = st.slider("Arrow scale", 0.1, 5.0,
                                          float(vv['scale']), 0.1, key="vv_scale")
                if st.button("💾 Save vector settings", key="vv_save"):
                    viz_cfg['vector'] = {
                        'enabled': vv_enabled, 'source': vv_source,
                        'mode': vv_mode, 'density': vv_density,
                        'color': '#ffffff', 'scale': vv_scale,
                    }
                    st.success("Vector settings saved.")
                    st.rerun()

                st.markdown("---")
                st.subheader("🎬 Animation settings")
                an = viz_cfg['animation']
                c1, c2, c3 = st.columns(3)
                with c1:
                    an_fmt = st.selectbox("Format", ['gif', 'mp4'],
                                           index=0 if an['format'] == 'gif' else 1,
                                           key="an_fmt")
                with c2:
                    an_fps = st.slider("FPS", 1, 30, int(an['fps']), 1, key="an_fps")
                with c3:
                    an_dpi = st.slider("DPI", 72, 400, int(an['dpi']), 12, key="an_dpi")
                an_wm = st.checkbox("Show watermark", value=an['watermark'], key="an_wm")
                an_wm_text = st.text_input("Watermark text",
                                            value=an['watermark_text'], key="an_wm_text")
                wm_pos_list = ['lower right', 'lower left', 'upper right', 'upper left']
                wm_idx = (wm_pos_list.index(an['watermark_pos'])
                          if an['watermark_pos'] in wm_pos_list else 0)
                an_wm_pos = st.selectbox("Watermark position", wm_pos_list,
                                          index=wm_idx, key="an_wm_pos")
                if st.button("💾 Save animation settings", key="an_save"):
                    viz_cfg['animation'] = {
                        'format': an_fmt, 'fps': an_fps, 'dpi': an_dpi,
                        'watermark': an_wm, 'watermark_text': an_wm_text,
                        'watermark_pos': an_wm_pos,
                    }
                    st.success("Animation settings saved.")
                    st.rerun()

            with sub3:
                fc = viz_cfg['figure_customization']
                st.info("💡 Per-figure customization is available inline above "
                        "each figure in the Results, Line Profiles, Plotly, "
                        "and Export tabs. The settings below are global defaults.")
                st.caption("Adjust the multi-field grid, overlay, vector, "
                           "diagnostics, and line-profile figure sizes here.")

                st.markdown("#### 📐 Figure sizes (inches)")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.caption("**Multi-field grid**")
                    mf_w = st.number_input("Width", value=float(fc.get('multi_field_width', 13.5)),
                                            min_value=4.0, max_value=30.0, step=0.5,
                                            key="fc_mf_w")
                    mf_h = st.number_input("Height", value=float(fc.get('multi_field_height', 11.0)),
                                            min_value=3.0, max_value=30.0, step=0.5,
                                            key="fc_mf_h")
                with c2:
                    st.caption("**Overlay / Vector**")
                    ov_w = st.number_input("Width", value=float(fc.get('overlay_width', 11.0)),
                                            min_value=4.0, max_value=30.0, step=0.5,
                                            key="fc_ov_w")
                    ov_h = st.number_input("Height", value=float(fc.get('overlay_height', 5.5)),
                                            min_value=3.0, max_value=30.0, step=0.5,
                                            key="fc_ov_h")
                with c3:
                    st.caption("**Diagnostics**")
                    dg_w = st.number_input("Width", value=float(fc.get('diagnostics_width', 13.0)),
                                            min_value=4.0, max_value=30.0, step=0.5,
                                            key="fc_dg_w")
                    dg_h = st.number_input("Height", value=float(fc.get('diagnostics_height', 8.0)),
                                            min_value=3.0, max_value=30.0, step=0.5,
                                            key="fc_dg_h")

                if st.button("💾 Save figure customization", key="fc_save",
                              type="primary"):
                    viz_cfg['figure_customization']['multi_field_width'] = float(mf_w)
                    viz_cfg['figure_customization']['multi_field_height'] = float(mf_h)
                    viz_cfg['figure_customization']['overlay_width'] = float(ov_w)
                    viz_cfg['figure_customization']['overlay_height'] = float(ov_h)
                    viz_cfg['figure_customization']['diagnostics_width'] = float(dg_w)
                    viz_cfg['figure_customization']['diagnostics_height'] = float(dg_h)
                    st.success("Figure customization saved.")
                    st.rerun()

        # ---- Results / Line Profiles / Plotly / Export tabs --------
        # (defined in the next chunk to keep this function readable)
        _render_results_tabs(tabs[3], tabs[4], tabs[5], tabs[6],
                              params, Nx, Ny, dx, dy, viz_cfg)

    elif mode == "Single Simulation View" and st.session_state.get('view_id'):
        _render_single_view(st.session_state.view_id, viz_cfg)

    elif mode == "Compare Saved" and st.session_state.get('compare_ids'):
        _render_compare_view(st.session_state.compare_ids, viz_cfg)


# ============================================================================
# RESULTS / LINE PROFILES / PLOTLY / EXPORT TAB RENDERERS
# ============================================================================
def _render_results_tabs(results_tab, lp_tab, plotly_tab, export_tab,
                          params, Nx, Ny, dx, dy, viz_cfg):
    """Render the Results, Line Profiles, Plotly, and Export tabs.
    Called from main() when the user has an initialized simulation."""

    # ---- Results tab ----------------------------------------------
    with results_tab:
        if 'history' not in st.session_state:
            st.info("Run the simulation first (▶️ Run tab).")
        else:
            history = st.session_state.history
            solver = st.session_state.solver
            idx = st.slider("Frame", 0, len(history) - 1,
                            len(history) - 1, key="res_frame")
            vis = EnhancedUO2Visualizer(Nx, Ny, dx, dy, dt=params['dt'])
            style = {
                'colormaps': {f: fset.get('cmap', FIELD_DEFAULT_CMAPS.get(f, 'viridis'))
                               for f, fset in viz_cfg['field_settings'].items()},
                'field_settings': viz_cfg['field_settings'],
                'title_font_size': viz_cfg['title_font_size'],
                'label_font_size': viz_cfg['label_font_size'],
                'colorbar_fraction': viz_cfg['colorbar_fraction'],
                'colorbar_pad': viz_cfg['colorbar_pad'],
                'figure_customization': viz_cfg.get('figure_customization', {}),
            }

            mode_res = st.radio(
                "Display mode",
                ["Multi-field grid", "Overlay", "Vector field"],
                horizontal=True, key="res_mode")

            if mode_res == "Multi-field grid":
                render_live_colormap_picker(
                    viz_cfg, key_prefix='res_cmap',
                    label='🎨 Per-field colormap (live, no Save)')
                render_enhanced_figure_customization(
                    viz_cfg,
                    size_keys=('multi_field_width', 'multi_field_height'),
                    key_prefix='res_mf',
                    label='🎨 Multi-field grid — full customization',
                    show_line_styles=False,
                    show_series_colors=False,
                    show_legend=False,
                    show_spines=True,
                    show_font_sizes=True,
                    show_colorbar=True,
                    show_axis_limits=True,
                    show_text_overrides=True,
                    show_grid=True)
            elif mode_res == "Overlay":
                render_enhanced_figure_customization(
                    viz_cfg,
                    size_keys=('overlay_width', 'overlay_height'),
                    key_prefix='res_ov',
                    label='🎨 Overlay — full customization',
                    show_line_styles=True,
                    show_series_colors=False,
                    show_legend=True,
                    show_spines=True,
                    show_font_sizes=True,
                    show_colorbar=True)
            else:
                render_enhanced_figure_customization(
                    viz_cfg,
                    size_keys=('vector_width', 'vector_height'),
                    key_prefix='res_vc',
                    label='🎨 Vector field — full customization',
                    show_line_styles=True,
                    show_series_colors=False,
                    show_legend=True,
                    show_spines=True,
                    show_font_sizes=True,
                    show_colorbar=True)
            style['figure_customization'] = viz_cfg.get('figure_customization', {})
            style['colormaps'] = {f: fset.get('cmap', FIELD_DEFAULT_CMAPS.get(f, 'viridis'))
                                   for f, fset in viz_cfg['field_settings'].items()}
            style['field_settings'] = viz_cfg['field_settings']

            if mode_res == "Multi-field grid":
                fig = vis.create_multi_field_comparison(history[idx], style)
                if fig is not None:
                    st.pyplot(fig); plt.close(fig)
            elif mode_res == "Overlay":
                ov = viz_cfg['overlay']
                fig = vis.create_overlay(
                    history[idx],
                    base_field=ov['base'],
                    overlay_field=ov['overlay'],
                    alpha=ov['alpha'],
                    mode=ov['mode'],
                    n_contours=ov['n_contours'],
                    show_grain_ghost=False,
                    grain_map=history[idx].get('phi_grain'),
                    style_params=style)
                if fig is not None:
                    st.pyplot(fig); plt.close(fig)
                else:
                    st.warning("Overlay failed — check field selections.")
            else:
                vv = viz_cfg['vector']
                base_choice = ('sigma_h' if vv['mode'] == 'principal_stress'
                               else 'sigma_eq')
                fig = vis.create_vector_overlay(
                    history[idx],
                    source=vv['source'],
                    mode=vv['mode'],
                    density=vv['density'],
                    color=vv['color'],
                    scale=vv['scale'],
                    base_field=base_choice,
                    style_params=style)
                if fig is not None:
                    st.pyplot(fig); plt.close(fig)
                else:
                    st.warning("Vector field unavailable.")

            # ---- Diagnostics time-series ----
            st.subheader("Region-Averaged Diagnostics")
            fc_diag = viz_cfg.setdefault('figure_customization',
                                          default_figure_customization())
            st.markdown("**📐 Diagnostics figure size**")
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                new_w = _clamped_number_input(
                    "Width (in)", value=fc_diag.get('diagnostics_width', 13.0),
                    min_value=4.0, max_value=30.0, step=0.5,
                    key="diag_quick_w")
            with c2:
                new_h = _clamped_number_input(
                    "Height (in)", value=fc_diag.get('diagnostics_height', 8.0),
                    min_value=3.0, max_value=30.0, step=0.5,
                    key="diag_quick_h")
            with c3:
                _cur_w = float(fc_diag.get('diagnostics_width', 13.0))
                _cur_h = float(fc_diag.get('diagnostics_height', 8.0))
                st.caption(f"Current: {_cur_w:.1f} × {_cur_h:.1f} in → "
                           f"each subplot ≈ {_cur_w / 2:.1f} × {_cur_h / 2:.1f} in.")
            if (new_w != fc_diag.get('diagnostics_width') or
                    new_h != fc_diag.get('diagnostics_height')):
                fc_diag['diagnostics_width'] = float(new_w)
                fc_diag['diagnostics_height'] = float(new_h)
                st.rerun()

            t_disp, t_unit = fmt_time_range(solver.history['time'], params)
            t = t_disp
            fig_cfg = viz_cfg.get('figure_customization', {})
            lw_main = float(fig_cfg.get('line_width', 2.0))
            lw_minor = float(fig_cfg.get('line_width_minor', 1.5))
            ls_main = fig_cfg.get('line_style', '-')
            ls_minor = fig_cfg.get('line_style_minor', '--')
            ms_main = fig_cfg.get('marker_style', 'none')
            ms_minor = fig_cfg.get('marker_style_minor', 'none')
            marker_size = float(fig_cfg.get('marker_size', 6))
            marker_ew = float(fig_cfg.get('marker_width', 1.0))
            col1 = fig_cfg.get('color_series_1', '#1f77b4')   # porosity
            col2 = fig_cfg.get('color_series_2', '#d62728')   # mean c_v
            col3 = fig_cfg.get('color_series_3', '#2ca02c')   # GB area
            col4 = fig_cfg.get('color_series_4', '#9467bd')   # Avg σ
            col5 = fig_cfg.get('color_series_5', '#8c564b')   # n_voids
            m_main = '' if ms_main in ('none', '') else ms_main
            m_minor = '' if ms_minor in ('none', '') else ms_minor
            l_main = '' if ls_main in ('none', '') else ls_main
            l_minor = '' if ls_minor in ('none', '') else ls_minor
            fig_w = float(fig_cfg.get('diagnostics_width', 13.0))
            fig_h = float(fig_cfg.get('diagnostics_height', 8.0))
            t_xlabel = fig_cfg.get('xlabel_override', '').strip() or f't ({t_unit})'

            fig = plt.figure(figsize=(fig_w, fig_h))
            gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1],
                          hspace=0.35, wspace=0.30)
            axes = np.array([
                [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
            ])

            # Top-left: porosity + mean c_v
            ax = axes[0, 0]
            ax.plot(t, solver.history['porosity'], color=col1,
                     linestyle=l_main, lw=lw_main, marker=m_main,
                     ms=marker_size, mew=marker_ew, label='Porosity (η²)')
            ax2 = ax.twinx()
            ax2.plot(t, solver.history['mean_c_v'], color=col2,
                      linestyle=l_minor, lw=lw_minor, marker=m_minor,
                      ms=marker_size, mew=marker_ew, label='mean c_v')
            ax2.set_yscale('log')
            ax2.set_ylabel('mean c_v (log)')
            apply_axis_styling(ax, fig_cfg,
                                title='Porosity & mean vacancy concentration',
                                xlabel=t_xlabel, ylabel='porosity',
                                force_aspect='auto')
            apply_legend_styling(ax, fig_cfg)

            # Top-right: GB area + n_voids
            ax = axes[0, 1]
            ax.plot(t, solver.history['gb_area'], color=col3,
                     linestyle=l_main, lw=lw_main, marker=m_main,
                     ms=marker_size, mew=marker_ew, label='GB area (nm²)')
            ax2 = ax.twinx()
            ax2.plot(t, solver.history['n_voids'], color=col5,
                      linestyle=l_minor, lw=lw_minor, marker=m_minor,
                      ms=marker_size, mew=marker_ew, label='n_voids')
            ax2.set_ylabel('n_voids')
            apply_axis_styling(ax, fig_cfg,
                                title='GB area & void count',
                                xlabel=t_xlabel, ylabel='GB area (nm²)',
                                force_aspect='auto')
            apply_legend_styling(ax, fig_cfg)

            # Bottom-left: stress evolution
            ax = axes[1, 0]
            ax.plot(t, np.array(solver.history['avg_stress']) / 1e9,
                     color=col4, linestyle=l_main, lw=lw_main,
                     marker=m_main, ms=marker_size, mew=marker_ew,
                     label='Avg σ_eq (GPa)')
            ax.plot(t, np.array(solver.history['max_stress']) / 1e9,
                     color=col2, linestyle=l_minor, lw=lw_minor,
                     marker=m_minor, ms=marker_size, mew=marker_ew,
                     label='Max σ_eq (GPa)')
            apply_axis_styling(ax, fig_cfg,
                                title='Von Mises stress evolution',
                                xlabel=t_xlabel, ylabel='σ_eq (GPa)',
                                force_aspect='auto')
            apply_legend_styling(ax, fig_cfg)

            # Bottom-right: plastic work + void volume
            ax = axes[1, 1]
            ax.plot(t, solver.history['plastic_work'], color=col1,
                     linestyle=l_main, lw=lw_main, marker=m_main,
                     ms=marker_size, mew=marker_ew, label='Plastic work')
            ax2 = ax.twinx()
            ax2.plot(t, solver.history['void_volume'], color=col5,
                      linestyle=l_minor, lw=lw_minor, marker=m_minor,
                      ms=marker_size, mew=marker_ew, label='Void volume (nm²)')
            ax2.set_ylabel('Void volume (nm²)')
            apply_axis_styling(ax, fig_cfg,
                                title='Plastic work & void volume',
                                xlabel=t_xlabel, ylabel='plastic work (a.u.)',
                                force_aspect='auto')
            apply_legend_styling(ax, fig_cfg)

            st.pyplot(fig); plt.close(fig)

    # ---- Line Profiles tab ----------------------------------------
    with lp_tab:
        if 'history' not in st.session_state:
            st.info("Run the simulation first.")
        else:
            history = st.session_state.history
            profiler = EnhancedLineProfiler(Nx, Ny, dx, dy)

            render_enhanced_figure_customization(
                viz_cfg,
                size_keys=('line_profile_width', 'line_profile_height'),
                key_prefix='lp',
                label='🎨 Line profile — full customization',
                show_line_styles=True,
                show_series_colors=False,
                show_legend=True,
                show_spines=True,
                show_font_sizes=True,
                show_colorbar=False,
                show_axis_limits=True,
                show_text_overrides=True,
                show_grid=True)

            sub_a, sub_b = st.tabs(["Single frame", "Multi-time compare"])

            with sub_a:
                idx = st.slider("Frame", 0, len(history) - 1,
                                len(history) - 1, key="lp_frame")
                res = history[idx]
                field = st.selectbox("Field", list(FIELD_DEFAULT_CMAPS.keys()),
                                      index=2, key="lp_field")
                ptype = st.selectbox("Profile direction",
                                      ["horizontal", "vertical",
                                       "diagonal", "anti_diagonal"],
                                      key="lp_ptype")
                pos = st.slider("Position ratio", 0.0, 1.0, 0.5, 0.05,
                                key="lp_pos")

                d, unit = _rescale_field(field, res[field])
                out = profiler.extract_profile(d, ptype, pos)
                if out is FAILED or out is None:
                    st.warning("Profile extraction failed.")
                else:
                    dist, prof, _ = out
                    if dist is not None:
                        fc = viz_cfg.get('figure_customization', {})
                        fig_w = float(fc.get('line_profile_width', 11.0))
                        fig_h = float(fc.get('line_profile_height', 4.0))
                        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                        ls_main = fc.get('line_style', '-')
                        ms_main = fc.get('marker_style', 'none')
                        m_main = '' if ms_main in ('none', '') else ms_main
                        l_main = '' if ls_main in ('none', '') else ls_main
                        s1_color = fc.get('color_series_1', '#1f77b4')
                        ax.plot(dist, prof, color=s1_color,
                                 linestyle=l_main,
                                 lw=float(fc.get('line_width', 2.0)),
                                 marker=m_main,
                                 ms=float(fc.get('marker_size', 6)),
                                 mew=float(fc.get('marker_width', 1.0)))
                        apply_axis_styling(ax, fc,
                                            title=f'{field} · {ptype} profile · frame {idx}',
                                            xlabel='Position (nm)',
                                            ylabel=f'{field} {unit}',
                                            force_aspect='auto')
                        st.pyplot(fig); plt.close(fig)

            with sub_b:
                field_mt = st.selectbox("Field", list(FIELD_DEFAULT_CMAPS.keys()),
                                         index=2, key="lp_mt_field")
                ptype_mt = st.selectbox("Profile direction",
                                         ["horizontal", "vertical",
                                          "diagonal", "anti_diagonal"],
                                         key="lp_mt_ptype")
                pos_mt = st.slider("Position ratio", 0.0, 1.0, 0.5, 0.05,
                                   key="lp_mt_pos")
                default_frames = list(range(0, len(history),
                                             max(1, len(history) // 6)))
                frames_sel = st.multiselect(
                    "Frames", list(range(len(history))),
                    default=default_frames)
                use_plotly_mt = st.checkbox("Use Plotly (interactive)",
                                             value=True, key="lp_mt_plotly")
                if frames_sel:
                    vis_lp = EnhancedUO2Visualizer(Nx, Ny, dx, dy,
                                                    dt=params['dt'])
                    fig_mt = vis_lp.create_multi_time_profiles(
                        history, field_mt, ptype_mt, frames_sel,
                        position_ratio=pos_mt, use_plotly=use_plotly_mt,
                        viz_cfg=viz_cfg)
                    if fig_mt is None:
                        st.warning("No profiles could be extracted.")
                    elif use_plotly_mt:
                        st.plotly_chart(fig_mt, use_container_width=True)
                    else:
                        st.pyplot(fig_mt); plt.close(fig_mt)

    # ---- Plotly tab -----------------------------------------------
    with plotly_tab:
        if 'history' not in st.session_state:
            st.info("Run the simulation first.")
        else:
            history = st.session_state.history
            vis = EnhancedUO2Visualizer(Nx, Ny, dx, dy, dt=params['dt'])

            view = st.radio("Plotly view",
                             ["Heatmap (frame)", "3D surface",
                              "Time-scrubbing animation",
                              "Line profiles"],
                             horizontal=True, key="px_view")

            if view in ("Heatmap (frame)", "Time-scrubbing animation"):
                _sz_keys = ('overlay_width', 'overlay_height')
                _lbl = 'Plotly heatmap customization'
            elif view == "3D surface":
                _sz_keys = None
                _lbl = 'Plotly 3D surface customization (text only)'
            else:
                _sz_keys = ('line_profile_width', 'line_profile_height')
                _lbl = 'Plotly line-profile customization'
            render_inline_figure_customization(
                viz_cfg,
                size_keys=_sz_keys,
                key_prefix=f'px_{view[:3]}',
                label=_lbl,
                show_lines=(view == "Line profiles"),
                show_colorbar=(view != "Line profiles"))

            if view == "Heatmap (frame)":
                idx = st.slider("Frame", 0, len(history) - 1,
                                len(history) - 1, key="px_frame")
                field = st.selectbox("Field", list(FIELD_DEFAULT_CMAPS.keys()),
                                      index=2, key="px_field")
                cmap = viz_cfg['field_settings'].get(field, {}).get(
                    'cmap', FIELD_DEFAULT_CMAPS.get(field, 'viridis'))
                fig_h = vis.create_plotly_heatmap(
                    history[idx], field, idx, cmap, viz_cfg=viz_cfg)
                if fig_h is not None:
                    st.plotly_chart(fig_h, use_container_width=True)

            elif view == "3D surface":
                idx = st.slider("Frame", 0, len(history) - 1,
                                len(history) - 1, key="px_3d_frame")
                field = st.selectbox("Field", list(FIELD_DEFAULT_CMAPS.keys()),
                                      index=2, key="px_3d_field")
                show_c = st.checkbox("Show contours", value=True,
                                      key="px_3d_cont")
                cmap = viz_cfg['field_settings'].get(field, {}).get(
                    'cmap', FIELD_DEFAULT_CMAPS.get(field, 'viridis'))
                fig_3d = vis.create_plotly_3d_surface(
                    history[idx], field, idx, cmap,
                    show_contours=show_c, viz_cfg=viz_cfg)
                if fig_3d is not None:
                    st.plotly_chart(fig_3d, use_container_width=True)

            elif view == "Time-scrubbing animation":
                field = st.selectbox("Field", list(FIELD_DEFAULT_CMAPS.keys()),
                                      index=2, key="px_anim_field")
                cmap = viz_cfg['field_settings'].get(field, {}).get(
                    'cmap', FIELD_DEFAULT_CMAPS.get(field, 'viridis'))
                with st.spinner("Building animation frames..."):
                    fig_anim = vis.create_plotly_animation(
                        history, field, params['dt'], cmap, viz_cfg=viz_cfg)
                if fig_anim is not None:
                    st.plotly_chart(fig_anim, use_container_width=True)
                else:
                    st.warning("Could not build animation.")

            else:
                idx = st.slider("Frame", 0, len(history) - 1,
                                len(history) - 1, key="px_lp_frame")
                field = st.selectbox("Field", list(FIELD_DEFAULT_CMAPS.keys()),
                                      index=2, key="px_lp_field")
                ptypes = st.multiselect(
                    "Directions",
                    ["horizontal", "vertical", "diagonal", "anti_diagonal"],
                    default=["horizontal", "vertical"], key="px_lp_types")
                pos = st.slider("Position ratio", 0.0, 1.0, 0.5, 0.05,
                                key="px_lp_pos")
                if ptypes:
                    fig_lp = vis.create_plotly_line_profiles(
                        history[idx], field, ptypes, pos, viz_cfg=viz_cfg)
                    if fig_lp is not None:
                        st.plotly_chart(fig_lp, use_container_width=True)

    # ---- Export tab -----------------------------------------------
    with export_tab:
        if 'history' not in st.session_state:
            st.info("Run the simulation first.")
        else:
            params = st.session_state.params
            history = st.session_state.history
            solver = st.session_state.solver
            sim_data = {'metadata':
                        {'created_at': datetime.now().isoformat()}}
            scenario = params.get('scenario', 'uo2')
            name = (f"uo2_{scenario}_Nx{params['Nx']}_Ny{params['Ny']}"
                    f"_seed{params['seed']}")

            st.subheader("📦 Data export")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Pickle (.pkl)"):
                    buf, fname = DataExporter.export_pkl(
                        sim_data, params, history, name)
                    if buf:
                        st.download_button("Download .pkl", buf, fname)
                if st.button("NumPy (.npz)"):
                    buf, fname = DataExporter.export_npz(
                        sim_data, params, history, name, Nx, Ny, dx, dy)
                    if buf:
                        st.download_button("Download .npz", buf, fname)
                if st.button("JSON"):
                    buf, fname = DataExporter.export_json(
                        sim_data, params, history, name)
                    if buf:
                        st.download_button("Download .json", buf, fname)
            with col2:
                if st.button("CSV (zip of per-frame)"):
                    extent = [-Nx * dx / 2, Nx * dx / 2, -Ny * dy / 2, Ny * dy / 2]
                    buf, fname = DataExporter.export_csv(
                        history, name, extent, Nx, Ny, dx, dy)
                    if buf:
                        st.download_button("Download .zip", buf, fname)
                if st.button("📊 Time-series CSV (porosity, σ, n_voids)"):
                    buf, fname = DataExporter.export_timeseries_csv(
                        solver.history, name)
                    if buf:
                        st.download_button("Download timeseries.csv", buf, fname)
                if st.button("HDF5 (.h5)"):
                    buf, fname = DataExporter.export_hdf5(
                        sim_data, params, history, name, Nx, Ny, dx, dy)
                    if buf:
                        st.download_button("Download .h5", buf, fname)

            st.markdown("---")
            st.subheader("🎬 Animation export")
            an = viz_cfg['animation']
            anim_field = st.selectbox("Field to animate",
                                       list(FIELD_DEFAULT_CMAPS.keys()),
                                       index=2, key="anim_field")
            c1, c2, c3 = st.columns(3)
            with c1:
                a_fmt = st.selectbox("Format", ['gif', 'mp4'],
                                      index=0 if an['format'] == 'gif' else 1,
                                      key="anim_fmt_ui")
            with c2:
                a_fps = st.slider("FPS", 1, 30, int(an['fps']), 1, key="anim_fps_ui")
            with c3:
                a_dpi = st.slider("DPI", 72, 300, int(an['dpi']), 12, key="anim_dpi_ui")
            a_wm = st.checkbox("Watermark", value=an['watermark'], key="anim_wm_ui")
            a_wm_text = st.text_input("Watermark text",
                                       value=an['watermark_text'],
                                       key="anim_wm_text_ui")
            wm_pos_list_ui = ['lower right', 'lower left',
                              'upper right', 'upper left']
            wm_idx_ui = (wm_pos_list_ui.index(an['watermark_pos'])
                         if an['watermark_pos'] in wm_pos_list_ui else 0)
            a_wm_pos = st.selectbox("Watermark position", wm_pos_list_ui,
                                     index=wm_idx_ui, key="anim_wm_pos_ui")

            if st.button("🎬 Generate single-field animation"):
                fs_anim = viz_cfg['field_settings'].get(anim_field, {})
                cmap = fs_anim.get('cmap',
                                    FIELD_DEFAULT_CMAPS.get(anim_field, 'viridis'))
                vis_exp = EnhancedUO2Visualizer(Nx, Ny, dx, dy, dt=params['dt'])
                buf = vis_exp.create_animation_buffer(
                    history, anim_field, params['dt'],
                    output_format=a_fmt, fps=a_fps, dpi=a_dpi,
                    cmap_name=cmap, watermark=a_wm,
                    watermark_text=a_wm_text,
                    watermark_pos=a_wm_pos, viz_cfg=viz_cfg)
                if buf is not None:
                    st.download_button(f"Download .{a_fmt}", buf,
                                        f"{name}_{anim_field}.{a_fmt}")
                else:
                    st.error("Animation generation failed. For MP4, ensure "
                             "ffmpeg is installed.")

            if st.button("📦 Batch generate ALL fields (zip of GIFs)"):
                vis_exp = EnhancedUO2Visualizer(Nx, Ny, dx, dy, dt=params['dt'])
                zip_buf = BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    prog = st.progress(0)
                    fields_all = list(FIELD_DEFAULT_CMAPS.keys())
                    for k, fname in enumerate(fields_all):
                        fs_b = viz_cfg['field_settings'].get(fname, {})
                        cmap = fs_b.get('cmap',
                                         FIELD_DEFAULT_CMAPS.get(fname, 'viridis'))
                        buf = vis_exp.create_animation_buffer(
                            history, fname, params['dt'],
                            output_format='gif', fps=a_fps, dpi=a_dpi,
                            cmap_name=cmap, watermark=a_wm,
                            watermark_text=a_wm_text,
                            watermark_pos=a_wm_pos, viz_cfg=viz_cfg)
                        if buf is not None:
                            zf.writestr(f"{name}_{fname}.gif", buf.getvalue())
                        prog.progress((k + 1) / len(fields_all))
                zip_buf.seek(0)
                st.download_button("Download all GIFs (.zip)", zip_buf,
                                    f"{name}_all_fields.zip")

            st.markdown("---")
            st.subheader("🖼️ Snapshot PNG export")
            snap_field = st.selectbox("Field for snapshot",
                                       list(FIELD_DEFAULT_CMAPS.keys()),
                                       index=2, key="snap_field")
            snap_frame = st.slider("Frame", 0, len(history) - 1,
                                    len(history) - 1, key="snap_frame")
            if st.button("📸 Generate multi-field snapshot PNG"):
                vis_snap = EnhancedUO2Visualizer(Nx, Ny, dx, dy, dt=params['dt'])
                style = {
                    'colormaps': {f: fset.get('cmap',
                                                FIELD_DEFAULT_CMAPS.get(f, 'viridis'))
                                   for f, fset in viz_cfg['field_settings'].items()},
                    'field_settings': viz_cfg['field_settings'],
                    'title_font_size': viz_cfg['title_font_size'],
                    'label_font_size': viz_cfg['label_font_size'],
                    'colorbar_fraction': viz_cfg['colorbar_fraction'],
                    'colorbar_pad': viz_cfg['colorbar_pad'],
                    'figure_customization': viz_cfg.get('figure_customization', {}),
                }
                fig = vis_snap.create_multi_field_comparison(
                    history[snap_frame], style)
                if fig is not None:
                    buf = BytesIO()
                    fig.savefig(buf, format='png', dpi=int(viz_cfg.get('dpi', 150)),
                                bbox_inches='tight')
                    plt.close(fig)
                    buf.seek(0)
                    st.download_button("Download PNG", buf,
                                        f"{name}_frame{snap_frame:04d}.png")


# ============================================================================
# SINGLE VIEW + COMPARE VIEW
# ============================================================================
def _render_single_view(sid, viz_cfg):
    sim = SimulationDatabase.get(sid)
    if sim is None:
        st.error("Simulation not found.")
        return
    st.header(f"Simulation: {sid}")
    params = sim['params']
    history = sim['results_history']
    Nx = params['Nx']; Ny = params['Ny']
    dx = params['dx']; dy = params.get('dy', dx)

    safe_params = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            continue
        if isinstance(v, (np.integer, np.floating, np.bool_)):
            v = v.item()
        if isinstance(v, tuple):
            v = list(v)
        safe_params[k] = v
    st.json(safe_params)

    if history:
        idx = st.slider("Frame", 0, len(history) - 1,
                        len(history) - 1, key=f"view_frame_{sid}")
        vis = EnhancedUO2Visualizer(Nx, Ny, dx, dy, dt=params['dt'])
        style = {
            'colormaps': {f: fset.get('cmap', FIELD_DEFAULT_CMAPS.get(f, 'viridis'))
                           for f, fset in viz_cfg['field_settings'].items()},
            'field_settings': viz_cfg['field_settings'],
            'title_font_size': viz_cfg['title_font_size'],
            'label_font_size': viz_cfg['label_font_size'],
            'colorbar_fraction': viz_cfg['colorbar_fraction'],
            'colorbar_pad': viz_cfg['colorbar_pad'],
            'figure_customization': viz_cfg.get('figure_customization', {}),
        }
        fig = vis.create_multi_field_comparison(history[idx], style)
        if fig is not None:
            st.pyplot(fig); plt.close(fig)

    if st.button("🗑️ Delete"):
        SimulationDatabase.delete(sid)
        if 'view_id' in st.session_state:
            del st.session_state.view_id
        st.rerun()


def _render_compare_view(ids, viz_cfg):
    sims = [SimulationDatabase.get(i) for i in ids]
    sims = [s for s in sims if s is not None]
    if not sims:
        st.warning("No valid simulations selected.")
        return

    st.header(f"Comparing {len(sims)} simulations")
    field = st.selectbox("Field", list(FIELD_DEFAULT_CMAPS.keys()),
                          index=2, key="cmp_field")
    fs_cmp = viz_cfg['field_settings'].get(field, {})
    cmap_name = fs_cmp.get('cmap', FIELD_DEFAULT_CMAPS.get(field, 'viridis'))
    cmap = ColormapRegistry.resolve(cmap_name,
                                     invert=fs_cmp.get('invert', False))

    # Unified bounds across all sims
    all_vals = []
    for sim in sims:
        h = sim['results_history']
        if h and field in h[-1]:
            dd, _ = _rescale_field(field, h[-1][field])
            finite = dd[np.isfinite(dd)]
            if finite.size:
                all_vals.append(finite)

    if not fs_cmp.get('auto', True):
        drng = FIELD_DEFAULT_RNG.get(field, (0.0, 1.0))
        vmin_g = (fs_cmp.get('vmin') if fs_cmp.get('vmin') is not None
                  else drng[0])
        vmax_g = (fs_cmp.get('vmax') if fs_cmp.get('vmax') is not None
                  else drng[1])
    elif field in FIELD_DEFAULT_RNG:
        vmin_g, vmax_g = FIELD_DEFAULT_RNG[field]
    elif all_vals:
        stack = np.concatenate(all_vals)
        vmin_g = float(np.percentile(stack, 2))
        vmax_g = float(np.percentile(stack, 98))
        if field == 'sigma_h':
            zm = max(abs(vmin_g), abs(vmax_g))
            vmin_g, vmax_g = -zm, zm
    else:
        vmin_g, vmax_g = None, None

    cols = min(3, len(sims))
    rows = (len(sims) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i, sim in enumerate(sims):
        p = sim['params']
        h = sim['results_history']
        if not h:
            continue
        last = h[-1]
        if field not in last:
            continue
        d, unit = _rescale_field(field, last[field])
        extent = [-p['Nx'] * p['dx'] / 2, p['Nx'] * p['dx'] / 2,
                  -p['Ny'] * p['dy'] / 2, p['Ny'] * p['dy'] / 2]
        ax = axes[i]
        interp = 'nearest' if field in DISCRETE_FIELDS else 'bilinear'
        im = ax.imshow(d, extent=extent, origin='lower', cmap=cmap,
                       vmin=vmin_g, vmax=vmax_g,
                       interpolation=interp, aspect='equal')
        ax.set_title(f"{sim['id']} — {field}\n"
                     f"({p.get('scenario', '?')}, K={p.get('K_dot', 0):.1e})")
        if field != 'phi_grain':
            plt.colorbar(im, ax=ax, fraction=0.046)
    for k in range(len(sims), len(axes)):
        axes[k].axis('off')
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()





