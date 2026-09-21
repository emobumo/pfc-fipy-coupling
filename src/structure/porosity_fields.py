# -*- coding: utf-8 -*-
"""
Equivalent porosity fields, evaluated at cell centres.

The continuum solve takes one structural input: a porosity per cell. Where
that comes from is a modelling choice made outside the solver -- a binned
particle pack, a prescribed trend, a channel, a random realisation -- and
this module holds the prescribed ones so cases stop hard-coding them.

Every generator takes cell-centre coordinate arrays (x, y) and returns a
porosity array of the same length, already clipped to `bounds`. Composition
is by ordinary array arithmetic: build a base, then overwrite where a
feature applies (see `where_band`).

    phi = linear_gradient(y, 0.12, 0.30, 0.0, 30.0)         # dense bottom, loose top
    phi = np.where(where_band(x, y, (4.0, 0.0), (0.0, 1.0), 1.25), 0.45, phi)

Physical readings of the features:

    linear_gradient   collapsed-fill zoning, dense at the base and loose at
                      the top (bulking factor 1.15-1.40 -> phi ~0.13-0.29)
    where_band        a through-going void band such as the contact between
                      backfill and host rock, or a persistent gap in the
                      fill -- the features the reference project's runaway
                      was traced to
    random_correlated stochastic heterogeneity with a correlation length,
                      anisotropic if the structure has a grain; used for a
                      realisation family rather than a single field

Bounds default to (0.05, 0.60): below ~0.05 the calibrated permeability law
has left the range it was anchored in, and above ~0.6 a "porous medium"
reading stops making sense (a cavity is not a loose pack -- model it as a
feature with its own law, not by pushing phi toward 1).
"""
import numpy as np

DEFAULT_BOUNDS = (0.05, 0.60)


def _clip(phi, bounds):
    lo, hi = bounds
    return np.clip(np.asarray(phi, dtype=float), lo, hi)


def uniform(x, value, bounds=DEFAULT_BOUNDS):
    return _clip(np.zeros(np.asarray(x).shape) + float(value), bounds)


def linear_gradient(y, phi_bottom, phi_top, y_min, y_max, bounds=DEFAULT_BOUNDS):
    """phi varies linearly with height from phi_bottom at y_min to phi_top at y_max."""
    y = np.asarray(y, dtype=float)
    span = float(y_max - y_min)
    if span <= 0.0:
        raise ValueError("y_max must exceed y_min")
    t = (y - float(y_min)) / span
    return _clip(float(phi_bottom) + (float(phi_top) - float(phi_bottom)) * t, bounds)


def where_band(x, y, point, direction, half_width, extent=None):
    """
    Boolean mask of cells within `half_width` of the line through `point`
    along `direction`. `extent=(s_min, s_max)` optionally limits the band to
    a segment, measured along the direction from `point`.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx, dy = float(direction[0]), float(direction[1])
    norm = np.hypot(dx, dy)
    if norm <= 0.0:
        raise ValueError("direction must be non-zero")
    dx, dy = dx / norm, dy / norm
    rx, ry = x - float(point[0]), y - float(point[1])
    along = rx * dx + ry * dy
    perp = np.abs(rx * (-dy) + ry * dx)
    mask = perp <= float(half_width)
    if extent is not None:
        mask = mask & (along >= float(extent[0])) & (along <= float(extent[1]))
    return mask


def random_correlated(x, y, mean, std, corr_x, corr_y, seed,
                      cell=None, bounds=DEFAULT_BOUNDS):
    """
    Gaussian random field with anisotropic correlation lengths, sampled at
    the cell centres of a rectangular grid.

    Builds white noise on the grid the (x, y) centres imply, smooths it with
    a Gaussian kernel whose standard deviations are corr_x/cell and
    corr_y/cell, renormalises to unit variance, and maps to phi = mean +
    std*z before clipping. Long corr_x with short corr_y gives horizontal
    streaks -- channel-like structure without prescribing where.

    `cell` is the grid spacing; inferred from the unique x spacing if None.
    The same seed on the same grid gives the same field.
    """
    from scipy.ndimage import gaussian_filter
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xs, ys = np.unique(x), np.unique(y)
    if cell is None:
        cell = float(np.min(np.diff(xs))) if xs.size > 1 else float(np.min(np.diff(ys)))
    nx, ny = xs.size, ys.size
    ix = np.rint((x - xs[0]) / cell).astype(int)
    iy = np.rint((y - ys[0]) / cell).astype(int)

    rng = np.random.RandomState(int(seed))
    noise = rng.normal(size=(ny, nx))
    sig = (max(float(corr_y) / cell, 1e-6), max(float(corr_x) / cell, 1e-6))
    smooth = gaussian_filter(noise, sigma=sig, mode="reflect")
    s = float(np.std(smooth))
    z = (smooth - float(np.mean(smooth))) / (s if s > 0.0 else 1.0)
    return _clip(float(mean) + float(std) * z[iy, ix], bounds)
