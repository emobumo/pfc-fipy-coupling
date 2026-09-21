# -*- coding: utf-8 -*-
"""
Fill diagnostics: where the grout could go, where it went, and what that says.

Pure post-processing on a solver state. Nothing here feeds back into the
solve, changes a field, or depends on solver internals beyond reading the
same (k, phi) the solver reads.

THE THREE NESTED DOMAINS
------------------------
Every "fill ratio" needs a denominator, and the choice of denominator is the
diagnosis. Three nested regions answer three different questions:

    target      the design volume the engineer promised to grout
                (given from outside -- it cannot be computed)
    reachable   where the grout CAN go under the stall condition
                integral(lambda ds) <= p0 + rho*g*(y_src - y)
    filled      where it DID go (S >= S_c)

and the differences between them are the findings:

    target \ reachable    design shortfall  -- pressure, spacing or drilling
                                               increment too small; the grout
                                               physically cannot get there
    reachable \ filled    fill shortfall    -- it could, and did not
        enclosed by filled cells    bypass void   (the real defect)
        connected to open ground    front shortfall (time, or numerics)

REACHABLE DOMAIN IS A SHORTEST PATH, NOT A CIRCLE
-------------------------------------------------
The stall condition is a path integral of the start-up gradient. In a
heterogeneous field the reachable region is therefore the set of cells whose
minimum-cost path from the source stays within the pressure budget, with
cost per face lambda_f * ds, plus rho*g*dy for the climb. That is a Dijkstra
problem on the cell graph. Gravity makes it directed (down is cheaper than
up); with Pi_g = rho*g/lambda < 1 every edge stays positive and Dijkstra
applies, otherwise Bellman-Ford takes over (a round trip always costs
2*lambda*ds > 0, so there are no negative cycles).

This is the same path-integral argument that explains why a 3.14x span in
local L_max produces only a 1.26x asymmetry: the reach is set by the
resistance along the way, not at the destination.

RUNAWAY UNDER CONSTANT PRESSURE
-------------------------------
In the field the runaway criterion is "pressure fails to build" under a
constant-rate pump. Under a constant-pressure source the dual is the
injection rate failing to decay: in a sealed finite domain Q(t) -> 0 as the
front stalls, and a plateau means the grout keeps finding somewhere to go.

Units follow the solver: 2D, volumes per unit thickness [m^3/m], rates
[m^3/m/s].
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, bellman_ford


# Per-cell fill categories returned by classify_unfilled.
FILLED = 0
UNREACHABLE = 1        # outside the reachable domain: not a defect
FRONT_SHORTFALL = 2    # reachable, unfilled, connected to open ground
BYPASS_VOID = 3        # reachable, unfilled, enclosed by filled cells
OVERSHOOT = 4          # filled, but outside the reachable domain

CATEGORY_NAMES = {
    FILLED: "filled",
    UNREACHABLE: "unreachable",
    FRONT_SHORTFALL: "front_shortfall",
    BYPASS_VOID: "bypass_void",
    OVERSHOOT: "overshoot",
}


def _to_array(value_or_var):
    if hasattr(value_or_var, "value"):
        return np.asarray(value_or_var.value, dtype=float)
    return np.asarray(value_or_var, dtype=float)


def _as_index_array(cells, n):
    """Accept a boolean mask or an index array; return sorted unique indices."""
    arr = np.asarray(cells)
    if arr.dtype == np.bool_:
        return np.where(arr)[0]
    return np.unique(arr.astype(int))


# --- mesh graph ------------------------------------------------------------

def interior_face_pairs(mesh):
    """(faces, cell_a, cell_b, centre_distance) for every interior face."""
    ids = mesh.faceCellIDs
    a_raw = ids[0]
    b_raw = ids[1]
    if hasattr(a_raw, "filled"):
        a_raw = a_raw.filled(0)
    if hasattr(b_raw, "filled"):
        b_raw = b_raw.filled(0)
    a = np.asarray(a_raw).astype(int)
    b = np.asarray(b_raw).astype(int)
    interior = np.logical_not(np.asarray(mesh.exteriorFaces.value, dtype=bool))
    faces = np.where(interior)[0]
    a, b = a[faces], b[faces]
    centres = np.asarray(mesh.cellCenters.value, dtype=float)
    dist = np.sqrt(np.sum((centres[:, a] - centres[:, b]) ** 2, axis=0))
    return faces, a, b, dist


def exterior_adjacent_cells(mesh):
    """Boolean mask of cells that own at least one exterior face."""
    ids = mesh.faceCellIDs
    a_raw = ids[0]
    if hasattr(a_raw, "filled"):
        a_raw = a_raw.filled(0)
    a = np.asarray(a_raw).astype(int)
    exterior = np.asarray(mesh.exteriorFaces.value, dtype=bool)
    mask = np.zeros(mesh.numberOfCells, dtype=bool)
    mask[a[exterior]] = True
    return mask


# --- start-up gradient on faces ------------------------------------------

def face_start_gradient(state):
    """
    lambda_f on every face, built exactly as the threshold solver builds it:
    harmonic face permeability, arithmetic face porosity,
    lambda = 2*tau0 / sqrt(8 k_f / phi_f).
    """
    params = state["slurry_parameters"]
    tau0 = float(params.get("yield_stress", 50.0))
    k_face = np.maximum(_to_array(state["permeability"].harmonicFaceValue), 1.0e-30)
    phi_face = np.clip(
        _to_array(state["porosity"].arithmeticFaceValue), 1.0e-6, 1.0 - 1.0e-6
    )
    return 2.0 * tau0 / np.maximum(np.sqrt(8.0 * k_face / phi_face), 1.0e-20)


# --- reachable domain ------------------------------------------------------

def reachable_domain(state, source_cells, p0=None, use_gravity=True):
    """
    Cells the grout can reach from the source under the stall condition.

    Cost to cross a face from cell i to cell j:

        lambda_f * |x_j - x_i|  +  rho*g*(y_j - y_i)        [Pa]

    A cell is reachable when the cheapest path from any source cell costs no
    more than p0. The gravity term is signed, so descending is cheaper and
    climbing dearer, exactly as Phi = p + rho*g*y says.

    Returns a dict:
        cost            min path cost per cell [Pa]  (inf if disconnected)
        reachable       boolean mask, cost <= p0
        budget          p0 used
        negative_edges  how many directed edges had cost < 0 (Pi_g >= 1)
        solver          "dijkstra" or "bellman_ford"
    """
    mesh = state["mesh"]
    params = state["slurry_parameters"]
    n = mesh.numberOfCells
    src = _as_index_array(source_cells, n)
    if p0 is None:
        p0 = float(state.get("interior_dirichlet_value",
                             params.get("inlet_pressure_core_value", 0.0)))

    faces, a, b, dist = interior_face_pairs(mesh)
    lam = face_start_gradient(state)[faces]
    yield_cost = lam * dist

    if use_gravity:
        rho = float(params.get("slurry_density", 2000.0))
        g = abs(float(params.get("gravity_y", -9.81)))
        y = np.asarray(mesh.cellCenters.value, dtype=float)[1]
        climb = rho * g * (y[b] - y[a])
    else:
        climb = np.zeros_like(yield_cost)

    w_ab = yield_cost + climb      # a -> b
    w_ba = yield_cost - climb      # b -> a
    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a])
    data = np.concatenate([w_ab, w_ba])
    negative = int(np.sum(data < 0.0))

    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    if negative == 0:
        solver = "dijkstra"
        d = dijkstra(graph, directed=True, indices=src)
    else:
        solver = "bellman_ford"
        d = bellman_ford(graph, directed=True, indices=src)
    cost = np.min(np.atleast_2d(d), axis=0)
    cost[src] = 0.0
    return {
        "cost": cost,
        "reachable": cost <= float(p0),
        "budget": float(p0),
        "negative_edges": negative,
        "solver": solver,
    }


# --- classify what is unfilled --------------------------------------------

def classify_unfilled(state, source_cells, reachable, s_c=0.5, target_mask=None):
    """
    Label every cell as filled / unreachable / front_shortfall / bypass_void.

    Unfilled cells (S < s_c) are grouped into face-connected components. A
    component is a BYPASS VOID when it lies entirely inside the reachable
    domain and touches no exterior face: the grout could have filled it,
    went around it instead, and sealed it in. Anything else unfilled but
    reachable is a FRONT SHORTFALL -- still connected to open ground, so the
    front simply has not (or not yet) got there. Unfilled cells outside the
    reachable domain are UNREACHABLE and are not counted against the grout.

    target_mask (optional) is the design volume; when given, the same
    fractions are also reported over it, plus the design shortfall
    target \ reachable.

    Filled cells OUTSIDE the reachable domain are OVERSHOOT: the solver put
    grout where the path integral says the budget had run out. On a stalled
    Bingham front that is the regularized-yield creep the ladder documents
    (a cell or so, growing with run time); in a channelled field it would
    mean the reach model is missing a path. Either way it is worth seeing.

    Returns a dict with per-cell `category`, component `labels` (-1 for
    filled), and area/fraction tables keyed by category name.
    """
    mesh = state["mesh"]
    n = mesh.numberOfCells
    vols = np.asarray(mesh.cellVolumes, dtype=float)
    s = np.clip(_to_array(state["saturation"]), 0.0, 1.0)
    src = _as_index_array(source_cells, n)
    reach = np.asarray(reachable, dtype=bool)

    filled = s >= float(s_c)
    filled[src] = True
    unfilled = np.logical_not(filled)

    # Components of the unfilled set, connected through interior faces.
    _, a, b, _ = interior_face_pairs(mesh)
    keep = unfilled[a] & unfilled[b]
    adj = csr_matrix(
        (np.ones(int(np.sum(keep))), (a[keep], b[keep])), shape=(n, n)
    )
    n_comp, comp = connected_components(adj, directed=False)
    labels = np.where(unfilled, comp, -1)

    touches_exterior = exterior_adjacent_cells(mesh)
    category = np.zeros(n, dtype=int)
    category[filled & np.logical_not(reach)] = OVERSHOOT
    category[unfilled & np.logical_not(reach)] = UNREACHABLE
    category[unfilled & reach] = FRONT_SHORTFALL
    for c in np.unique(comp[unfilled]):
        members = unfilled & (comp == c)
        enclosed = np.all(reach[members]) and not np.any(touches_exterior[members])
        if enclosed:
            category[members] = BYPASS_VOID

    def _table(mask):
        total = float(np.sum(vols[mask]))
        out = {"area": total}
        for code, name in CATEGORY_NAMES.items():
            area = float(np.sum(vols[mask & (category == code)]))
            out[name] = area
            out[name + "_fraction"] = area / total if total > 0.0 else 0.0
        return out

    result = {
        "s_c": float(s_c),
        "category": category,
        "labels": labels,
        "n_components": int(n_comp),
        "n_bypass_voids": int(len(np.unique(comp[category == BYPASS_VOID]))),
        "over_reachable": _table(reach),
        "over_domain": _table(np.ones(n, dtype=bool)),
    }
    if target_mask is not None:
        tgt = np.asarray(target_mask, dtype=bool)
        result["over_target"] = _table(tgt)
        result["design_shortfall_area"] = float(
            np.sum(vols[tgt & np.logical_not(reach)])
        )
        total = float(np.sum(vols[tgt]))
        result["design_shortfall_fraction"] = (
            result["design_shortfall_area"] / total if total > 0.0 else 0.0
        )
    return result


def fill_ratio_ladder(state, source_cells, reachable, thresholds=(0.3, 0.5, 0.7),
                      target_mask=None):
    """classify_unfilled at several S_c, as a list of (s_c, result)."""
    return [
        (s_c, classify_unfilled(state, source_cells, reachable, s_c=s_c,
                                target_mask=target_mask))
        for s_c in thresholds
    ]


# --- injection rate --------------------------------------------------------

def record_injection_rate(state, source_cells, dt, t, history):
    """
    Append (t, Q) to `history` after a transport step. Q is the instantaneous
    net rate leaving the source cells, from the same explicit flux field the
    ledger uses [m^3/m/s]. Call once per accepted step, in the marching loop.
    """
    mesh = state["mesh"]
    vols = np.asarray(mesh.cellVolumes, dtype=float)
    src = _as_index_array(source_cells, mesh.numberOfCells)
    div_q = state.get("last_div_q")
    if div_q is None:
        return history
    div_q = np.asarray(div_q, dtype=float)
    q = float(np.sum(div_q[src] * vols[src]))
    history.append((float(t), q))
    return history


def injection_rate_diagnostics(history, tail_fraction=0.2, stall_ratio=0.05):
    """
    Has the injection rate decayed the way a stalling front makes it decay?

    Returns peak and tail statistics plus two readings:
        decay_ratio     mean Q over the last tail_fraction of the record,
                        divided by peak Q
        tail_slope      d(ln Q)/dt over the tail (negative = still decaying;
                        ~0 = plateau)
        stalled         decay_ratio < stall_ratio
    A plateau well above stall_ratio, held for a long time, is the
    constant-pressure signature of runaway: the grout keeps finding
    somewhere to go.
    """
    if len(history) < 2:
        return {"n": len(history), "stalled": False}
    t = np.array([h[0] for h in history], dtype=float)
    q = np.array([h[1] for h in history], dtype=float)
    q_peak = float(np.max(q))
    i_peak = int(np.argmax(q))
    n_tail = max(2, int(round(tail_fraction * len(q))))
    t_tail, q_tail = t[-n_tail:], q[-n_tail:]
    q_tail_mean = float(np.mean(q_tail))
    decay_ratio = q_tail_mean / q_peak if q_peak > 0.0 else 0.0

    pos = q_tail > 0.0
    if int(np.sum(pos)) >= 2 and (t_tail[pos][-1] - t_tail[pos][0]) > 0.0:
        slope = float(np.polyfit(t_tail[pos], np.log(q_tail[pos]), 1)[0])
    else:
        slope = 0.0
    return {
        "n": int(len(q)),
        "t_final": float(t[-1]),
        "q_peak": q_peak,
        "t_peak": float(t[i_peak]),
        "q_final": float(q[-1]),
        "q_tail_mean": q_tail_mean,
        "decay_ratio": decay_ratio,
        "tail_slope": slope,
        "stalled": bool(decay_ratio < stall_ratio),
    }


# --- volume / distance consistency ----------------------------------------

def volume_distance_consistency(state, source_cells, v_in, r_obs=None):
    """
    Did the injected volume travel further than uniform filling would allow?

    Sort cells by Euclidean distance from the nearest source cell and
    accumulate pore volume n*V outward. r_equiv is the distance uniform
    filling of v_in would reach; v_required(r_obs) is what uniform filling
    out to the observed spread would have needed. reach_ratio = r_obs/r_equiv
    well above 1 says the grout channelled: it got far without filling what
    it passed. This automates the argument that a per-hole volume sufficient
    for ~5 m of uniform fill turned up ~37 m away.
    """
    mesh = state["mesh"]
    n = mesh.numberOfCells
    vols = np.asarray(mesh.cellVolumes, dtype=float)
    phi = np.clip(_to_array(state["porosity"]), 0.0, 1.0)
    src = _as_index_array(source_cells, n)
    centres = np.asarray(mesh.cellCenters.value, dtype=float)

    d = np.full(n, np.inf)
    for c in src:
        dc = np.sqrt(np.sum((centres - centres[:, c:c + 1]) ** 2, axis=0))
        d = np.minimum(d, dc)
    d[src] = 0.0

    order = np.argsort(d)
    cum = np.cumsum((phi * vols)[order])
    d_sorted = d[order]
    v_in = float(v_in)

    if v_in <= 0.0:
        r_equiv = 0.0
    elif v_in >= cum[-1]:
        r_equiv = float(d_sorted[-1])
    else:
        i = int(np.searchsorted(cum, v_in))
        r_equiv = float(d_sorted[i])

    out = {
        "v_in": v_in,
        "r_equiv": r_equiv,
        "pore_capacity_total": float(cum[-1]),
    }
    if r_obs is not None:
        within = d <= float(r_obs)
        v_req = float(np.sum((phi * vols)[within]))
        out["r_obs"] = float(r_obs)
        out["v_required_at_r_obs"] = v_req
        out["volume_ratio"] = v_in / v_req if v_req > 0.0 else np.inf
        out["reach_ratio"] = float(r_obs) / r_equiv if r_equiv > 0.0 else np.inf
    return out


def observed_spread(state, source_cells, s_c=0.5):
    """Max over filled cells (S >= s_c, source excluded) of the distance to
    the NEAREST source cell -- how far the grout actually got."""
    mesh = state["mesh"]
    n = mesh.numberOfCells
    s = np.clip(_to_array(state["saturation"]), 0.0, 1.0)
    src = _as_index_array(source_cells, n)
    centres = np.asarray(mesh.cellCenters.value, dtype=float)
    filled = np.setdiff1d(np.where(s >= float(s_c))[0], src)
    if filled.size == 0:
        return 0.0
    d = np.full(filled.size, np.inf)
    for c in src:
        dc = np.sqrt(np.sum((centres[:, filled] - centres[:, c:c + 1]) ** 2, axis=0))
        d = np.minimum(d, dc)
    return float(np.max(d))
