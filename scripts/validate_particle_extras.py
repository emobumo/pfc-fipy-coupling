from __future__ import print_function

import math

import itasca.ball as balls

from src.coupling import driver as drv


def _read_extra(ball, slot_id):
    # Keep this tolerant to minor API naming differences across PFC versions.
    if hasattr(ball, "extra"):
        return ball.extra(slot_id)
    if hasattr(ball, "get_extra"):
        return ball.get_extra(slot_id)
    if hasattr(ball, "extra_get"):
        return ball.extra_get(slot_id)
    raise AttributeError("Cannot read particle extra slot API on ball object")


def _is_scalar_finite(value):
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return False
    return (not math.isnan(fval)) and (not math.isinf(fval))


def _get_vector2_components(value):
    # tuple/list
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return value[0], value[1]

    # ndarray-like / indexable vectors
    try:
        return value[0], value[1]
    except (TypeError, IndexError, KeyError):
        pass

    # PFC vec2-style objects with x/y attrs or methods
    if hasattr(value, "x") and hasattr(value, "y"):
        xval = value.x() if callable(value.x) else value.x
        yval = value.y() if callable(value.y) else value.y
        return xval, yval

    return None


def _is_vector2_finite(value):
    comps = _get_vector2_components(value)
    if comps is None:
        return False
    return _is_scalar_finite(comps[0]) and _is_scalar_finite(comps[1])


def main(sample_count=3, dt=0.01):
    print("Run one cycle to trigger map_back_to_particles() ...")
    drv.run_one_cycle(dt=dt)

    plist = list(balls.list())
    if len(plist) == 0:
        raise RuntimeError("No particles found in current PFC model")

    check_n = min(sample_count, len(plist))
    print("Check extras on {0} particle(s) ...".format(check_n))

    for idx, b in enumerate(plist[:check_n]):
        e1 = _read_extra(b, 1)
        e2 = _read_extra(b, 2)
        e3 = _read_extra(b, 3)
        e4 = _read_extra(b, 4)

        if not _is_scalar_finite(e1):
            raise ValueError("Particle {0}: extra(1) invalid: {1}".format(idx, e1))
        if not _is_vector2_finite(e2):
            raise ValueError("Particle {0}: extra(2) invalid: {1}".format(idx, e2))
        if not _is_scalar_finite(e3):
            raise ValueError("Particle {0}: extra(3) invalid: {1}".format(idx, e3))
        if not _is_scalar_finite(e4):
            raise ValueError("Particle {0}: extra(4) invalid: {1}".format(idx, e4))
        e2x, e2y = _get_vector2_components(e2)

        print(
            "Particle {0}: e1={1:.4e}, e2=({2:.4e},{3:.4e}), e3={4:.4e}, e4={5:.4e}".format(
                idx, float(e1), float(e2x), float(e2y), float(e3), float(e4)
            )
        )

    print("Validation passed: particle extras 1..4 were written and are finite.")


if __name__ == "__main__":
    main()
