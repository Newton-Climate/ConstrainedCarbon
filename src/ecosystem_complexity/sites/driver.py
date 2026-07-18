"""The shared canonical OE inversion driver.

``run_oe_canonical`` is THE single ``optimize_oe`` call behind every canonical
site (Harvard Forest, Barrow, Eight Mile Lake, Howland). It previously lived in
``notebooks/sites/canonical.py`` as ``_run_oe_canonical``, where it was
importable only through a ``sys.path`` hack.

Both the prior and the MAP are run from their *own* analytical steady state —
the exact operating points ``optimize_oe`` costs against — so the returned
diagnostics are self-consistent regardless of site.
"""
from __future__ import annotations

import logging
import time

import jax
import numpy as np

from ecosystem_complexity.api import optimize_oe, run_model
from ecosystem_complexity.oe_utils import ss_state_for_params
from ecosystem_complexity.state import make_default_params

logger = logging.getLogger(__name__)


def run_oe_canonical(
    model, forcing, state0, obs_full,
    extra_blocks: list,
    opt_fields: tuple,
    site_label: str,
) -> dict:
    """Single canonical ``optimize_oe`` call with a structured return.

    Returns the prior and MAP parameter sets, the forward runs at each, the
    steady states they were run from, and the raw ``OEResult``.
    """
    params_prior = make_default_params(model.config)
    logger.info("[%s] prior forward simulation…", site_label)
    # Run the prior from its own analytical steady state — the same operating
    # point optimize_oe costs the prior against (y_prior = _forward(xa)). Using
    # the observed-stock state0 here would produce a "prior" that is not at
    # steady state and does not match the prior the inversion actually sees.
    state_at_prior = ss_state_for_params(model, forcing, state0, params_prior)
    out_prior = run_model(model, forcing, state0=state_at_prior, params=params_prior)
    jax.block_until_ready(out_prior.delta14C)

    logger.info(
        "[%s] optimize_oe fields=%s extras=%s",
        site_label, opt_fields, [b.name for b in extra_blocks],
    )
    t0 = time.perf_counter()
    result = optimize_oe(
        model, forcing, obs_full, state0=state0,
        fields=opt_fields, extra_obs_blocks=extra_blocks,
    )
    ch = np.array(result.cost_history)
    logger.info(
        "  Done [%.1fs]  J %.2f → %.2f  (%d iter, converged=%s)",
        time.perf_counter() - t0, ch[0], ch[-1], result.n_iter, result.converged,
    )

    params_opt = result.params_opt
    state_at_map = ss_state_for_params(model, forcing, state0, params_opt)
    out_opt = run_model(model, forcing, state0=state_at_map, params=params_opt)
    jax.block_until_ready(out_opt.delta14C)

    return {
        "params_prior": params_prior, "params_opt": params_opt,
        "out_prior": out_prior, "out_opt": out_opt,
        "state_at_prior": state_at_prior, "state_at_map": state_at_map,
        "oe_result": result,
    }
