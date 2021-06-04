#!/usr/bin/env python
"""
This file has a bunch of collected utilities for doing shear measurement tests
with metadetect.
"""
import sys
import time
import logging

import numpy as np
import fitsio

from esutil.numpy_util import combine_arrlist
import ngmix
from metadetect.metadetect import do_metadetect

import multiprocessing
import contextlib

import tqdm
import schwimmbad
import yaml

MDET_CONFIG = yaml.safe_load("""\
metacal:
  psf: fitgauss
  types: [noshear, 1p, 1m, 2p, 2m]
  use_noise_image: True

psf:
  lm_pars:
    maxfev: 2000
    ftol: 1.0e-05
    xtol: 1.0e-05
  model: gauss

  # we try many times because if this fails we get no psf info
  # for the entire patch
  ntry: 10

sx:
  # Minimum contrast parameter for deblending
  deblend_cont: 1.0e-05

  # in sky sigma
  detect_thresh: 0.8

  # minimum number of pixels above threshold
  minarea: 4

  filter_type: conv
  # 7x7 convolution mask of a gaussian PSF with FWHM = 3.0 pixels.
  filter_kernel:
    - [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963]
    - [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388]
    - [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328]
    - [0.068707, 0.296069, 0.710525, 0.951108, 0.710525, 0.296069, 0.068707]
    - [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328]
    - [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388]
    - [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963]

weight:
  fwhm: 1.2  # arcsec

meds:
  box_padding: 2
  box_type: iso_radius
  max_box_size: 64
  min_box_size: 32
  rad_fac: 2
  rad_min: 4

# check for an edge hit
bmask_flags: 536870912  # 2**29
""")


@contextlib.contextmanager
def backend_pool(backend, n_workers=None):
    """Context manager to build a schwimmbad `pool` object with the `map` method.

    Parameters
    ----------
    backend : str
        One of 'sequential', `loky`, or 'mpi'.
    n_workers : int, optional
        The number of workers to use. Defaults to 1 for the 'sequential' backend,
        the cpu count for the 'loky' backend, and the size of the default global
        communicator for the 'mpi' backend.
    """
    try:
        if backend == "sequential":
            pool = schwimmbad.JoblibPool(1, backend=backend, verbose=0)
        else:
            if backend == "mpi":
                from mpi4py import MPI
                pool = schwimmbad.choose_pool(
                    mpi=True,
                    processes=n_workers or MPI.COMM_WORLD.Get_size(),
                )
            else:
                pool = schwimmbad.JoblibPool(
                    n_workers or multiprocessing.cpu_count(),
                    backend=backend,
                    verbose=100,
                )
        yield pool
    finally:
        if "pool" in locals():
            pool.close()


def cut_nones(presults, mresults):
    """Cut entries that are None in a pair of lists. Any entry that is None
    in either list will exclude the item in the other.

    Parameters
    ----------
    presults : list
        One the list of things.
    mresults : list
        The other list of things.

    Returns
    -------
    pcut : list
        The cut list.
    mcut : list
        The cut list.
    """
    prr_keep = []
    mrr_keep = []
    for pr, mr in zip(presults, mresults):
        if pr is None or mr is None:
            continue
        prr_keep.append(pr)
        mrr_keep.append(mr)

    return prr_keep, mrr_keep


def _run_boostrap(x1, y1, x2, y2, wgts):
    rng = np.random.RandomState(seed=100)
    mvals = []
    cvals = []
    for _ in tqdm.trange(500, leave=False):
        ind = rng.choice(len(y1), replace=True, size=len(y1))
        _wgts = wgts[ind].copy()
        _wgts /= np.sum(_wgts)
        mvals.append(np.mean(y1[ind] * _wgts) / np.mean(x1[ind] * _wgts) - 1)
        cvals.append(np.mean(y2[ind] * _wgts) / np.mean(x2[ind] * _wgts))

    return (
        np.mean(y1 * wgts) / np.mean(x1 * wgts) - 1, np.std(mvals),
        np.mean(y2 * wgts) / np.mean(x2 * wgts), np.std(cvals))


def _run_jackknife(x1, y1, x2, y2, wgts, jackknife):
    n_per = x1.shape[0] // jackknife
    n = n_per * jackknife
    x1j = np.zeros(jackknife)
    y1j = np.zeros(jackknife)
    x2j = np.zeros(jackknife)
    y2j = np.zeros(jackknife)
    wgtsj = np.zeros(jackknife)

    loc = 0
    for i in range(jackknife):
        wgtsj[i] = np.sum(wgts[loc:loc+n_per])
        x1j[i] = np.sum(x1[loc:loc+n_per] * wgts[loc:loc+n_per]) / wgtsj[i]
        y1j[i] = np.sum(y1[loc:loc+n_per] * wgts[loc:loc+n_per]) / wgtsj[i]
        x2j[i] = np.sum(x2[loc:loc+n_per] * wgts[loc:loc+n_per]) / wgtsj[i]
        y2j[i] = np.sum(y2[loc:loc+n_per] * wgts[loc:loc+n_per]) / wgtsj[i]

        loc += n_per

    mbar = np.mean(y1 * wgts) / np.mean(x1 * wgts) - 1
    cbar = np.mean(y2 * wgts) / np.mean(x2 * wgts)
    mvals = np.zeros(jackknife)
    cvals = np.zeros(jackknife)
    for i in range(jackknife):
        _wgts = np.delete(wgtsj, i)
        mvals[i] = (
            np.sum(np.delete(y1j, i) * _wgts) / np.sum(np.delete(x1j, i) * _wgts)
            - 1
        )
        cvals[i] = (
            np.sum(np.delete(y2j, i) * _wgts) / np.sum(np.delete(x2j, i) * _wgts)
        )

    return (
        mbar,
        np.sqrt((n - n_per) / n * np.sum((mvals-mbar)**2)),
        cbar,
        np.sqrt((n - n_per) / n * np.sum((cvals-cbar)**2)),
    )


def _estimate_m_and_c(
    presults,
    mresults,
    g_true,
    swap12=False,
    step=0.01,
    weights=None,
    jackknife=None,
):
    """Estimate m and c from paired lensing simulations.

    Parameters
    ----------
    presults : list of iterables
        A list of iterables, each with g1p, g1m, g1, g2p, g2m, g2
        from running metadetect with a `g1` shear in the 1-component and
        0 true shear in the 2-component.
    mresults : list of iterables
        A list of iterables, each with g1p, g1m, g1, g2p, g2m, g2
        from running metadetect with a -`g1` shear in the 1-component and
        0 true shear in the 2-component.
    g_true : float
        The true value of the shear on the 1-axis in the simulation. The other
        axis is assumd to havea true value of zero.
    swap12 : bool, optional
        If True, swap the roles of the 1- and 2-axes in the computation.
    step : float, optional
        The step used in metadetect for estimating the response. Default is
        0.01.
    weights : list of weights, optional
        Weights to apply to each sample. Will be normalized if not already.
    jackknife : int, optional
        The number of jackknife sections to use for error estimation. Default of
        None will do no jackknife and default to bootstrap error bars.

    Returns
    -------
    m : float
        Estimate of the multiplicative bias.
    merr : float
        Estimat of the 1-sigma standard error in `m`.
    c : float
        Estimate of the additive bias.
    cerr : float
        Estimate of the 1-sigma standard error in `c`.
    """

    prr_keep, mrr_keep = cut_nones(presults, mresults)

    def _get_stuff(rr):
        _a = np.vstack(rr)
        g1p = _a[:, 0]
        g1m = _a[:, 1]
        g1 = _a[:, 2]
        g2p = _a[:, 3]
        g2m = _a[:, 4]
        g2 = _a[:, 5]

        if swap12:
            g1p, g1m, g1, g2p, g2m, g2 = g2p, g2m, g2, g1p, g1m, g1

        return (
            g1, (g1p - g1m) / 2 / step * g_true,
            g2, (g2p - g2m) / 2 / step)

    g1p, R11p, g2p, R22p = _get_stuff(prr_keep)
    g1m, R11m, g2m, R22m = _get_stuff(mrr_keep)

    if weights is not None:
        wgts = np.array(weights).astype(np.float64)
    else:
        wgts = np.ones(len(g1p)).astype(np.float64)
    wgts /= np.sum(wgts)

    msk = (
        np.isfinite(g1p) &
        np.isfinite(R11p) &
        np.isfinite(g1m) &
        np.isfinite(R11m) &
        np.isfinite(g2p) &
        np.isfinite(R22p) &
        np.isfinite(g2m) &
        np.isfinite(R22m))
    g1p = g1p[msk]
    R11p = R11p[msk]
    g1m = g1m[msk]
    R11m = R11m[msk]
    g2p = g2p[msk]
    R22p = R22p[msk]
    g2m = g2m[msk]
    R22m = R22m[msk]
    wgts = wgts[msk]

    x1 = (R11p + R11m)/2
    y1 = (g1p - g1m) / 2

    x2 = (R22p + R22m) / 2
    y2 = (g2p + g2m) / 2

    if jackknife:
        return _run_jackknife(x1, y1, x2, y2, wgts, jackknife)
    else:
        return _run_boostrap(x1, y1, x2, y2, wgts)


def estimate_m_and_c(
    pdata,
    mdata,
    g_true=0.02,
    swap12=False,
    step=0.01,
    weights=None,
    jackknife=None,
):
    """Estimate m and c from paired lensing simulations.

    Parameters
    ----------
    pdata : np.ndarray
        The sim data from the plus simulations.
    mdata : np.ndarray
        The sim data form the minus simulations.
    g_true : float, optional
        The true value of the shear on the 1-axis in the simulation. The other
        axis is assumd to havea true value of zero. Defualt value is 0.02.
    swap12 : bool, optional
        If True, swap the roles of the 1- and 2-axes in the computation.
    step : float, optional
        The step used in metadetect for estimating the response. Default is
        0.01.
    weights : list of weights, optional
        Weights to apply to each sample. Will be normalized if not already.
    jackknife : int, optional
        The number of jackknife sections to use for error estimation. Default of
        None will do no jackknife and default to bootstrap error bars.

    Returns
    -------
    m : float
        Estimate of the multiplicative bias.
    merr : float
        Estimat of the 1-sigma standard error in `m`.
    c : float
        Estimate of the additive bias.
    cerr : float
        Estimate of the 1-sigma standard error in `c`.
    """
    pres = [
        (
            pdata["g1p"][i], pdata["g1m"][i], pdata["g1"][i],
            pdata["g2p"][i], pdata["g2m"][i], pdata["g2"][i],
        )
        for i in range(pdata.shape[0])
    ]

    mres = [
        (
            mdata["g1p"][i], mdata["g1m"][i], mdata["g1"][i],
            mdata["g2p"][i], mdata["g2m"][i], mdata["g2"][i],
        )
        for i in range(pdata.shape[0])
    ]

    return _estimate_m_and_c(
        pres,
        mres,
        g_true,
        swap12=swap12,
        step=step,
        weights=weights,
        jackknife=jackknife,
    )


def measure_shear_metadetect(res, *, s2n_cut, t_ratio_cut, ormask_cut, mfrac_cut):
    """Measure the shear parameters for metadetect.

    NOTE: Returns None if nothing can be measured.

    Parameters
    ----------
    res : dict
        The metadetect results.
    s2n_cut : float
        The cut on `wmom_s2n`. Typically 10.
    t_ratio_cut : float
        The cut on `t_ratio_cut`. Typically 1.2.
    ormask_cut : bool
        If True, cut on the `ormask` flags.
    mfrac_cut : float or None
        If not None, cut objects with a masked fraction higher than this
        value.

    Returns
    -------
    g1p : float
        The mean 1-component shape for the plus metadetect measurement.
    g1m : float
        The mean 1-component shape for the minus metadetect measurement.
    g1 : float
        The mean 1-component shape for the zero-shear metadetect measurement.
    g2p : float
        The mean 2-component shape for the plus metadetect measurement.
    g2m : float
        The mean 2-component shape for the minus metadetect measurement.
    g2 : float
        The mean 2-component shape for the zero-shear metadetect measurement.
    """
    def _mask(data):
        _cut_msk = (
            (data['flags'] == 0)
            & (data['wmom_s2n'] > s2n_cut)
            & (data['wmom_T_ratio'] > t_ratio_cut)
        )
        if ormask_cut:
            _cut_msk = _cut_msk & (data['ormask'] == 0)
        if mfrac_cut is not None:
            _cut_msk = _cut_msk & (data["mfrac"] <= mfrac_cut)
        return _cut_msk

    op = res['1p']
    q = _mask(op)
    if not np.any(q):
        return None
    g1p = op['wmom_g'][q, 0]

    om = res['1m']
    q = _mask(om)
    if not np.any(q):
        return None
    g1m = om['wmom_g'][q, 0]

    o = res['noshear']
    q = _mask(o)
    if not np.any(q):
        return None
    g1 = o['wmom_g'][q, 0]
    g2 = o['wmom_g'][q, 1]

    op = res['2p']
    q = _mask(op)
    if not np.any(q):
        return None
    g2p = op['wmom_g'][q, 1]

    om = res['2m']
    q = _mask(om)
    if not np.any(q):
        return None
    g2m = om['wmom_g'][q, 1]

    return (
        np.mean(g1p), np.mean(g1m), np.mean(g1),
        np.mean(g2p), np.mean(g2m), np.mean(g2))


def _run_mdet(obs, seed):
    obs.mfrac = np.zeros_like(obs.image)

    mbobs = ngmix.MultiBandObsList()
    obslist = ngmix.ObsList()
    obslist.append(obs)
    mbobs.append(obslist)

    return do_metadetect(MDET_CONFIG, mbobs, np.random.RandomState(seed=seed))


def _run_sim_pair(args):
    num, backend, sim_func, sim_kwargs, start, seed = args
    pobs = sim_func(g1=0.02, g2=0.0, seed=seed, **sim_kwargs)
    mobs = sim_func(g1=-0.02, g2=0.0, seed=seed, **sim_kwargs)

    pres = _run_mdet(pobs, seed+1024768)
    mres = _run_mdet(mobs, seed+1024769)

    if pres is None or mres is None:
        return None, None

    fkeys = ["g1p", "g1m", "g1", "g2p", "g2m", "g2"]
    dtype = []
    for key in fkeys:
        dtype.append((key, "f8"))

    pgm = measure_shear_metadetect(
        pres, s2n_cut=10, t_ratio_cut=1.2,
        ormask_cut=False, mfrac_cut=None,
    )
    mgm = measure_shear_metadetect(
        mres, s2n_cut=10, t_ratio_cut=1.2,
        ormask_cut=False, mfrac_cut=None,
    )
    if pgm is None or mgm is None:
        return None, None

    datap = [pgm]
    datam = [mgm]

    if backend == "mpi":
        print(
            "[% 10ds] did %04d" % (time.time() - start, num+1),
            flush=True,
        )

    return np.array(datap, dtype=dtype), np.array(datam, dtype=dtype)


def run_mdet_sims(
    sim_func, sim_kwargs, seed, n_sims,
    log_level='warning', backend='sequential', n_workers=None
):
    """Run simulation(s) and analyze them with metadetect.

    Parameters
    ----------
    sim_func : callable
        A function accepting only keyword args with the following signature:

            def sim_func(*, g1, g2, seed, extra kwargs here...):
                # do computations here

        It should make a simulation and return an ngmix observation for it.
        See the shape_measurement_102.ipynb notebook for an example.
    sim_kwargs : dict
        any extra sim kwargs to pass to `sim_func`.
    seed : int
        An RNG seed for seeding the simulations.
    n_sims : int
        The number of simulations to run.
    log_level : str, optional
        The logging level for the sim. Set to 'debug' if you'd like more output.
        Only works if `backend` is 'sequential'.
    backend : str, optional
        Set to 'loky' to run simulations in parallel. The default is 'sequential'.
    n_workers : int, optional
        The number of workers to use when running in parallel. Default of None
        will choose a correct number based on the local system and the setting
        for `backend`.

    Returns
    -------
    pdata : np.ndarray
        The sim data from the plus simulations.
    mdata : np.ndarray
        The sim data form the minus simulations.
    """

    start = time.time()

    if backend == "sequential":
        logging.basicConfig(stream=sys.stdout)
        for code in ["ngmix", "metadetect"]:
            logging.getLogger(code).setLevel(
                getattr(logging, log_level.upper()))

    if backend == "mpi":
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    else:
        rank = 0

    if rank == 0:
        rng = np.random.RandomState(seed=seed)
        sim_rng_seeds = rng.randint(low=1, high=2**29, size=n_sims)

        args = []
        for i, rng_seed in enumerate(sim_rng_seeds):
            args.append((
                i,
                backend,
                sim_func,
                sim_kwargs,
                start,
                rng_seed,
            ))
    else:
        args = []

    with backend_pool(backend, n_workers=n_workers) as pool:
        outputs = pool.map(_run_sim_pair, args)

    if rank == 0:
        pdata, mdata = zip(*outputs)
        pdata, mdata = cut_nones(pdata, mdata)
        if len(pdata) > 0 and len(mdata) > 0:
            pdata = combine_arrlist(list(pdata))
            mdata = combine_arrlist(list(mdata))

            m, msd, c, csd = estimate_m_and_c(
                pdata,
                mdata,
            )

            print("""\
    # of sims: {n_sims}
    noise cancel m   : {m: f} +/- {msd: f} [1e-3, 3-sigma]
    noise cancel c   : {c: f} +/- {csd: f} [1e-5, 3-sigma]""".format(
                    n_sims=len(pdata),
                    m=m/1e-3,
                    msd=msd/1e-3 * 3,
                    c=c/1e-5,
                    csd=csd/1e-5 * 3,
                ),
                flush=True,
            )

            return pdata, mdata
        else:
            return None, None


def write_sim_data(filename, pdata, mdata):
    """Write sim data to a file.

    Parameters
    ----------
    filename : str
        The full path and name of the file to write. The name should end in `.fits`.
    pdata : np.ndarray
        The sim data from the plus simulations.
    mdata : np.ndarray
        The sim data form the minus simulations.
    """
    with fitsio.FITS(filename, 'rw', clobber=True) as fits:
        fits.write(pdata, extname='plus')
        fits.write(mdata, extname='minus')


def read_sim_data(filename):
    """Read sim data from a path.

    Parameters
    ----------
    filename : str
        The full path and name of the file to read. The name should end in `.fits`.

    Returns
    -------
    pdata : np.ndarray
        The sim data from the plus simulations.
    mdata : np.ndarray
        The sim data form the minus simulations.
    """
    with fitsio.FITS(filename, 'r', clobber=True) as fits:
        pdata = fits['plus'].read()
        mdata = fits['minus'].read()

    return pdata, mdata
