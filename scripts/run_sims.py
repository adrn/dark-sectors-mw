import pathlib
import time
from itertools import product

import astropy.table as at
import astropy.units as u
import gala.dynamics as gd
import gala.potential as gp
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from gala.units import galactic
from schwimmbad import MultiPool
from streamsubhalosim import StreamSubhaloSimulation, get_in_stream_frame


def sim_worker(task):
    i, pars, sim_kw, impact_site, cache_file, overwrite = task
    M_subhalo, impact_v, impact_b_fac, t_post_impact, dxdv = pars

    with h5py.File(cache_file, mode="r") as f:
        if f"{i}" in f.keys() and not overwrite:
            return None

    c_subhalo = 1.005 * u.kpc * (M_subhalo / (1e8 * u.Msun)) ** 0.5 / 2.0  # MAGIC
    impact_b = impact_b_fac * c_subhalo

    # Rotation matrix to impact site coordinates:
    xhat = impact_site.xyz / np.linalg.norm(impact_site.xyz)
    yhat = impact_site.v_xyz / np.linalg.norm(impact_site.v_xyz)
    xhat = xhat - xhat.dot(yhat) * yhat
    zhat = np.cross(xhat, yhat)
    R = np.stack((xhat, yhat, zhat)).T

    dxhat = dxdv[0] / np.linalg.norm(dxdv[0])
    dx = R @ (dxhat * impact_b)
    dvhat = dxdv[1] / np.linalg.norm(dxdv[1])
    dv = R @ (dvhat * impact_v)

    sim = StreamSubhaloSimulation(t_post_impact=t_post_impact, **sim_kw)

    # Buffer time is 32 times the crossing time:
    MAGIC = 32
    print(f"[{i}]: starting simulation...")
    time0 = time.time()
    stream, _ = sim.run_perturbed_stream(
        impact_site_w=impact_site,
        subhalo_impact_dw=gd.PhaseSpacePosition(dx, dv),
        subhalo_potential=gp.HernquistPotential(
            m=M_subhalo, c=c_subhalo, units=galactic
        ),
        t_buffer_impact=np.round((MAGIC * c_subhalo / impact_v).to(u.Myr)),
        impact_dt=np.round((c_subhalo / impact_v / MAGIC).to(u.Myr), decimals=2),
    )

    print(
        f"[{i}]: Simulation done after {time.time() - time0:.1f} seconds - "
        "writing to disk..."
    )

    pars = {
        "M_subhalo": M_subhalo,
        "impact_v": impact_v,
        "impact_b": impact_b,
        "t_post_impact": t_post_impact,
        "dx": dx,
        "dv": dv,
    }
    return f"{i}", cache_file, stream, impact_site, pars


def sim_callback(res):
    if res is None:
        return

    name, cache_file, stream, impact_site, pars = res
    with h5py.File(cache_file, mode="r+") as f:
        if name in f:
            del f[name]

        group = f.create_group(name)

        if len(pars.keys()) > 0:
            t = at.QTable()
            for k, v in pars.items():
                t[k] = [v]
            t.write(group, serialize_meta=True, path="/parameters")

        stream.to_hdf5(group.create_group("stream"))
        impact_site.to_hdf5(group.create_group("impact_site"))


def plot_worker():
    idxs, batch, sim_kw, impact_site, cache_path, overwrite = task

    for i, (M_subhalo, impact_v, impact_b_fac, t_post_impact, dxdv) in zip(
        range(*idxs), batch
    ):
        print(f"[{i}]: Plotting...")
        stream_style = dict(
            marker="o",
            ms=1.0,
            markeredgewidth=0,
            ls="none",
            alpha=0.2,
            plot_function=plt.plot,
        )

        # ------------------------------------------
        # x-y particle plot with impact site marked:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
        stream.plot(["x", "y"], axes=[ax], **stream_style)
        impact_site.plot(
            ["x", "y"], axes=[ax], color="tab:red", autolim=False, zorder=100
        )
        ax.set(xlim=(-25, 25), ylim=(-25, 25))
        fig.savefig(f"{str(plot_filename_base)}-xy.png", dpi=200)
        plt.close(fig)

        # ----------------------------------------
        # sky projection, all simulated particles:
        xlim = (-45, 45)
        stream_sfr = get_in_stream_frame(stream, impact_site)

        lon = stream_sfr.lon.wrap_at(180 * u.deg).degree
        _mask = (lon > xlim[0]) & (lon < xlim[1])

        fig, axes = plt.subplots(
            5, 1, figsize=(16, 20), sharex=True, constrained_layout=True
        )

        comps = ["lat", "distance", "pm_lon_coslat", "pm_lat", "radial_velocity"]
        lims = [(-1, 1), (20, 26), (0, 1), (-0.2, 0.1), (-225, 100)]
        for ax, comp, ylim in zip(axes, comps, lims):
            ax.hist2d(
                lon[_mask],
                getattr(stream_sfr, comp).value[_mask],
                bins=(np.linspace(*xlim, 512), np.linspace(*ylim, 151)),
                norm=mpl.colors.LogNorm(vmin=0.1),
                cmap="Greys",
            )
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.set_ylabel(comp)

        ax = axes[-1]
        ax.text(
            20,
            90,
            f"$M_s = ${M_subhalo:.1e}\n"
            + f"$b = ${impact_b.to_value(u.pc):.1f} {u.pc:latex_inline}\n"
            + f"$∆v = ${impact_v.to_value(u.pc/u.Myr):.1f} {u.pc/u.Myr:latex_inline}\n"
            + f"$∆t = ${t_post_impact.to_value(u.Myr):.0f} {u.Myr:latex_inline}\n"
            + r"$\Delta\hat{x} = "
            + f"${str(dxhat)}\n"
            + r"$\Delta\hat{v} = "
            + f"${str(dvhat)}",
            ha="left",
            va="top",
        )

        axes[-1].set(xlim=xlim, xlabel="longitude [deg]")
        fig.suptitle("all simulated particles", fontsize=22)
        fig.savefig(f"{str(plot_filename_base)}-sky-all.png", dpi=200)
        plt.close(fig)


def main(pool, overwrite=False):
    print(f"Setting up job with n={pool.size} processes...")
    rng = np.random.default_rng(123)

    # Make a cache directory to save the simulation output:
    cache_path = (pathlib.Path(__file__).parent / "../cache").resolve().absolute()
    cache_file = cache_path / "stream-sims.hdf5"
    plot_path = cache_path / "plots"
    plot_path.mkdir(exist_ok=True, parents=True)

    # Ensure that cache file exists"
    mode = "a" if not overwrite else "w"
    with h5py.File(cache_file, mode=mode) as f:
        pass

    # Default potential model:
    mw = gp.load(
        "/mnt/home/apricewhelan/projects/gaia-actions/potentials/"
        "MilkyWayPotential2022.yml"
    )

    # Final phase-space coordinates of the progenitor:
    wf = gd.PhaseSpacePosition(pos=[15, 0.0, 0.0] * u.kpc, vel=[0, 275, 0] * u.km / u.s)

    # Simulation parameters:
    sim_kw = dict(
        mw_potential=mw,
        final_prog_w=wf,
        M_stream=5e4 * u.Msun,
        t_pre_impact=3 * u.Gyr,
        dt=0.25 * u.Myr,
        n_particles=4,
        seed=42,
    )
    print("Setting up simulation instance...")
    sim = StreamSubhaloSimulation(t_post_impact=0 * u.Myr, **sim_kw)

    print("Running initial stream simulation...")
    init_stream, init_prog = sim.run_init_stream()
    print("Finding a good impact site...")
    impact_site = sim.get_impact_site(init_stream, init_prog)

    # Save to cache file:
    sim_callback(("init", cache_file, init_stream, impact_site, {}))

    # Define the grid of subhalo/interaction parameters to run with
    Ms = [5e5, 1e6, 5e6, 1e7] * u.Msun
    vs = [25, 50, 100, 200] * u.pc / u.Myr
    b_facs = [0, 0.5, 1.0, 2.0, 5]
    ts = [100, 200, 400, 800] * u.Myr

    # Some custom geometries for the subhalo at interaction, in a coordinate space
    # defined by dv of the impact site
    rand_dxdvs = [
        ([1.0, 0, 0], [0, 0, 1.0]),
        ([0, 0, 1.0], [1.0, 0, 0]),
        (rng.uniform(size=3), rng.uniform(size=3)),
        (rng.uniform(size=3), rng.uniform(size=3)),
    ]
    par_tasks = list(product(Ms, vs, b_facs, ts, rand_dxdvs))
    par_tasks = par_tasks[:128]  # TODO: REMOVE ME

    sim_tasks = [
        (i, pars, sim_kw, impact_site, cache_file, overwrite)
        for i, pars in enumerate(par_tasks)
    ]

    for _ in pool.map(sim_worker, sim_tasks, callback=sim_callback):
        pass

    # Make plots:
    with h5py.File(cache_file, mode="r") as f:
        sim_keys = f.keys()

    print(f"{len(sim_keys)} simulations to plot...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, default=None)
    parser.add_argument("-o", "--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    with MultiPool(processes=args.nproc) as pool:
        main(pool, overwrite=args.overwrite)
