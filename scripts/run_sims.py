import glob
import pathlib
import time
from itertools import product

import astropy.table as at
import astropy.units as u
import gala.dynamics as gd
import gala.potential as gp
import h5py
import matplotlib.pyplot as plt
import numpy as np
from gala.units import galactic
from streamsubhaloplot import plot_sky_projections
from streamsubhalosim import (
    StreamSubhaloSimulation,
    get_in_stream_frame,
    get_stream_track,
)


def sim_worker(task):
    i, pars, sim_kw, impact_site, cache_path, overwrite = task
    M_subhalo, impact_v, impact_b_fac, t_post_impact, dxdv = pars

    cache_file = cache_path / f"stream-sim-{i:04d}.hdf5"

    if cache_file.exists() and not overwrite:
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
    stream, _, final_prog = sim.run_perturbed_stream(
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
        "id": i,
        "M_subhalo": M_subhalo,
        "c_subhalo": c_subhalo,
        "impact_v": impact_v,
        "impact_b_fac": impact_b_fac,
        "impact_b": impact_b,
        "t_post_impact": t_post_impact,
        "dx": dx,
        "dv": dv,
        "dx_hat": dxhat,
        "dv_hat": dvhat,
    }

    with h5py.File(cache_file, mode="w") as f:
        if len(pars.keys()) > 0:
            t = at.QTable()
            for k, v in pars.items():
                t[k] = [v]
            t.write(f, serialize_meta=True, path="/parameters")

        stream.to_hdf5(f.create_group("stream"))
        final_prog.to_hdf5(f.create_group("prog"))
        impact_site.to_hdf5(f.create_group("impact_site"))


def plot_worker(task):
    id_, cache_file, stream_frame, tracks, plot_path, overwrite = task

    plot_filename_base = plot_path / f"stream-{id_:04d}"
    filenames = {
        "xy": pathlib.Path(f"{str(plot_filename_base)}-xy.png"),
        "sky-all": pathlib.Path(f"{str(plot_filename_base)}-sky-all.png"),
        "sky-all-dtrack": pathlib.Path(f"{str(plot_filename_base)}-sky-all-dtrack.png"),
    }

    with h5py.File(cache_file, "r") as f:
        # Read subhalo simulation parameters:
        pars = at.QTable.read(cache_file, path="parameters")[0]

        # Read stream and impact site:
        stream = gd.PhaseSpacePosition.from_hdf5(f["stream"])
        impact_site = gd.PhaseSpacePosition.from_hdf5(f["impact_site"])

    print(f"[{id_}]: Plotting...")
    stream_style = dict(
        marker="o",
        ms=1.0,
        markeredgewidth=0,
        ls="none",
        alpha=0.2,
        plot_function=plt.plot,
    )

    par_summary_text = (
        f"$M_s = ${pars['M_subhalo']:.1e}\n"
        + f"$b = ${pars['impact_b'].to_value(u.pc):.1f} {u.pc:latex_inline}\n"
        + f"$∆v = ${pars['impact_v']:.1f} {pars['impact_v'].unit:latex_inline}\n"
        + f"$∆t = ${pars['t_post_impact'].to_value(u.Myr):.0f} {u.Myr:latex_inline}\n"
        + r"$\Delta\hat{x} = "
        + f"${str(pars['dx_hat'])}\n"
        + r"$\Delta\hat{v} = "
        + f"${str(pars['dv_hat'])}"
    )

    if not filenames["xy"].exists() or overwrite:
        # ------------------------------------------
        # x-y particle plot with impact site marked:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
        stream.plot(["x", "y"], axes=[ax], **stream_style)
        impact_site.plot(
            ["x", "y"], axes=[ax], color="tab:red", autolim=False, zorder=100
        )
        ax.set(xlim=(-25, 25), ylim=(-25, 25))
        fig.savefig(filenames["xy"], dpi=200)
        plt.close(fig)

    if not filenames["sky-all"].exists() or overwrite:
        # ----------------------------------------
        # sky projection, all simulated particles:
        stream_sfr = get_in_stream_frame(stream, stream_frame=stream_frame)

        fig, axes = plot_sky_projections(stream_sfr)
        ax = axes[-1]
        ax.text(
            20,
            ax.get_ylim()[1] * 0.9,
            par_summary_text,
            ha="left",
            va="top",
        )
        ax.set_xlabel("longitude [deg]")
        fig.suptitle("all simulated particles", fontsize=22)
        fig.savefig(filenames["sky-all"], dpi=200)
        plt.close(fig)

    if not filenames["sky-all-dtrack"].exists() or overwrite:
        # ------------------------------------------------------------
        # sky projection, all simulated particles, relative to tracks:
        stream_sfr = get_in_stream_frame(stream, stream_frame=stream_frame)

        fig, axes = plot_sky_projections(stream_sfr, tracks=tracks)
        ax = axes[-1]
        ax.text(
            20,
            ax.get_ylim()[1] * 0.9,
            par_summary_text,
            ha="left",
            va="top",
        )
        ax.set_xlabel("longitude [deg]")
        fig.suptitle("all simulated particles", fontsize=22)
        fig.savefig(filenames["sky-all-dtrack"], dpi=200)
        plt.close(fig)


def main(pool, dist, overwrite=False):
    print(f"Setting up job with n={pool.size} processes...")
    rng = np.random.default_rng(123)

    # Make a cache directory to save the simulation output:
    root_cache_path = (pathlib.Path(__file__).parent / "../cache").resolve().absolute()
    root_cache_path = root_cache_path / "dist-{:.0f}kpc".format(dist)
    plot_path = root_cache_path / "plots"
    plot_path.mkdir(exist_ok=True, parents=True)
    cache_path = root_cache_path / "sims"
    cache_path.mkdir(exist_ok=True, parents=True)
    meta_path = root_cache_path / "stream-sims-metadata.fits"

    # Default potential model:
    mw = gp.load(
        "/mnt/home/apricewhelan/projects/gaia-actions/potentials/"
        "MilkyWayPotential2022.yml"
    )

    # Final phase-space coordinates of the progenitor:
    pos = [-8, 0, dist] * u.kpc
    vcirc = mw.circular_velocity(pos)[0]
    wf = gd.PhaseSpacePosition(pos=pos, vel=[0, 1.3, 0] * vcirc)

    # Simulation parameters:
    sim_kw = dict(
        mw_potential=mw,
        final_prog_w=wf,
        M_stream=8e4 * u.Msun,
        t_pre_impact=4 * u.Gyr,
        dt=0.5 * u.Myr,
        n_particles=4,
        seed=42,
    )

    init_cache_file = cache_path / "stream-sim-init.hdf5"

    if not init_cache_file.exists():
        print("Setting up simulation instance...")
        sim = StreamSubhaloSimulation(t_post_impact=0 * u.Myr, **sim_kw)

        print("Running initial stream simulation...")
        init_stream, init_prog = sim.run_init_stream()
        print("Finding a good impact site...")
        impact_site = sim.get_impact_site(init_stream, init_prog)

        with h5py.File(init_cache_file, mode="w") as f:
            init_stream.to_hdf5(f.create_group("stream"))
            init_prog.to_hdf5(f.create_group("prog"))
            impact_site.to_hdf5(f.create_group("impact_site"))

    else:
        with h5py.File(init_cache_file, mode="r") as f:
            impact_site = gd.PhaseSpacePosition.from_hdf5(f["impact_site"])

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

    sim_tasks = [
        (i, pars, sim_kw, impact_site, cache_path, overwrite)
        for i, pars in enumerate(par_tasks)
    ]

    for _ in pool.map(sim_worker, sim_tasks):
        pass

    # Make a summary table with the simulation parameters:
    print("Making metadata table...")

    allfilenames = []
    allpars = []
    for filename in glob.glob(str(cache_path / "stream-sim-*.hdf5")):
        filename = pathlib.Path(filename)
        if "init" in filename.parts[-1]:
            continue

        pars = at.QTable.read(filename, path="/parameters")
        allpars.append(pars)
        allfilenames.append(filename)

    allpars = at.vstack(allpars)
    allpars.write(meta_path, overwrite=True)

    # ---------------------------------------------------------------------------------
    # Make plots:
    with h5py.File(init_cache_file, mode="r") as f:
        stream = gd.PhaseSpacePosition.from_hdf5(f["stream"])
        prog = gd.PhaseSpacePosition.from_hdf5(f["prog"])
        impact_site = gd.PhaseSpacePosition.from_hdf5(f["impact_site"])

    print(f"{len(allfilenames)} simulations to plot...")

    stream_sfr = get_in_stream_frame(stream, impact=impact_site, prog=prog)
    tracks = get_stream_track(stream_sfr, lon_lim=(-45, 45))

    plot_tasks = [
        (
            pars["id"],
            cache_file,
            stream_sfr.replicate_without_data(),
            tracks,
            plot_path,
            overwrite,
        )
        for pars, cache_file in zip(allpars, allfilenames)
    ]

    for _ in pool.map(plot_worker, plot_tasks):
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--nproc", type=int, default=None)
    grp.add_argument("--mpi", action="store_true", default=False)

    parser.add_argument("--dist", type=float, default=None, required=True)
    parser.add_argument("-o", "--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    if args.mpi:
        from custommpipool import MPIPoolExecutor

        Pool = MPIPoolExecutor
        Pool_kw = dict()
    elif args.nproc is not None:
        from schwimmbad import MultiPool

        Pool = MultiPool
        Pool_kw = dict(processes=args.nproc)
    else:
        from schwimmbad import SerialPool

        Pool = SerialPool
        Pool_kw = dict()

    with Pool(**Pool_kw) as pool:
        main(pool, dist=args.dist, overwrite=args.overwrite)
