import glob
import pathlib
import time
from itertools import product

import astropy.table as at
import astropy.units as u
import gala.dynamics as gd
import gala.integrate as gi
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
    get_subhalo_w0,
)


def sim_worker(task):
    i, pars, sim_kw, impact_site, cache_path, overwrite, *plot_args = task
    M_subhalo, t_post_impact, impact_b_fac, phi, vphi, vz = pars

    cache_file = cache_path / f"stream-sim-{i:04d}.hdf5"

    if not cache_file.exists() or overwrite:
        sim = StreamSubhaloSimulation(t_post_impact=t_post_impact, **sim_kw)

        impact_site_at_impact = sim.H.integrate_orbit(
            impact_site,
            dt=-sim.dt,
            t1=sim.t_pre_impact + sim.t_post_impact,
            t2=sim.t_pre_impact,
            Integrator=gi.DOPRI853Integrator,
        )[-1]

        # HACK: factor of 2.0 = denser than CDM!
        c_subhalo = 1.005 * u.kpc * (M_subhalo / (1e8 * u.Msun)) ** 0.5 / 2.0
        subhalo_potential = gp.HernquistPotential(
            m=M_subhalo, c=c_subhalo, units=galactic
        )

        impact_b = impact_b_fac * c_subhalo

        subhalo_w0 = get_subhalo_w0(
            impact_site_at_impact, b=impact_b, phi=phi, vphi=vphi, vz=vz
        )

        # Compute "buffer" time duration and timestep
        # Buffer time is 32 times the crossing time:
        BUFFER_N = 32
        subhalo_dv = np.linalg.norm(subhalo_w0.v_xyz - impact_site.v_xyz)
        subhalo_dx = np.max(u.Quantity([impact_b, c_subhalo]))
        t_buffer_impact = np.round(
            (BUFFER_N * subhalo_dx / subhalo_dv).to(u.Myr), decimals=0
        )
        impact_dt = np.round((t_buffer_impact / 256).to(u.Myr), decimals=1)

        print(f"[{i}]: starting simulation...")
        time0 = time.time()
        stream, _, final_prog, final_t = sim.run_perturbed_stream(
            subhalo_w0, subhalo_potential, t_buffer_impact, impact_dt
        )

        print(
            f"[{i}]: Simulation done after {time.time() - time0:.1f} seconds - "
            "writing to disk..."
        )

        pars = {
            "id": i,
            "M_subhalo": M_subhalo,
            "c_subhalo": c_subhalo,
            "impact_b_fac": impact_b_fac,
            "impact_b": impact_b,
            "phi": phi,
            "vphi": vphi,
            "vz": vz,
            "t_post_impact": t_post_impact,
            "filename": str(cache_file),
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

    plot_worker(i, str(cache_file), *plot_args)


def plot_worker(id_, cache_file, stream_frame, tracks, plot_path, overwrite):
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
        f"$M_s = ${pars['M_subhalo'].value:.1e} {pars['M_subhalo'].unit:latex_inline}\n"
        + f"$b = {pars['impact_b'].to_value(u.pc):.1f}$ {u.pc:latex_inline}\n"
        + r"$\phi = "
        + f"{pars['phi'].to_value(u.deg):.1f}$ deg\n"
        + r"$v_{\phi} = "
        + f"{pars['vphi'].value:.1f}$ {pars['vphi'].unit:latex_inline}\n"
        + r"$v_{z} = "
        + f"{pars['vz'].value:.1f}$ {pars['vz'].unit:latex_inline}\n"
        + f"$∆t = {pars['t_post_impact'].to_value(u.Myr):.0f}$ {u.Myr:latex_inline}\n"
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
            -42,
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

        ylims = [(-2, 2), (-1.5, 1.5), (-0.15, 0.15), (-0.15, 0.15), (-12, 12)]
        for ax, ylim in zip(axes, ylims):
            ax.set_ylim(ylim)

        ax.text(
            -42,
            ax.get_ylim()[1] * 0.9,
            par_summary_text,
            ha="left",
            va="top",
        )

        ax.set_xlabel("longitude [deg]")
        fig.suptitle("all simulated particles", fontsize=22)
        fig.savefig(filenames["sky-all-dtrack"], dpi=200)
        plt.close(fig)


def main(pool, dist, overwrite=False, overwrite_plots=False):
    print(f"Setting up job with n={pool.size} processes...")

    # Make a cache directory to save the simulation output:
    root_cache_path = (pathlib.Path(__file__).parent / "../cache").resolve().absolute()
    root_cache_path = root_cache_path / f"dist-{dist:.0f}kpc"
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
        M_stream=5e4 * u.Msun,
        t_pre_impact=5 * u.Gyr,
        dt=0.25 * u.Myr,
        n_particles=5,
        seed=42,
    )

    init_cache_file = cache_path / "stream-sim-init.hdf5"

    if not init_cache_file.exists():
        print("Setting up simulation instance...")
        sim = StreamSubhaloSimulation(t_post_impact=0 * u.Myr, **sim_kw)

        print("Running initial stream simulation...")
        (init_stream, init_prog), _ = sim.run_init_stream()
        print("Finding a good impact site...")
        impact_site = sim.get_impact_site(init_stream, init_prog, prog_dist=10 * u.kpc)

        with h5py.File(init_cache_file, mode="w") as f:
            init_stream.to_hdf5(f.create_group("stream"))
            init_prog[0].to_hdf5(f.create_group("prog"))
            impact_site.to_hdf5(f.create_group("impact_site"))

    with h5py.File(init_cache_file, mode="r") as f:
        init_stream = gd.PhaseSpacePosition.from_hdf5(f["stream"])
        init_prog = gd.PhaseSpacePosition.from_hdf5(f["prog"])
        impact_site = gd.PhaseSpacePosition.from_hdf5(f["impact_site"])

    stream_sfr = get_in_stream_frame(init_stream, impact=impact_site, prog=init_prog)
    tracks = get_stream_track(stream_sfr, lon_lim=(-45, 45))
    stream_frame = stream_sfr.replicate_without_data()

    # Define the grid of subhalo/interaction parameters to run with
    # ts = [50, 100, 200, 400, 800] * u.Myr
    # Ms = [5e5, 1e6, 5e6, 1e7] * u.Msun
    # b_facs = [0, 0.5, 1.0, 2.0, 5]
    # phis = np.arange(0, 180 + 1, 45) * u.deg
    # vphis = [25, 50, 100, 200] * u.pc / u.Myr
    # vzs = [-50, 0, 50] * u.pc / u.Myr
    # par_tasks = list(product(Ms, ts, b_facs, phis, vphis, vzs))

    # HACK FOR TESTING:
    ts = [50] * u.Myr
    Ms = [5e5, 1e7] * u.Msun
    b_facs = [0.5]
    phis = [0] * u.deg
    vphis = [50] * u.pc / u.Myr
    vzs = [0] * u.pc / u.Myr
    par_tasks = list(product(Ms, ts, b_facs, phis, vphis, vzs))

    print(f"Running {len(par_tasks)} simulations...")
    sim_tasks = [
        (
            i,
            pars,
            sim_kw,
            impact_site,
            cache_path,
            overwrite,
            stream_frame,
            tracks,
            plot_path,
            overwrite_plots,
        )
        for i, pars in enumerate(par_tasks)
    ]

    for _ in pool.map(sim_worker, sim_tasks):
        pass

    # Make a summary table with the simulation parameters:
    allfilenames = sorted(glob.glob(str(cache_path / "stream-sim-*.hdf5")))

    make_meta = False
    if meta_path.exists():
        allpars = at.QTable.read(meta_path)
        if (len(allpars) != (len(allfilenames) - 1)) or overwrite:
            make_meta = True
        else:
            print("Metadata table already exists...")

    if not meta_path.exists() or make_meta:
        print("Making metadata table...")
        allpars = []
        for filename in allfilenames:
            filename = pathlib.Path(filename)
            if "init" in filename.parts[-1]:
                continue

            pars = at.QTable.read(filename, path="/parameters")
            allpars.append(pars)

        allpars = at.vstack(allpars)
        allpars.write(meta_path, overwrite=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--nproc", type=int, default=None)
    grp.add_argument("--mpi", action="store_true", default=False)

    parser.add_argument("--dist", type=float, default=None, required=True)
    parser.add_argument("-o", "--overwrite", action="store_true", default=False)
    parser.add_argument("--overwriteplots", action="store_true", default=False)
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
        main(
            pool,
            dist=args.dist,
            overwrite=args.overwrite,
            overwrite_plots=args.overwriteplots,
        )
