import pathlib
import time
from itertools import product

import astropy.units as u
import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
import h5py
import numpy as np
from gala.dynamics import mockstream as ms
from gala.units import galactic
from schwimmbad import MultiPool
from schwimmbad.utils import batch_tasks


class StreamSubhaloSimulation:
    @u.quantity_input(
        M_stream=u.Msun, t_pre_impact=u.Myr, t_post_impact=u.Myr, dt=u.Myr
    )
    def __init__(
        self,
        mw_potential,
        final_prog_w,
        M_stream,
        t_pre_impact,
        t_post_impact,
        dt=1 * u.Myr,
        release_every=1,
        n_particles=1,
        seed=None,
    ) -> None:
        self.mw_potential = mw_potential
        self.final_prog_w = final_prog_w
        if not isinstance(self.mw_potential, gp.PotentialBase):
            raise TypeError("Input potential must be a gala potential object")
        if not isinstance(self.final_prog_w, gd.PhaseSpacePosition):
            raise TypeError(
                "Input final_prog_w must be a gala PhaseSpacePosition object"
            )

        # Mockstream and simulation parameters validated by the quantity_input decorator
        self.M_stream = M_stream
        self.t_pre_impact = t_pre_impact
        self.t_post_impact = t_post_impact
        self.dt = dt
        self._mockstream_kw = dict(release_every=release_every, n_particles=n_particles)
        self.rng = np.random.default_rng(seed=seed)

        self.H = gp.Hamiltonian(mw_potential)
        self._df = ms.FardalStreamDF(random_state=self.rng)
        self._mockstream_gen = ms.MockStreamGenerator(df=self._df, hamiltonian=self.H)
        self._prog_w0 = self.H.integrate_orbit(
            self.final_prog_w,
            dt=-dt,
            t1=t_pre_impact + t_post_impact,
            t2=0,
            Integrator=gi.Ruth4Integrator,
            store_all=False,
        )[0]

    def run_init_stream(self, release_every=1, n_particles=1):
        """
        Generate the initial, unperturbed stream model. This is used to pick a random
        star particle to then define the impact site of the perturber.
        """
        return self._mockstream_gen.run(
            self._prog_w0,
            self.M_stream,
            dt=self.dt,
            t1=0,
            t2=self.t_pre_impact,
            release_every=release_every,
            n_particles=n_particles,
        )

    def get_impact_site(
        self, init_stream, init_prog, prog_dist=8 * u.kpc, leading=True
    ):
        if leading:
            lead_trail_sign = 1.0
        else:
            lead_trail_sign = -1.0

        dx = init_stream.xyz - init_prog.xyz
        prog_dists = np.linalg.norm(dx, axis=0)

        # pick zeroth star that matches - MAGIC NUMBER
        rando_star_idx = np.where(
            np.isclose(prog_dists, prog_dist, atol=0.1 * u.kpc)
            & (np.sign(dx[1]) == np.sign(lead_trail_sign * init_prog.v_y))
        )[0][0]

        nearby_mask = (
            np.linalg.norm(
                init_stream.xyz - init_stream[rando_star_idx].xyz[:, None], axis=0
            )
            < 1 * u.kpc
        )  # MAGIC NUMBER

        return gd.PhaseSpacePosition(
            np.mean(init_stream.xyz[:, nearby_mask], axis=1),
            np.mean(init_stream.v_xyz[:, nearby_mask], axis=1),
        )

    @u.quantity_input(t_buffer_impact=u.Myr, impact_dt=u.Myr)
    def run_perturbed_stream(
        self,
        impact_site_w,
        subhalo_impact_dw,
        subhalo_potential,
        t_buffer_impact=None,
        impact_dt=None,
    ):
        subhalo_v = np.linalg.norm(subhalo_impact_dw.v_xyz)
        if t_buffer_impact is None:
            t_buffer_impact = np.round((1 * u.kpc / subhalo_v).to(u.Myr), decimals=1)
        if impact_dt is None:
            impact_dt = (1.0 * u.pc / subhalo_v).to(u.Myr)

        w_subhalo_impact = gd.PhaseSpacePosition(
            impact_site_w.xyz + subhalo_impact_dw.xyz,
            impact_site_w.v_xyz + subhalo_impact_dw.v_xyz,
        )
        w_subhalo_buffer = self.H.integrate_orbit(
            w_subhalo_impact,
            dt=-self.dt / 10,
            t1=t_buffer_impact,
            t2=0,
            Integrator=gi.Ruth4Integrator,
            store_all=False,
        )[0]

        stream_buffer_pre, prog_w_buffer_pre = self._mockstream_gen.run(
            self._prog_w0,
            self.M_stream,
            dt=self.dt,
            t1=0,
            t2=self.t_pre_impact - t_buffer_impact,
            **self._mockstream_kw,
        )

        tmp = gd.PhaseSpacePosition(
            stream_buffer_pre.pos, stream_buffer_pre.vel, frame=stream_buffer_pre.frame
        )
        nbody_w0 = gd.combine((w_subhalo_buffer, tmp))

        null_potential = gp.NullPotential(units=galactic)
        nbody = gd.DirectNBody(
            w0=nbody_w0,
            particle_potentials=[subhalo_potential] + [null_potential] * tmp.shape[0],
            external_potential=self.H.potential,
            frame=self.H.frame,
            save_all=False,
        )
        stream_impact = nbody.integrate_orbit(
            dt=impact_dt,
            t1=self.t_pre_impact - t_buffer_impact,
            t2=self.t_pre_impact + t_buffer_impact,
        )
        stream_after_impact = self.H.integrate_orbit(
            stream_impact,
            dt=self.dt,
            t1=self.t_pre_impact + t_buffer_impact,
            t2=self.t_pre_impact + self.t_post_impact,
            Integrator=gi.Ruth4Integrator,
            store_all=False,
        )[0]

        unpert_stream_post, _ = self._mockstream_gen.run(
            prog_w_buffer_pre[0],
            self.M_stream,
            dt=self.dt,
            t1=self.t_pre_impact - t_buffer_impact,
            t2=self.t_pre_impact + self.t_post_impact,
            **self._mockstream_kw,
        )

        w_impact_end = self.H.integrate_orbit(
            impact_site_w,
            dt=self.dt,
            t1=self.t_pre_impact,
            t2=self.t_pre_impact + self.t_post_impact,
        )

        return stream_after_impact, unpert_stream_post, w_impact_end


# Run a mockstream simulation with Gala and save the output to an hdf5 file
def main(pool, overwrite=False):
    rng = np.random.default_rng(123)

    mw = gp.load(
        "/mnt/home/apricewhelan/projects/gaia-actions/potentials/"
        "MilkyWayPotential2022.yml"
    )

    wf = gd.PhaseSpacePosition(pos=[15, 0.0, 0.0] * u.kpc, vel=[0, 275, 0] * u.km / u.s)

    sim_kw = dict(
        mw_potential=mw,
        final_prog_w=wf,
        M_stream=5e4 * u.Msun,
        t_pre_impact=3 * u.Gyr,
        n_particles=4,
        seed=42,
    )
    print("Setting up simulation instance...")
    sim = StreamSubhaloSimulation(t_post_impact=0 * u.Myr, **sim_kw)

    print("Running initial stream simulation...")
    init_stream, init_prog = sim.run_init_stream()
    print("Finding a good impact site...")
    impact_site = sim.get_impact_site(init_stream, init_prog)

    Ms = [1e6, 5e6, 1e7, 5e7] * u.Msun
    vs = [25, 50, 100, 200] * u.pc / u.Myr
    b_facs = [0, 0.5, 1.0, 2.0]
    ts = [100, 200, 400, 800] * u.Myr

    # Some custom geometries for the subhalo at interaction, some random:
    rand_dxdvs = [
        ([1.0, 0, 0], [0, 0, 1.0]),
        ([0, 0, 1.0], [1.0, 0, 0]),
        (rng.normal(size=3), rng.normal(size=3)),
        (rng.normal(size=3), rng.normal(size=3)),
    ]
    par_tasks = list(product(Ms, vs, b_facs, ts, rand_dxdvs))

    # HACK: ESTIMATE
    time_per_sim = 5 * u.min
    print(
        f"Running a total of {len(par_tasks)} simulations - this will take about "
        f"{(len(par_tasks) * time_per_sim / pool.size).to(u.hour):.1f}"
    )

    # Make a cache directory to save the simulation output:
    cache_path = (pathlib.Path(__file__).parent / "../cache").resolve().absolute()
    cache_path.mkdir(exist_ok=True)

    tasks = batch_tasks(
        arr=par_tasks,
        n_batches=max(pool.size - 1, 1),
        args=(sim_kw, impact_site, cache_path, overwrite),
    )

    for res in pool.starmap(worker, tasks):
        pass


def worker(idxs, batch, sim_kw, impact_site, cache_path, overwrite):
    for i, (M_subhalo, impact_v, impact_b_fac, t_post_impact, dxdv) in zip(
        range(*idxs), batch
    ):
        filename = cache_path / f"stream-{i:04d}.hdf5"
        if filename.exists():
            if not overwrite:
                print(f"[{i}]: {filename} already exists, skipping...")
                continue
            else:
                print(f"[{i}]: {filename} exists - overwriting...")

        c_subhalo = 1.005 * u.kpc * (M_subhalo / (1e8 * u.Msun)) ** 0.5 / 2.0  # MAGIC
        impact_b = impact_b_fac * c_subhalo

        sim = StreamSubhaloSimulation(t_post_impact=t_post_impact, **sim_kw)

        # Buffer time is 32 times the crossing time:
        MAGIC = 32
        print(f"[{i}]: starting simulation...")
        time0 = time.time()
        stream, _, w_impact_end = sim.run_perturbed_stream(
            impact_site_w=impact_site,
            subhalo_impact_dw=gd.PhaseSpacePosition(
                dxdv[0] / np.linalg.norm(dxdv[0]) * impact_b,
                dxdv[1] / np.linalg.norm(dxdv[1]) * impact_v,
            ),
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
        with h5py.File(filename, mode="w") as f:
            g = f.create_group("stream")
            stream.to_hdf5(g)

            g = f.create_group("impact")
            w_impact_end.to_hdf5(g)

        break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("-o", "--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    with MultiPool(processes=args.nproc) as pool:
        main(pool, overwrite=args.overwrite)
