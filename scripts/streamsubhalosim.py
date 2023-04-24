import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from gala.dynamics import mockstream as ms
from gala.units import galactic


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
        progenitor_potential=None,
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
        self._mockstream_gen = ms.MockStreamGenerator(
            df=self._df, hamiltonian=self.H, progenitor_potential=progenitor_potential
        )
        self._prog_w0 = self.H.integrate_orbit(
            self.final_prog_w,
            dt=-dt,
            t1=t_pre_impact + t_post_impact,
            t2=0,
            Integrator=gi.Ruth4Integrator,
            store_all=False,
        )[0]

    def run_init_stream(self):
        """
        Generate the initial, unperturbed stream model. This is used to pick a random
        star particle to then define the impact site of the perturber.
        """
        return self._mockstream_gen.run(
            self._prog_w0,
            self.M_stream,
            dt=self.dt,
            t1=0,
            t2=self.t_pre_impact + self.t_post_impact,
            **self._mockstream_kw,
        )

    def get_impact_site(
        self, init_stream, init_prog, prog_dist=8 * u.kpc, leading=True
    ):
        """
        Given a simulation of the unperturbed stream, pick a random star particle at
        some distance away from the progenitor in either the leading or trailing tail.
        This is used to define the impact site of the perturber. The input stream and
        progenitor position are assumed to be at the final timestep of the eventual
        impact simulation (i.e. the impact site is defined at the end timestep of the
        unperturbed stream simulation).
        """
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

        final_time = self.t_pre_impact + self.t_post_impact

        # Backwards-integrate the impact site location from the end of the simulation
        # to the time of impact
        impact_site_at_impact = self.H.integrate_orbit(
            impact_site_w,
            dt=-self.dt / 10.0,
            t1=final_time,
            t2=self.t_pre_impact,
            Integrator=gi.Ruth4Integrator,
        )[-1]

        # At the time of impact, define the subhalo relative phase-space coordinates,
        # using parameters relative to the impact site
        w_subhalo_impact = gd.PhaseSpacePosition(
            impact_site_at_impact.xyz + subhalo_impact_dw.xyz,
            impact_site_at_impact.v_xyz + subhalo_impact_dw.v_xyz,
        )

        # Integrate the subhalo orbit from time of impact back to the buffer time
        w_subhalo_buffer = self.H.integrate_orbit(
            w_subhalo_impact,
            dt=-self.dt / 10,
            t1=self.t_pre_impact,
            t2=self.t_pre_impact - t_buffer_impact,
            Integrator=gi.Ruth4Integrator,
            store_all=False,
        )[0]

        # Generate the mock stream up to the buffer time relative to the impact
        stream_buffer_pre, prog_w_buffer_pre = self._mockstream_gen.run(
            self._prog_w0,
            self.M_stream,
            dt=self.dt,
            t1=0,
            t2=self.t_pre_impact - t_buffer_impact,
            **self._mockstream_kw,
        )

        # Extract mock stream particle initial conditions to prepare to forward
        # integrate through the impact using N-body forces
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

        if (self.t_pre_impact + t_buffer_impact) > final_time:
            buffer_t2 = final_time
        else:
            buffer_t2 = self.t_pre_impact + t_buffer_impact

        stream_impact = nbody.integrate_orbit(
            dt=impact_dt,
            t1=self.t_pre_impact - t_buffer_impact,
            t2=buffer_t2,
        )
        if buffer_t2 != final_time:
            stream_after_impact = self.H.integrate_orbit(
                stream_impact,
                dt=self.dt,
                t1=buffer_t2,
                t2=final_time,
                Integrator=gi.Ruth4Integrator,
                store_all=False,
            )[0]
        else:
            stream_after_impact = stream_impact

        unpert_stream_post, _ = self._mockstream_gen.run(
            prog_w_buffer_pre[0],
            self.M_stream,
            dt=self.dt,
            t1=self.t_pre_impact - t_buffer_impact,
            t2=self.t_pre_impact + self.t_post_impact,
            **self._mockstream_kw,
        )

        return stream_after_impact, unpert_stream_post


def get_in_stream_frame(stream, w_impact_end):
    stream_galcen = coord.Galactocentric(stream.data)
    stream_gal = stream_galcen.transform_to(coord.Galactic())

    perturb_end_galcen = coord.Galactocentric(w_impact_end.data)
    perturb_end_gal = perturb_end_galcen.transform_to(coord.Galactic())

    stream_frame = coord.SkyOffsetFrame(origin=perturb_end_gal)
    stream_sfr = stream_gal.transform_to(stream_frame)
    return stream_sfr
