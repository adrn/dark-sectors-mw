import astropy.units as u
import numpy as np
import gala.dynamics as gd
from gala.dynamics import mockstream as ms
import gala.integrate as gi
import gala.potential as gp
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
        release_every=1,
        n_particles=1,
        seed=None,
    ) -> None:
        self.mw_potential = mw_potential
        self.final_prog_w = final_prog_w
        if not isinstance(self.mw_potential, gp.PotentialBase):
            raise TypeError("Input potential must be a gala potential object")
        if not isinstance(self.final_prog_w, gp.PhaseSpacePosition):
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
        self._mockstream_gen = ms.MockStreamGenerator(df=self._df, hamiltonian=H)
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
        self._init_stream, _ = self._mockstream_gen.run(
            self._prog_w0,
            self.M_stream,
            dt=self.dt,
            t1=0,
            t2=self.t_pre_impact,
            release_every=release_every,
            n_particles=n_particles,
        )
        return self._init_stream

    def run_perturbed_stream(
        self, impact_site_w, subhalo_impact_dw, subhalo_potential, t_buffer_impact=None
    ):

        subhalo_v = np.linalg.norm(subhalo_impact_dw.v_xyz)
        t_buffer_impact = np.round((1 * u.kpc / subhalo_v).to(u.Myr), decimals=1)
        impact_dt = np.round((0.1 * u.pc / subhalo_v).to(u.Myr), decimals=2)

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
            **self._mockstream_kw
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
            stream_impact[-1],
            dt=self.dt,
            t1=self.t_pre_impact + t_buffer_impact,
            t2=self.t_pre_impact + self.t_post_impact,
            Integrator=gi.Ruth4Integrator,
            store_all=False,
        )

        unpert_stream_post, _ = self._mockstream_gen.run(
            prog_w_buffer_pre[0],
            self.M_stream,
            dt=self.dt,
            t1=self.t_pre_impact - t_buffer_impact,
            t2=self.t_pre_impact + self.t_post_impact,
            **self._mockstream_kw
        )

        w_impact_end = self.H.integrate_orbit(
            impact_site_w,
            dt=self.dt,
            t1=self.t_pre_impact,
            t2=self.t_pre_impact + self.t_post_impact,
        )

        return stream_after_impact, unpert_stream_post, w_impact_end


# Run a mockstream simulation with Gala and save the output to an hdf5 file
def main():
    pass


if __name__ == "__main__":
    main()
