import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_sky_projections(
    stream_sfr,
    components=None,
    tracks=None,
    annotate_impact=True,
    xlim=(-45, 45),
    ylims=None,
    hist2d=True,
    hist2d_kwargs=None,
    hist2d_n_xbins=501,
    hist2d_n_ybins=151,
    scatter=False,
    scatter_kwargs=None,
    axes=None,
    xlabel=True,
    ylabel=True,
):
    lon = stream_sfr.phi1.wrap_at(180 * u.deg).degree
    _mask = (lon > xlim[0]) & (lon < xlim[1])

    if components is None:
        components = [
            "phi2",
            "distance",
            "pm_phi1_cosphi2",
            "pm_phi2",
            "radial_velocity",
        ]
    assert len(components) >= 1

    # Make data either by getting component data, or computing relative to tracks:
    ys = {}
    for comp in components:
        ys[comp] = getattr(stream_sfr, comp).value[_mask]
        if tracks is not None:
            ys[comp] = ys[comp] - tracks[comp](lon[_mask])

    if ylims is None:
        ylims = {}

    for comp in components:
        tmp = np.percentile(ys[comp], [1, 99])
        w = tmp[1] - tmp[0]
        ylims.setdefault(comp, (tmp[0] - 0.5 * w, tmp[1] + 0.5 * w))

    if hist2d_kwargs is None:
        hist2d_kwargs = dict()
    hist2d_kwargs.setdefault("norm", mpl.colors.LogNorm(vmin=0.1))
    hist2d_kwargs.setdefault("cmap", "Greys")

    if scatter_kwargs is None:
        scatter_kwargs = dict()
    scatter_kwargs.setdefault("marker", "o")
    scatter_kwargs.setdefault("s", 6)
    scatter_kwargs.setdefault("alpha", 0.5)
    scatter_kwargs.setdefault("linewidth", scatter_kwargs.pop("lw", 0.0))

    if axes is None:
        fig, axes = plt.subplots(
            len(components),
            1,
            figsize=(16, 4 * len(components)),
            sharex=True,
            constrained_layout=True,
        )

        if len(components) == 1:
            axes = [axes]

    else:
        fig = axes[0].figure

    for ax, comp in zip(axes, components):
        ylim = ylims[comp]

        if hist2d:
            hist2d_kwargs["bins"] = (
                np.linspace(*xlim, hist2d_n_xbins),
                np.linspace(*ylim, hist2d_n_ybins),
            )
            ax.hist2d(lon[_mask], ys[comp], **hist2d_kwargs)

        if scatter:
            ax.scatter(lon[_mask], ys[comp], **scatter_kwargs)

        ax.set_ylim(ylim)
        if ylabel:
            ax.set_ylabel(comp)

        if annotate_impact:
            yloc = ylim[0] + 0.3 * (ylim[1] - ylim[0])
            yloctext = ylim[0] + 0.1 * (ylim[1] - ylim[0])
            ann_style = dict(
                ha="center",
                va="center",
                arrowprops=dict(
                    arrowstyle="->",
                    lw=1.5,
                    color="tab:red",
                    shrinkA=0,
                    shrinkB=0,
                    alpha=0.5,
                ),
                fontsize=14,
                zorder=100,
                color="tab:red",
                alpha=0.9,
            )
            ax.annotate("impact", xy=(0, yloc), xytext=(0, yloctext), **ann_style)
            ax.axvline(0, color="tab:red", lw=1.0, ls="--", alpha=0.5, zorder=-10)
            annotate_impact = False

    if xlabel:
        axes[-1].set(xlim=xlim, xlabel=r"stream longitude $\phi_1$ [deg]")

    return fig, axes
