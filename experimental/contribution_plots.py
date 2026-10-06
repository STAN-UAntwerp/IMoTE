"""Experimental contribution plots, not part of the IMoTE package (not shipped to PyPI)."""
import os
from datetime import datetime
from subprocess import run

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from experimental import colors
from experimental.beeswarm import beeswarm


def scatter_contributions(df_X, df_contributions, feature, color_feature = None):
    if color_feature is None:
        corr = df_contributions.corr()
        color_feature = corr[feature].nlargest(2).index[1]

    x_color = df_X[color_feature].to_numpy()
    median = np.nanmedian(x_color)
    mad = 1.4826 * np.nanmedian(np.abs(x_color - median))
    vmin = median - 3 * mad
    vmax = median + 3 * mad


    fig, ax = plt.subplots(figsize=(8, 6))

    # scatter
    sc = ax.scatter(
        df_X[feature],
        df_contributions[feature],
        c=df_X[color_feature].clip(vmin, vmax),
        cmap=colors.red_blue,
        s=10,
        marker='.',
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(f"{feature} contribution")
    ax.set_xlabel(feature)
    ax.set_ylabel("Contribution")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(color_feature)

    # histogram inset at bottom
    ax_hist = inset_axes(
        ax,
        width="100%",
        height="18%",
        loc="lower left",
        bbox_to_anchor=(0, 0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    ax_hist.hist(
        df_X[feature],
        bins=50,
        color="gray",
        alpha=0.3
    )

    ax_hist.set_yticks([])
    ax_hist.set_xticks([])
    ax_hist.patch.set_alpha(0)

    for spine in ax_hist.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.show()

def beeswarm_wrap(df_contributions, X, y_hat, directory):
    id_results = datetime.now().strftime('%d-%m-%y_%H-%M-%S')
    file_directory = os.path.join(directory, f"beeswarm_{id_results}.pdf")
    feature_names = df_contributions.columns.tolist()
    beeswarm(np.array(df_contributions),
             X,
             y_hat,
             n_max=10,
             fig_size=(10, 6),
             truncate_total_pred=True,
             variable_tick_width=True,
             file_directory=file_directory,
             highlight_x=None,
             staircase=False,
             feature_names=feature_names,
             )
    run(["open", "-a", "Preview", file_directory])