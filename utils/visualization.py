import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable


def visualize_segmentation_heatmap(
    time_series_sample,
    importance_mask,
    segmentation_mask,
    title="Time Series Attributions",
):
    # Flatten to 1D arrays
    ts = time_series_sample.squeeze()
    mask = importance_mask.squeeze()
    segmentation_mask = segmentation_mask.squeeze()
    n_steps = len(ts)

    # Normalize colors from red (negative) to green (positive)
    vmax = np.max(np.abs(mask))
    vmin = -vmax
    try:
        norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
    except:  # noqa: E722
        print("ERROR. COULD NOT DISPLAY SEGMENT EXPLANATION")

        # Save text showing all zeros
        data_path = Path(
            f"../outputs/chronos/data/Segment_LIME/{title.replace(' ', '')}.txt"
        )
        data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(data_path, mask, fmt="%.6f")
        return
    cmap = plt.cm.RdYlGn  # Red to Green

    fig, ax = plt.subplots(figsize=(12, 4))

    # Heatmap background
    ax.imshow(
        mask[np.newaxis, :],
        cmap=cmap,
        aspect="auto",
        extent=[0, n_steps, np.min(ts), np.max(ts)],
        norm=norm,
        alpha=1.0,
    )

    # Plot time series
    ax.plot(ts, color="blue", linewidth=1.5)

    # Draw vertical lines at segment boundaries
    segment_boundaries = np.where(np.diff(segmentation_mask) != 0)[0] + 1
    for boundary in segment_boundaries:
        ax.axvline(x=boundary, color="black", linestyle="--", linewidth=1)

    # Labels and title
    ax.set_xlim([0, n_steps])
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

    # Add colorbar to indicate attribution scale
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("Segment Importance")

    plt.tight_layout()

    return plt
