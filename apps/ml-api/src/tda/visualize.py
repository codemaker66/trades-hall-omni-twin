"""
Persistence diagram and barcode visualization (TDA-1 supplementary).

Generates matplotlib figures for persistence diagrams and barcodes.
Returns base64-encoded PNG images for API responses.
"""

import numpy as np
import io
import base64


def persistence_diagram_to_png(
    diagrams: dict,
    title: str = "Persistence Diagram",
    figsize: tuple = (8, 6),
) -> str:
    """
    Generate a persistence diagram as a base64-encoded PNG.

    Points above the diagonal have positive lifespan.
    Further from diagonal = more persistent (more significant) feature.

    Args:
        diagrams: Dict with 'H0', 'H1', 'H2' keys, each list of [birth, death]
        title: Plot title
        figsize: Figure size in inches

    Returns:
        Base64-encoded PNG string
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    colors = {"H0": "#1f77b4", "H1": "#ff7f0e", "H2": "#2ca02c"}
    labels = {"H0": "H₀ (clusters)", "H1": "H₁ (loops)", "H2": "H₂ (voids)"}

    all_vals: list[float] = []
    for dim in ["H0", "H1", "H2"]:
        dgm = np.array(diagrams.get(dim, []))
        if len(dgm) == 0:
            continue

        finite_mask = np.isfinite(dgm[:, 1])
        finite = dgm[finite_mask]
        infinite = dgm[~finite_mask]

        if len(finite) > 0:
            ax.scatter(
                finite[:, 0], finite[:, 1],
                c=colors[dim], label=labels[dim],
                s=30, alpha=0.7, edgecolors="black", linewidths=0.5,
            )
            all_vals.extend(finite[:, 0].tolist())
            all_vals.extend(finite[:, 1].tolist())

        if len(infinite) > 0:
            # Plot infinite features as triangles at the top
            max_val = max(all_vals) if all_vals else 1.0
            ax.scatter(
                infinite[:, 0], [max_val * 1.1] * len(infinite),
                c=colors[dim], marker="^", s=50, alpha=0.7,
                edgecolors="black", linewidths=0.5,
            )

    # Diagonal line
    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        margin = (hi - lo) * 0.1
        ax.plot(
            [lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", alpha=0.3, linewidth=1,
        )

    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def barcode_to_png(
    diagrams: dict,
    title: str = "Persistence Barcode",
    figsize: tuple = (10, 6),
) -> str:
    """
    Generate a persistence barcode as a base64-encoded PNG.

    Each bar represents a topological feature; length = lifespan = significance.

    Args:
        diagrams: Dict with 'H0', 'H1', 'H2' keys
        title: Plot title
        figsize: Figure size

    Returns:
        Base64-encoded PNG string
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    colors = {"H0": "#1f77b4", "H1": "#ff7f0e", "H2": "#2ca02c"}
    labels = {"H0": "H₀", "H1": "H₁", "H2": "H₂"}

    y_pos = 0
    y_ticks: list[float] = []
    y_labels: list[str] = []

    for dim in ["H0", "H1", "H2"]:
        dgm = np.array(diagrams.get(dim, []))
        if len(dgm) == 0:
            continue

        # Sort by birth time
        order = np.argsort(dgm[:, 0])
        dgm = dgm[order]

        first_in_group = True
        for birth, death in dgm:
            if not np.isfinite(death):
                death = max(dgm[np.isfinite(dgm[:, 1]), 1].max() * 1.2, birth + 0.1) \
                    if np.any(np.isfinite(dgm[:, 1])) else birth + 0.5

            ax.barh(
                y_pos, death - birth, left=birth,
                height=0.6, color=colors[dim], alpha=0.7,
                edgecolor="black", linewidth=0.5,
                label=labels[dim] if first_in_group else None,
            )
            first_in_group = False
            y_pos += 1

    ax.set_xlabel("Filtration Value")
    ax.set_ylabel("Features")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")
