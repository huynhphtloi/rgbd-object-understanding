"""Render the spatial-relation graph to an image (matplotlib, headless)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..reasoning.object_instance import ObjectInstance
from ..reasoning.relations import Relation
from ..reasoning.graph_builder import to_networkx


def render_relation_graph(
    instances: list[ObjectInstance],
    relations: list[Relation],
    out_path: str | None = None,
):
    """Draw the relation DiGraph; save to out_path if given, return the figure."""
    import networkx as nx

    g = to_networkx(instances, relations)
    fig, ax = plt.subplots(figsize=(7, 6))

    if g.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "no objects", ha="center", va="center")
        ax.axis("off")
        if out_path:
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
        return fig

    pos = nx.spring_layout(g, seed=42, k=1.5)
    labels = {n: g.nodes[n].get("label", str(n)) for n in g.nodes}
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color="#cfe3ff",
                           node_size=2600, edgecolors="#2b6cb0")
    nx.draw_networkx_labels(g, pos, labels, ax=ax, font_size=8)

    # Draw each relation edge with its label; offset parallel edges.
    for i, (u, v, key, data) in enumerate(g.edges(keys=True, data=True)):
        rad = 0.15 * (1 + (i % 3))
        nx.draw_networkx_edges(
            g, pos, edgelist=[(u, v)], ax=ax, arrows=True, arrowsize=15,
            edge_color="#555", connectionstyle=f"arc3,rad={rad}",
        )
        xm = (pos[u][0] + pos[v][0]) / 2
        ym = (pos[u][1] + pos[v][1]) / 2 + 0.08 * (1 + i % 3)
        ax.text(xm, ym, data["relation"], fontsize=7, color="#b83280",
                ha="center", va="center")

    ax.set_title("Spatial-relation graph")
    ax.axis("off")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig
