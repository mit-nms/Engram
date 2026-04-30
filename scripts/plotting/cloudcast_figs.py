import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle


def plot_relay_on_world_image(
    img_path,
    source, worker, destinations,
    figsize=(12, 6),
    node_radius_px=26,      # BIGGER nodes
    direct_color="#d62728",
    relay_color="#2ca02c",
    direct_lw=2.0,
    relay_lw=2.0,
    relay_ls="--",
    alpha=0.95,
    arrow_head_scale=20,
    extent_lonlat=(-180, 180, -90, 90),
    savepath=None,
    show=True,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from matplotlib.patches import Circle

    # -------------------- load image --------------------
    img = Image.open(img_path).convert("RGBA")
    W, H = img.size

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, extent=(0, W, H, 0))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    lon_min, lon_max, lat_min, lat_max = extent_lonlat

    # -------------------- coordinate transform --------------------
    def lonlat_to_xy(lon, lat):
        x = (lon - lon_min) / (lon_max - lon_min) * W
        y = (lat_max - lat) / (lat_max - lat_min) * H
        return float(x), float(y)

    # -------------------- draw node --------------------
    def draw_node(lon, lat, label, color, textcolor="black"):
        x, y = lonlat_to_xy(lon, lat)
        circ = Circle(
            (x, y),
            radius=node_radius_px,
            facecolor=color,
            edgecolor="none",    # ❌ NO BORDER
            linewidth=0,
            zorder=6
        )
        ax.add_patch(circ)
        ax.text(
            x, y, label,
            ha="center", va="center",
            fontsize=13, fontweight="bold",
            color=textcolor, zorder=7
        )

    # -------------------- draw straight arrow --------------------
    def draw_arrow(lon1, lat1, lon2, lat2, color, lw, ls="-"):
        x1, y1 = lonlat_to_xy(lon1, lat1)
        x2, y2 = lonlat_to_xy(lon2, lat2)

        ax.annotate(
            "",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                linewidth=lw,
                linestyle=ls,
                mutation_scale=arrow_head_scale,
                alpha=alpha,
            ),
            zorder=4,
        )

    # -------------------- unpack points --------------------
    s_lon, s_lat, s_lab = source["lon"], source["lat"], source.get("label", "S")
    w_lon, w_lat, w_lab = worker["lon"], worker["lat"], worker.get("label", "W")

    # -------------------- arrows --------------------
    for d in destinations:
        draw_arrow(s_lon, s_lat, d["lon"], d["lat"], color=direct_color, lw=direct_lw, ls="-")

    draw_arrow(s_lon, s_lat, w_lon, w_lat, color=relay_color, lw=relay_lw, ls=relay_ls)
    for d in destinations:
        draw_arrow(w_lon, w_lat, d["lon"], d["lat"], color=relay_color, lw=relay_lw, ls=relay_ls)

    # -------------------- nodes --------------------
    draw_node(s_lon, s_lat, s_lab, color="#3b1b7a", textcolor="white")
    draw_node(w_lon, w_lat, w_lab, color="#f2c94c", textcolor="black")
    for d in destinations:
        draw_node(d["lon"], d["lat"], d.get("label", "D"), color="#bfe3f6", textcolor="black")

    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax



# ---------------- Example ----------------
if __name__ == "__main__":
    img_path = "simple-world-map-in-flat-style-isolated-on-white-background-illustration-free-vector.jpg"

    # Reuse the same set of points as before, but swap roles
    # so the layout is different from the original paper.
    source = {"lon": 10.0, "lat": 50.0, "label": "S"}    # Europe-ish
    worker = {"lon": -122.4, "lat": 37.8, "label": "W"}  # west coast US

    destinations = [
        {"lon": -105.0, "lat": 55.0,  "label": "D"},     # Canada-ish
        {"lon": -60.0,  "lat": -15.0, "label": "D"},
        {"lon":  30.0,  "lat":  10.0, "label": "D"},
        {"lon":  78.0,  "lat":  22.0, "label": "D"},
        {"lon": 105.0,  "lat":  35.0, "label": "D"},
        {"lon": 135.0,  "lat": -25.0, "label": "D"},
    ]

    plot_relay_on_world_image(img_path, source, worker, destinations, savepath="relay_world.pdf")
