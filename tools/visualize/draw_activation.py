import glob
import os
from typing import List, Optional
from io import BytesIO

import imageio
import matplotlib
matplotlib.use("agg")
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def draw_skeleton(
    result: np.ndarray,
    points: np.ndarray,
    label,
    connect_joint: Optional[List[int]] = None,
    render_gif: bool = True,
    min_conf: float = 0.0,
    out_dir: str = "data/output",
    dpi: int = 160,
):
    os.makedirs(f"{out_dir}/png", exist_ok=True)
    os.makedirs(f"{out_dir}/gif", exist_ok=True)

    result = np.array(result)
    if result.ndim == 2:
        result = result[np.newaxis, ...]

    K, T, V = result.shape
    result = np.maximum(result, 0.0)
    if np.max(result) > 0:
        result = result / np.max(result)

    if connect_joint is None:
        connect_joint = [max(0, v - 1) for v in range(V)]

    mean_pos = np.mean(np.mean(points[:2], -1), -1)
    all_x = points[0] - mean_pos[0]
    all_y = mean_pos[1] - points[1]

    xmin, xmax = np.min(all_x), np.max(all_x)
    ymin, ymax = np.min(all_y), np.max(all_y)
    max_range = max(xmax - xmin, ymax - ymin)
    pad = max_range * 0.25
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2

    scalar_map = cmx.ScalarMappable(cmap=plt.get_cmap("plasma"), norm=colors.Normalize(vmin=0, vmax=1))
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    fig.colorbar(scalar_map, ax=ax, fraction=0.045, pad=0.02)

    for t in range(T):
        ax.clear()
        x = points[0, t, :] - mean_pos[0]
        y = mean_pos[1] - points[1, t, :]
        conf = points[2, t, :]

        ax.set_xlim(cx - max_range / 2 - pad, cx + max_range / 2 + pad)
        ax.set_ylim(cy - max_range / 2 - pad, cy + max_range / 2 + pad)
        ax.set_aspect("equal")
        ax.axis("off")

        node_colors, activation = [], []
        for v in range(V):
            k = connect_joint[v]
            r = float(np.max(result[:, t, v]))
            activation.append(r)

            if conf[k] < min_conf or conf[v] < min_conf:
                node_colors.append([0, 0, 0, 0])
                continue

            node_colors.append(scalar_map.to_rgba(r))
            ax.plot([x[v], x[k]], [y[v], y[k]], "-", c=[0.15, 0.15, 0.15], alpha=0.8, linewidth=5, zorder=1)

        node_colors = np.array(node_colors, dtype=np.float64)
        size = np.array(activation, dtype=np.float64) * 900 + 80
        ax.scatter(x, y, marker="o", c=node_colors, s=size, zorder=3)

        plt.tight_layout()


        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, dpi=dpi)
        buf.seek(0)
        final_path = f"{out_dir}/png/{int(label[0])}-{int(label[1])}-{int(label[2])}-{t:03}.png"
        tmp_path = final_path + ".tmp"
        with open(tmp_path, 'wb') as f:
            f.write(buf.getvalue())
        buf.close()
        try:
            os.replace(tmp_path, final_path)
        except Exception:
            os.rename(tmp_path, final_path)

    plt.close()

    if render_gif:
        filenames = sorted(glob.glob(f"{out_dir}/png/{int(label[0])}-{int(label[1])}-{int(label[2])}-*.png"))
    
        filenames = [f for f in filenames if os.path.getsize(f) > 0]
        if not filenames:
            return

        images = []
        for f in filenames:
            try:
                img = imageio.imread(f)
            except Exception:
                continue
            images.append(img)

        if not images:
            return

        target_shape = images[0].shape
        resized = []
        for img in images:
            if img.shape != target_shape:
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((target_shape[1], target_shape[0]), Image.LANCZOS)
                img = np.array(pil_img)
            resized.append(img)

        imageio.mimwrite(f"{out_dir}/gif/{int(label[0])}-{int(label[1])}-{int(label[2])}.gif", resized, duration=0.15)
