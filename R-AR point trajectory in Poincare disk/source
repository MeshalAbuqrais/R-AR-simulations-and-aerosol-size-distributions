import numpy as np
from geomstats.geometry.hyperboloid import Hyperboloid
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path
from matplotlib.colors import Normalize

H2 = Hyperboloid(dim=2)
metric = H2.metric

def lorentz_inner(x, y) -> float:

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(-x[0] * y[0] + x[1] * y[1] + x[2] * y[2])

def project_upper(x):

    y = H2.projection(np.asarray(x, dtype=float))
    if y[0] < 0:
        y = -y
    return y

def to_poincare(x):

    x = np.asarray(x, dtype=float)
    p = x[1:] / (1.0 + x[0])
    r = np.linalg.norm(p)
    if r >= 1.0:
        p *= (1.0 - 1e-12) / max(r, 1e-18)
    return p

def tangent_frame_at(mu_vec):
    a1 = np.array([0.0, 1.0, 0.0])
    a2 = np.array([0.0, 0.0, 1.0])


    w1 = a1 + lorentz_inner(a1, mu_vec) * mu_vec
    n1 = lorentz_inner(w1, w1)
    if n1 <= 1e-15:
        a1 = np.array([0.0, 0.0, 1.0])
        w1 = a1 + lorentz_inner(a1, mu_vec) * mu_vec
        n1 = lorentz_inner(w1, w1)
    w1 = w1 / np.sqrt(n1)


    w2 = a2 + lorentz_inner(a2, mu_vec) * mu_vec - lorentz_inner(a2, w1) * w1
    n2 = lorentz_inner(w2, w2)
    if n2 <= 1e-15:
        a2 = np.array([0.0, 1.0, 0.0])
        w2 = a2 + lorentz_inner(a2, mu_vec) * mu_vec - lorentz_inner(a2, w1) * w1
        n2 = lorentz_inner(w2, w2)
    w2 = w2 / np.sqrt(n2)
    return w1, w2


def sample_uniform_noise(delta: float, e1: np.ndarray, e2: np.ndarray, rng: np.random.Generator) -> np.ndarray:

    U = rng.random()
    R = delta * np.sqrt(U)
    theta = rng.uniform(0.0, 2.0 * np.pi)
    return R * (np.cos(theta) * e1 + np.sin(theta) * e2)

def step(metric, x: np.ndarray, mu: np.ndarray, phi: float, eps: np.ndarray) -> np.ndarray:

    v = phi * metric.log(point=x, base_point=mu) + eps
    y = metric.exp(tangent_vec=v, base_point=mu)
    return project_upper(y)

def run_trajectory_poincare(
    N: int,
    phi: float,
    delta: float,
    seed: int,
    r0: float,
    mu_input: tuple[float, float, float] | list[float] | np.ndarray = (1.0, 0.0, 0.0),
    arc_samples: int = 64,
    plot_points: bool = True,
    show_colorbar: bool = True,
    save: bool = False,
    save_dir: str | Path | None = None,
    filename_pattern: str = "phi{phi}_del{delta}_N{N}_seed{seed}_r0{r0}_mu{mu}",
    decimals: int = 3,
    dpi: int = 300,
):
    def _fmt_num(x: float, places: int) -> str:
        s = f"{x:.{places}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s if s != "" else "0"

    def _fmt_mu(mu_arr: np.ndarray, places: int) -> str:
        parts = [_fmt_num(float(v), places) for v in mu_arr]
        parts = [p if p != "" else "0" for p in parts]
        return "_".join(parts)

    mu = project_upper(mu_input)
    e1, e2 = tangent_frame_at(mu)
    rng = np.random.default_rng(seed)

    def dist(x, y) -> float:
        x = project_upper(x)
        y = project_upper(y)
        return float(metric.dist(x, y))

    theta0 = rng.uniform(0.0, 2.0 * np.pi)
    u0 = np.cos(theta0) * e1 + np.sin(theta0) * e2
    x0 = metric.exp(tangent_vec=r0 * u0, base_point=mu)
    x = project_upper(x0)

    xs = np.zeros((N + 1, 3), dtype=float)
    rs = np.zeros(N + 1, dtype=float)
    xs[0] = x
    rs[0] = dist(x, mu)

    for n in range(N):
        eps = sample_uniform_noise(delta, e1, e2, rng)
        x = step(metric, x, mu, phi, eps)
        xs[n + 1] = x
        rs[n + 1] = dist(x, mu)

    pts_all = np.array([to_poincare(x_) for x_ in xs])
    mu_pt = to_poincare(mu)


    tvals = np.linspace(0.0, 1.0, len(xs))
    cmap = matplotlib.colormaps["viridis"]
    norm = Normalize(vmin=0.0, vmax=1.0)

    def _setup_disk_axis():
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        ax.add_artist(
            plt.Circle(
                (0, 0), 1.0,
                fill=False,
                linestyle='-',
                lw=1.3,
                color='0.25',
                zorder=1,
            )
        )
        ax.set_aspect('equal')
        ax.set_xlim(-1.05, +1.05)
        ax.set_ylim(-1.05, +1.05)
        ax.axis('off')
        return fig, ax

    def _add_time_points(ax):
        sc = ax.scatter(
            pts_all[:, 0], pts_all[:, 1],
            c=tvals, s=8, alpha=0.85, zorder=3, cmap=cmap, norm=norm,
            linewidths=0.25, edgecolors='k'
        )
        if show_colorbar:
            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Time")
        return sc

    def _add_start_end_mu(ax):
        ax.scatter(
            pts_all[0, 0], pts_all[0, 1],
            s=60, edgecolors='k', facecolors='blue',
            zorder=5, label="start"
        )
        ax.scatter(
            pts_all[-1, 0], pts_all[-1, 1],
            s=60, marker='^', facecolors='black',
            linewidths=1.2, zorder=5, label="end"
        )
        ax.scatter(
            mu_pt[0], mu_pt[1],
            s=60, marker='D', edgecolors='k', facecolors='red',
            linewidths=0.7, zorder=6, label="μ"
        )
        ax.legend(loc="upper right", fontsize=8, scatterpoints=1, handlelength=1.0)

    fig1, ax1 = _setup_disk_axis()

    seg_list, col_list = [], []
    for n in range(N):
        v = metric.log(point=xs[n + 1], base_point=xs[n])
        if not np.all(np.isfinite(v)):
            continue
        ts = np.linspace(0.0, 1.0, max(2, int(arc_samples)))
        arc_pts = np.array([to_poincare(metric.exp(t * v, base_point=xs[n])) for t in ts])

        segs = [arc_pts[i:i + 2] for i in range(len(arc_pts) - 1)]
        seg_list.extend(segs)

        local_mid = 0.5 * (ts[:-1] + ts[1:])
        t0, t1 = tvals[n], tvals[n + 1]
        col_list.extend(cmap(norm(t0 + (t1 - t0) * local_mid)))

    if seg_list:
        lc = LineCollection(seg_list, colors=col_list, linewidths=1.0, alpha=0.9, zorder=2)
        ax1.add_collection(lc)

    if plot_points:
        _add_time_points(ax1)

    _add_start_end_mu(ax1)
    fig1.tight_layout()

    fig2, ax2 = _setup_disk_axis()

    seg_list2, col_list2 = [], []
    for n in range(len(xs)):
        v = metric.log(point=xs[n], base_point=mu)
        if not np.all(np.isfinite(v)):
            continue
        ts = np.linspace(0.0, 1.0, max(2, int(arc_samples)))
        ray_pts = np.array([to_poincare(metric.exp(t * v, base_point=mu)) for t in ts])

        segs = [ray_pts[i:i + 2] for i in range(len(ray_pts) - 1)]
        seg_list2.extend(segs)

        col_list2.extend([cmap(norm(tvals[n]))] * (len(ray_pts) - 1))

    if seg_list2:
        lc2 = LineCollection(seg_list2, colors=col_list2, linewidths=0.9, alpha=0.85, zorder=2)
        ax2.add_collection(lc2)

    if plot_points:
        _add_time_points(ax2)

    _add_start_end_mu(ax2)
    fig2.tight_layout()

    if save and save_dir is not None:
        tokens = {
            "N": int(N),
            "phi": _fmt_num(phi, decimals),
            "delta": _fmt_num(delta, decimals),
            "seed": int(seed),
            "r0": _fmt_num(r0, decimals),
            "mu": _fmt_mu(mu, decimals),
        }
        base = filename_pattern.format(**tokens)
        base = (
            base.strip()
                .replace(" ", "-")
                .replace("/", "-")
                .replace("\\", "-")
                .replace(":", "-")
                .replace(",", "")
        )

        path1 = Path(save_dir) / f"{base}_trajectory.pdf"
        path2 = Path(save_dir) / f"{base}_rays.pdf"

        path1.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(path1, format="pdf", dpi=dpi, bbox_inches="tight")

        path2.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(path2, format="pdf", dpi=dpi, bbox_inches="tight")

    return xs, rs, np.asarray(mu), np.asarray(mu_pt)


# =============================
# RUN
# =============================
if __name__ == "__main__":

    xs, rs, mu, mu_pt = run_trajectory_poincare(
        N=100,
        phi=1.2,
        delta=1.5,
        seed=5,
        r0=1.0,
        mu_input=(np.cosh(0.5), np.sinh(0.5), 0.0),
        arc_samples=64,
        plot_points=True,
        show_colorbar=True,
        save=True,
        save_dir="figs",
    )
