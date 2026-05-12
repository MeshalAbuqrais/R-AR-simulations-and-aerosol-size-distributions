import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib.ticker import MaxNLocator
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.learning.frechet_mean import FrechetMean


H2 = Hyperboloid(dim=2)
metric = H2.metric


def lorentz_inner(x, y):
    return -x[0]*y[0] + x[1]*y[1] + x[2]*y[2]


def tangent_frame_at(mu):
    mu = np.asarray(mu, float)

    a1 = np.array([0., 1., 0.])
    a2 = np.array([0., 0., 1.])

    w1 = a1 + lorentz_inner(a1, mu) * mu
    w1 /= np.sqrt(lorentz_inner(w1, w1))

    w2 = (
        a2
        + lorentz_inner(a2, mu) * mu
        - lorentz_inner(a2, w1) * w1
    )
    w2 /= np.sqrt(lorentz_inner(w2, w2))

    return w1, w2



def sample_uniform_noise(delta, e1, e2, rng):
    U = rng.random()
    R = delta * np.sqrt(U)
    theta = rng.uniform(0, 2*np.pi)
    return R * (np.cos(theta)*e1 + np.sin(theta)*e2)


def step(x, mu, phi, eps):
    v = phi * metric.log(x, mu) + eps
    return metric.exp(v, mu)

def simulate_process(mu, phi, delta, T, seed):
    mu = np.asarray(mu, float)
    e1, e2 = tangent_frame_at(mu)
    rng = np.random.default_rng(seed)

    x = metric.exp(sample_uniform_noise(delta, e1, e2, rng), mu)

    data = []
    for _ in range(T):
        eps = sample_uniform_noise(delta, e1, e2, rng)
        x = step(x, mu, phi, eps)
        if not np.all(np.isfinite(x)):
            break
        data.append(np.asarray(x, float))

    return np.asarray(data)

def frechet_and_phi_estimation_function(mu, data, phi_true, n_step, burn_in):
    mu = np.asarray(mu, float)
    mean_est = FrechetMean(space=H2)

    n_vals = np.arange(burn_in, len(data) + 1, n_step)
    mu_err = np.empty(len(n_vals))
    phi_err = np.empty(len(n_vals))

    for i, n in enumerate(n_vals):
        mean_est.fit(data[:n])
        mu_hat = mean_est.estimate_
        mu_err[i] = metric.dist(mu, mu_hat)

        logs = metric.log(data[:n], mu_hat)
        u = logs[:-1]
        v = logs[1:]

        num = np.sum(metric.inner_product(v, u, mu_hat))
        den = np.sum(metric.inner_product(u, u, mu_hat))
        phi_hat = num / den if den > 1e-15 else np.nan
        phi_err[i] = np.abs(phi_hat - phi_true)

    return n_vals, mu_err, phi_err

def compute_grid(n):
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols

def run_experiment(config, mode, base_dir):
    mu_true = config["mu"]
    phis = config["phis"]

    out_dir = os.path.join(base_dir, config["out_dir"])
    os.makedirs(out_dir, exist_ok=True)

    n_plots = len(phis)
    nrows, ncols = compute_grid(n_plots)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False
    )

    global_max = 0.0
    stored = []

    for phi in phis:
        for k in range(config["n_paths"]):
            data = simulate_process(
                mu_true, phi, config["delta"],
                config["N"], config["seed_base"] + k
            )
            if len(data) < config["burn_in"]:
                continue

            n_vals, mu_err, phi_err = frechet_and_phi_estimation_function(
                mu_true, data, phi,
                config["n_step"], config["burn_in"]
            )

            err = mu_err if mode == "mu" else phi_err
            global_max = max(global_max, np.nanmax(err))
            stored.append((phi, n_vals, err))

    for idx, phi in enumerate(phis):
        ax = axes.flat[idx]

        for phi_i, n_vals, err in stored:
            if phi_i == phi:
                ax.plot(n_vals, err, alpha=0.7)

        ax.set_title(rf"$\phi={phi}$")
        ax.set_xlabel("Sample size $n$")
        ax.set_ylim(0, 1.05 * global_max)
        ax.xaxis.set_major_locator(MaxNLocator(8))
        ax.grid(alpha=0.3)

    for ax in axes.flat[n_plots:]:
        ax.axis("off")

    ylabel = (
        r"$d_{H^2}(\mu,\hat\mu_n)$"
        if mode == "mu"
        else r"$|\hat\phi_n - \phi|$"
    )
    for r in range(nrows):
        axes[r, 0].set_ylabel(ylabel)

    fname = (
        "frechet_mu_paths.pdf"
        if mode == "mu"
        else "frechet_phi_paths.pdf"
    )

    plt.tight_layout()
    plt.subplots_adjust(left=0.12)
    plt.savefig(
        os.path.join(out_dir, fname),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


config_mu = {
    "phis": [0.8, 0.9, 0.95, 0.99],
    "n_paths": 5,
    "N": 1000,
    "n_step": 20,
    "burn_in": 10,
    "delta": 0.1,
    "seed_base": 2,
    "mu": (1., 0., 0.),
    "out_dir": "fig_paths"
}

config_phi = {
    "phis": [0.8, 0.9, 0.95, 0.99],
    "n_paths": 5,
    "N": 200,
    "n_step": 20,
    "burn_in": 10,
    "delta": 0.1,
    "seed_base": 100,
    "mu": (1., 0., 0.),
    "out_dir": "fig_paths"
}


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    run_experiment(config_mu, mode="mu", base_dir=BASE_DIR)
    run_experiment(config_phi, mode="phi", base_dir=BASE_DIR)

