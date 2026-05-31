import os
import pandas as pd
import numpy as np
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.geometry.hyperboloid import Hyperboloid
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ======================================================
# Geometry setup and loading the data
# ======================================================
H2 = Hyperboloid(dim=2)
metric = H2.metric

FR_DISTANCE_SCALE = np.sqrt(2)


def load_fisherrao_as_hyperbolic(csv_path):
    df = pd.read_csv(csv_path)
    df["mu"] = df["mu_log"].astype(float)
    df["sigma"] = df["sigma_log"].astype(float)
    df["Date"] = pd.to_datetime(df["Date"])

    if (df["sigma"] <= 0).any():
        raise ValueError("All sigma values must be strictly positive.")

    return df[["Date", "mu", "sigma"]].sort_values("Date").reset_index(drop=True)


# ======================================================
# Conversions between hyperbolic models
# ======================================================
def upper_to_hyperboloid(x, y):
    X0 = (1 + x ** 2 + y ** 2) / (2 * y)
    X1 = x / y
    X2 = (1 - x ** 2 - y ** 2) / (2 * y)
    return np.column_stack((X0, X1, X2))


def hyperboloid_to_upper(X):
    X0, X1, X2 = X[:, 0], X[:, 1], X[:, 2]
    y = 1.0 / (X0 + X2)
    x = X1 * y
    return x, y


# ======================================================
# Estimation and prediction
# ======================================================
def estimate_mu(hyp_data: np.ndarray):
    mean_estimator = FrechetMean(space=H2)
    mean_estimator.fit(hyp_data)
    return mean_estimator.estimate_


def estimate_phi(hyp_data: np.ndarray, mu_hat: np.ndarray, metric):
    logs = metric.log(point=hyp_data, base_point=mu_hat)
    u, w = logs[:-1], logs[1:]

    num = np.sum(metric.inner_product(w, u, base_point=mu_hat))
    den = np.sum(metric.inner_product(u, u, base_point=mu_hat))

    if np.isclose(den, 0.0):
        return 0.0

    return num / den


def estimate_window(subset: pd.DataFrame):
    x = subset["mu"].to_numpy(dtype=float) / np.sqrt(2)
    y = subset["sigma"].to_numpy(dtype=float)
    hyp = upper_to_hyperboloid(x, y)
    mu_hat = estimate_mu(hyp)
    phi_hat = estimate_phi(hyp, mu_hat, metric)
    last_pt = hyp[-1]

    return mu_hat, phi_hat, last_pt


def predict_k_steps(mu_hat, phi_hat, last_point, k_steps):
    preds = []
    current = last_point.copy()
    for _ in range(k_steps):
        v = metric.log(point=current, base_point=mu_hat)
        next_point = metric.exp(tangent_vec=(phi_hat * v), base_point=mu_hat)
        preds.append(next_point)
        current = next_point

    return np.vstack(preds)


def get_true_params_and_errors(future_df, preds, k_steps):
    actual_params, actual_dates, dists = [], [], []

    for i in range(k_steps):
        if i >= len(future_df):
            actual_params.append((None, None))
            actual_dates.append(None)
            dists.append(np.nan)
            continue

        row = future_df.iloc[i]
        amu, asig = float(row["mu"]), float(row["sigma"])

        actual_params.append((amu, asig))
        actual_dates.append(row["Date"])

        actual_point = upper_to_hyperboloid(np.array([amu / np.sqrt(2)]), np.array([asig]))[0]
        dist = FR_DISTANCE_SCALE * float(metric.dist(preds[i], actual_point))
        dists.append(dist)

    return actual_params, actual_dates, dists


# ======================================================
# Execution Functions
# ======================================================
def fisherrao_predict(df, start_date, window_size, k_steps):
    start = pd.to_datetime(start_date)
    subset = df[df["Date"] >= start].head(window_size)

    if len(subset) < window_size:
        raise ValueError("Not enough available observations in the window.")

    mu_hat, phi_hat, last_point = estimate_window(subset)
    end_date = subset["Date"].iloc[-1]

    preds = predict_k_steps(mu_hat, phi_hat, last_point, k_steps)
    x_pred, y_pred = hyperboloid_to_upper(preds)
    mu_pred, sigma_pred = x_pred * np.sqrt(2), y_pred

    future = df[df["Date"] > end_date].head(k_steps)
    actual_params, actual_dates, dists = get_true_params_and_errors(future, preds, k_steps)

    return mu_pred, sigma_pred, actual_params, actual_dates, dists, end_date


def compute_forecast_errors(df, valid_indices, L_est, L_pred):
    errors = []

    for i in valid_indices:
        subset = df.iloc[i - L_est + 1: i + 1]
        if len(subset) < L_est:
            continue

        mu_hat, phi_hat, last_point = estimate_window(subset)
        preds = predict_k_steps(mu_hat, phi_hat, last_point, L_pred)

        future = df.iloc[i + 1: i + 1 + L_pred]
        if len(future) < L_pred:
            continue

        _, _, dists = get_true_params_and_errors(future, preds, L_pred)
        errors.extend(dists)

    return errors


def run_all_schemes(df, start_date="2022-03-01"):
    start_date = pd.to_datetime(start_date)

    Lest_list = [14, 30, 60]
    Lpred_list = [1, 3, 7]
    max_L_est, max_L_pred = max(Lest_list), max(Lpred_list)

    valid_indices = []
    total_rows = len(df)

    for i in range(total_rows):
        if df["Date"].iloc[i] < start_date:
            continue

        if (i + 1) >= max_L_est and (total_rows - 1 - i) >= max_L_pred:
            valid_indices.append(i)

    results = {}
    for L_est in Lest_list:
        for L_pred in Lpred_list:
            results[(L_est, L_pred)] = compute_forecast_errors(df, valid_indices, L_est, L_pred)

    return results


def extract_phi_timelines(df, start_date="2022-03-01"):
    start_date = pd.to_datetime(start_date)
    Lest_list = [14, 30, 60]

    phi_series = {L_est: [] for L_est in Lest_list}
    phi_dates = {L_est: [] for L_est in Lest_list}
    total_rows = len(df)

    for L_est in Lest_list:
        for i in range(total_rows):
            date_val = df["Date"].iloc[i]

            if date_val < start_date or (i + 1) < L_est:
                continue

            subset = df.iloc[i - L_est + 1: i + 1]
            _, phi_hat, _ = estimate_window(subset)

            phi_series[L_est].append(phi_hat)
            phi_dates[L_est].append(date_val)
    return phi_series, phi_dates


# ======================================================
# Lognormal PDF
# ======================================================
def lognormal_pdf(D, mu, sigma):
    return (1.0 / (D * sigma * np.sqrt(2 * np.pi))) * \
        np.exp(-0.5 * ((np.log(D) - mu) / sigma) ** 2)


# ======================================================
# Visualizations
# ======================================================
def plot_boxplots(results, save_path=None):
    keys = sorted(results.keys())
    data = [results[k] for k in keys]
    labels = [f"E{ke[0]}-P{ke[1]}" for ke in keys]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=labels, showfliers=False, medianprops={"linewidth": 2})
    plt.ylabel("Fisher–Rao forecast error")
    plt.title("Forecast error distributions across 9 schemes")
    plt.grid(alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_phi_timelines(phi_series, phi_dates, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    colors = {14: "tab:blue", 30: "tab:orange", 60: "tab:green"}

    for L_est in sorted(phi_series.keys()):
        ax.plot(phi_dates[L_est], phi_series[L_est], label=f"L_est={L_est}",
                color=colors[L_est], linewidth=1.8)

    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.6)
    ax.set_ylabel("φ")
    ax.set_title("φ for different estimation window sizes", pad=30)
    ax.grid(alpha=0.3)

    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        fontsize=12,
        frameon=False
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_all_predictions_3x5(pred_list, save_path=None):
    rows, cols = 3, 5
    D = np.linspace(0.01, 250, 400)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), sharey=True, constrained_layout=True)
    axes = axes.flatten()

    legend_handles = {}
    panel = 0

    for (mu_pred, sigma_pred, actual_params, actual_dates, dists, end_date) in pred_list:
        for i in range(5):
            ax = axes[panel]
            pdf_pred = lognormal_pdf(D, mu_pred[i], sigma_pred[i])
            line_pred, = ax.plot(D, pdf_pred, "--", color="green", label="Predicted")

            muA, sA = actual_params[i]
            if muA is not None:
                pdf_actual = lognormal_pdf(D, muA, sA)
                line_actual, = ax.plot(D, pdf_actual, color="black", label="Observed")

            legend_handles["Predicted"] = line_pred
            if muA is not None:
                legend_handles["Observed"] = line_actual

            date_str = actual_dates[i].strftime("%Y-%m-%d") if actual_dates[i] is not None else f"step {i + 1}"

            if not np.isnan(dists[i]):
                title_text = f"{date_str}\nFR Dist: {dists[i]:.3f}"
            else:
                title_text = date_str

            ax.set_title(title_text, fontsize=12)
            ax.grid(alpha=0.3)
            ax.set_xlim(0, 250)

            panel += 1

    fig.supylabel("PDF")
    fig.supxlabel("Diameter")

    fig.legend(
        legend_handles.values(), legend_handles.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        fontsize=12,
        frameon=False
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_geodesic(ax, x1, y1, x2, y2, **kwargs):
    if np.isclose(x1, x2):
        ax.plot([x1, x2], [y1, y2], **kwargs)
    else:
        xc = (x2 ** 2 - x1 ** 2 + y2 ** 2 - y1 ** 2) / (2 * (x2 - x1))
        r = np.sqrt((x1 - xc) ** 2 + y1 ** 2)
        theta1, theta2 = np.arctan2(y1, x1 - xc), np.arctan2(y2, x2 - xc)
        t = np.linspace(theta1, theta2, 100)
        ax.plot(xc + r * np.cos(t), r * np.sin(t), **kwargs)


def plot_upper_half_plane_predictions(df, prediction_windows, save_path=None):
    cols = len(prediction_windows)
    fig, axes = plt.subplots(1, cols, figsize=(18, 6), sharey=True, constrained_layout=True)
    if cols == 1: axes = [axes]

    text_outline = [pe.withStroke(linewidth=3, foreground="white")]
    legend_handles = {}
    global_y_min, global_y_max = float("inf"), float("-inf")

    for ax, (start_date, window_size, k_steps) in zip(axes, prediction_windows):
        start = pd.to_datetime(start_date)
        subset = df[df["Date"] >= start].head(window_size)

        mu_hat, phi_hat, last_point = estimate_window(subset)
        end_date = subset["Date"].iloc[-1]

        preds = predict_k_steps(mu_hat, phi_hat, last_point, k_steps)
        x_pred, y_pred = hyperboloid_to_upper(preds)

        future = df[df["Date"] > end_date].head(k_steps)
        actual_params, _, _ = get_true_params_and_errors(future, preds, k_steps)
        valid_params = [p for p in actual_params if p[0] is not None]
        x_actual = np.array([p[0] / np.sqrt(2) for p in valid_params])
        y_actual = np.array([p[1] for p in valid_params])

        all_y_ax = np.concatenate([y_pred, y_actual]) if len(x_actual) > 0 else y_pred
        global_y_min, global_y_max = min(global_y_min, all_y_ax.min()), max(global_y_max, all_y_ax.max())
        date_strs = future["Date"].dt.strftime('%b %d').tolist()

        ax.axhline(0, color="black", linewidth=1.0, alpha=0.5)

        if len(x_actual) > 0:
            ax.scatter(x_actual, y_actual, color="tab:blue", marker="o", s=70, zorder=5, label="Observed")
            for i in range(len(x_actual)):
                y_offset = 15 if i % 2 == 0 else -20
                ax.annotate(
                    date_strs[i], (x_actual[i], y_actual[i]),
                    xytext=(0, y_offset), textcoords="offset points",
                    ha="center", va="center", fontsize=9, color="tab:blue",
                    path_effects=text_outline, arrowprops=dict(arrowstyle="-", color="tab:blue", alpha=0.4, lw=1)
                )

        ax.scatter(x_pred, y_pred, color="tab:red", marker="x", s=70, zorder=5, label="Predicted")

        for h in range(min(k_steps, len(x_actual))):
            lbl = "Geodesic error" if h == 0 else ""
            plot_geodesic(ax, x_actual[h], y_actual[h], x_pred[h], y_pred[h],
                          color="gray", linestyle="-", linewidth=1.5, alpha=0.7, label=lbl)

        all_x = np.concatenate([x_pred, x_actual]) if len(x_actual) > 0 else x_pred
        x_pad = 0.15 * (all_x.max() - all_x.min() + 1e-8)
        ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.2)
        ax.set_title(f"End Window: {end_date.strftime('%Y-%m-%d')}", fontsize=12)
        ax.set_xlabel(r"$x=\mu/\sqrt{2}$")

        if ax == axes[0]: ax.set_ylabel(r"$y=\sigma$")

        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in legend_handles: legend_handles[label] = handle

    y_pad = 0.15 * (global_y_max - global_y_min + 1e-8)
    axes[0].set_ylim(max(0, global_y_min - y_pad), global_y_max + y_pad)

    fig.legend(
        legend_handles.values(), legend_handles.keys(),
        loc="upper center", bbox_to_anchor=(0.5, 1.08),
        ncol=3, fontsize=12, frameon=False
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ======================================================
# MAIN EXECUTION
# ======================================================
if __name__ == "__main__":
    csv_file = "lognormal_parameters_2022.csv"
    output_folder = os.path.dirname(os.path.abspath(csv_file))

    main_data = load_fisherrao_as_hyperbolic(csv_file)

    boxplot_path = os.path.join(output_folder, "forecast_error_boxplots.png")
    phi_path = os.path.join(output_folder, "phi_timelines.png")
    pdf_path = os.path.join(output_folder, "pdf_predictions_3x5.png")
    uhp_path = os.path.join(output_folder, "upper_half_plane_predictions.png")

    # ----- 1) Boxplot -----
    results = run_all_schemes(main_data, start_date="2022-03-01")
    plot_boxplots(results, save_path=boxplot_path)

    # ----- 2) φ(t) plot -----
    phi_series, phi_dates = extract_phi_timelines(main_data, start_date="2022-03-01")
    plot_phi_timelines(phi_series, phi_dates, save_path=phi_path)

    # ----- 3) 3×5 PDF plots -----
    pdf_windows = [
        ("2022-04-01", 30, 5),
        ("2022-07-01", 30, 5),
        ("2022-11-01", 30, 5),
    ]

    pred_list = []
    for (start_date, window_size, k_steps) in pdf_windows:
        mp, sp, ap, ad, dists, ed = fisherrao_predict(
            main_data, start_date, window_size, k_steps
        )
        pred_list.append((mp, sp, ap, ad, dists, ed))

    plot_all_predictions_3x5(pred_list, save_path=pdf_path)

    # ----- 4) Upper half-plane prediction plots -----
    plot_upper_half_plane_predictions(main_data, pdf_windows, save_path=uhp_path)

    print(f"Saved boxplot to: {boxplot_path}")
    print(f"Saved phi plot to: {phi_path}")
    print(f"Saved PDF predictions to: {pdf_path}")
    print(f"Saved upper half-plane plot to: {uhp_path}")
