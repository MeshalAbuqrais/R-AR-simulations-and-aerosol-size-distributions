import os
import pandas as pd
import numpy as np
from datetime import timedelta
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.geometry.hyperboloid import Hyperboloid
import matplotlib.pyplot as plt

# ======================================================
# Geometry setup
# ======================================================
H2 = Hyperboloid(dim=2)
metric = H2.metric

# Fisher–Rao distance scale:
# ds_FR^2 = 2 ds_H^2, hence d_FR = sqrt(2) d_H
FR_DISTANCE_SCALE = np.sqrt(2)

# ======================================================
# Fisher–Rao loader
# ======================================================
def load_fisherrao_as_hyperbolic(csv_path):
    df = pd.read_csv(csv_path)
    df["mu"] = df["mu_log"].astype(float)
    df["sigma"] = df["sigma_log"].astype(float)
    df["Date"] = pd.to_datetime(df["Date"])

    if (df["sigma"] <= 0).any():
        raise ValueError("All sigma values must be strictly positive.")

    return df[["Date", "mu", "sigma"]]


# ======================================================
# Estimation functions
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
# Window estimation for prediction (PDF part)
# ======================================================
def fisherrao_window_estimation(csv_path, start_date, window_size):
    df = load_fisherrao_as_hyperbolic(csv_path)
    df = df.sort_values("Date").reset_index(drop=True)

    start = pd.to_datetime(start_date)

    # Take the first window_size available observations from start_date onward
    subset = df[df["Date"] >= start].head(window_size)

    if len(subset) < window_size:
        raise ValueError("Not enough available observations in the window.")

    x = subset["mu"].to_numpy(dtype=float) / np.sqrt(2)
    y = subset["sigma"].to_numpy(dtype=float)
    hyp = upper_to_hyperboloid(x, y)

    mu_hat = estimate_mu(hyp)
    phi_hat = estimate_phi(hyp, mu_hat, metric)

    end = subset["Date"].iloc[-1]

    return mu_hat, phi_hat, hyp, end, df


# ======================================================
# Prediction & comparison for PDF panels
# ======================================================
def fisherrao_predict(csv_path, start_date, window_size, k_steps):
    mu_hat, phi_hat, hyp, end_date, df = fisherrao_window_estimation(
        csv_path, start_date, window_size
    )

    preds = []
    current = hyp[-1].copy()

    for _ in range(k_steps):
        v = metric.log(point=current, base_point=mu_hat)
        next_point = metric.exp(tangent_vec=phi_hat * v, base_point=mu_hat)
        preds.append(next_point)
        current = next_point

    preds = np.vstack(preds)

    x_pred, y_pred = hyperboloid_to_upper(preds)
    mu_pred = x_pred * np.sqrt(2)
    sigma_pred = y_pred

    actual_params = []
    actual_dates = []
    dists = []

    # Compare against the next k_steps available observations
    future = df[df["Date"] > end_date].head(k_steps)

    for i in range(k_steps):
        if i >= len(future):
            actual_params.append((None, None))
            actual_dates.append(None)
            dists.append(np.nan)
            continue

        row = future.iloc[i]

        amu = float(row["mu"])
        asig = float(row["sigma"])

        actual_params.append((amu, asig))
        actual_dates.append(row["Date"])

        actual_point = upper_to_hyperboloid(
            np.array([amu / np.sqrt(2)]),
            np.array([asig])
        )[0]

        dist = FR_DISTANCE_SCALE * float(metric.dist(preds[i], actual_point))
        dists.append(dist)

    return mu_pred, sigma_pred, actual_params, actual_dates, dists, end_date


# ======================================================
# Lognormal PDF
# ======================================================
def lognormal_pdf(D, mu, sigma):
    return (1.0 / (D * sigma * np.sqrt(2 * np.pi))) * \
        np.exp(-0.5 * ((np.log(D) - mu) / sigma) ** 2)


# ======================================================
# 3×5 PDF visualization
# ======================================================
def plot_all_predictions_3x5(pred_list, save_path=None):
    rows, cols = 3, 5
    D = np.linspace(0.01, 250, 400)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5 * cols, 3 * rows),
        sharey=True,
        constrained_layout=True
    )
    axes = axes.flatten()

    panel = 0

    for (mu_pred, sigma_pred, actual_params, actual_dates, end_date) in pred_list:
        for i in range(5):
            ax = axes[panel]

            # predicted PDF (green dashed)
            pdf_pred = lognormal_pdf(D, mu_pred[i], sigma_pred[i])
            ax.plot(D, pdf_pred, "--", color="green", label="Predicted")

            # actual PDF (black)
            muA, sA = actual_params[i]
            if muA is not None:
                pdf_actual = lognormal_pdf(D, muA, sA)
                ax.plot(D, pdf_actual, color="black", label="Actual")

            # Use the actual observed date from the CSV
            if actual_dates[i] is not None:
                date = actual_dates[i].strftime("%Y-%m-%d")
            else:
                date = f"step {i + 1}"

            ax.set_title(date, fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_xlim(0, 250)

            if panel == 0:
                ax.legend(fontsize=8)

            panel += 1

    fig.supylabel("PDF")
    fig.supxlabel("Diameter")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# ======================================================
# ONE estimation window
# ======================================================
def estimate_window(subset, L_est):
    if len(subset) < L_est:
        return None

    x = subset["mu"].to_numpy(dtype=float) / np.sqrt(2)
    y = subset["sigma"].to_numpy(dtype=float)
    hyp = upper_to_hyperboloid(x, y)

    mu_hat = estimate_mu(hyp)
    phi_hat = estimate_phi(hyp, mu_hat, metric)
    last_pt = hyp[-1]

    return mu_hat, phi_hat, last_pt


# ======================================================
# Predict k steps ahead
# ======================================================
def predict_k_steps(mu_hat, phi_hat, last_point, k_steps):
    preds = []
    current = last_point.copy()

    for _ in range(k_steps):
        v = metric.log(point=current, base_point=mu_hat)
        next_point = metric.exp(tangent_vec=(phi_hat * v), base_point=mu_hat)
        preds.append(next_point)
        current = next_point

    return np.vstack(preds)


# ======================================================
# Forecast errors for one (L_est, L_pred)
# ======================================================
def compute_forecast_errors(df, valid_indices, L_est, L_pred):
    errors = []

    for i in valid_indices:
        subset = df.iloc[i - L_est + 1: i + 1]

        est = estimate_window(subset, L_est)
        if est is None:
            continue

        mu_hat, phi_hat, last_point = est
        preds = predict_k_steps(mu_hat, phi_hat, last_point, L_pred)

        future = df.iloc[i + 1: i + 1 + L_pred]

        if len(future) < L_pred:
            continue

        for h in range(1, L_pred + 1):
            actual = future.iloc[h - 1]

            amu = float(actual["mu"])
            asig = float(actual["sigma"])

            act = upper_to_hyperboloid(
                np.array([amu / np.sqrt(2)]),
                np.array([asig])
            )[0]

            dist = FR_DISTANCE_SCALE * float(metric.dist(preds[h - 1], act))
            errors.append(dist)

    return errors


# ======================================================
# MAIN MULTI-SCHEME FORECASTING
# ======================================================
def run_all_schemes(csv_path, start_date="2022-03-01"):
    df = load_fisherrao_as_hyperbolic(csv_path)
    df = df.sort_values("Date").reset_index(drop=True)

    start_date = pd.to_datetime(start_date)

    Lest_list = [14, 30, 60]
    Lpred_list = [1, 3, 7]

    max_L_est = max(Lest_list)
    max_L_pred = max(Lpred_list)

    valid_indices = []
    total_rows = len(df)

    for i in range(total_rows):
        if df["Date"].iloc[i] < start_date:
            continue

        past_count = i + 1
        future_count = total_rows - 1 - i

        if past_count >= max_L_est and future_count >= max_L_pred:
            valid_indices.append(i)

    results = {}

    for L_est in Lest_list:
        for L_pred in Lpred_list:
            errs = compute_forecast_errors(df, valid_indices, L_est, L_pred)
            results[(L_est, L_pred)] = errs

    return results


# ======================================================
# BOXPLOT
# ======================================================
def plot_boxplots(results, save_path=None):
    keys = sorted(results.keys())
    data = [results[k] for k in keys]
    labels = [f"E{ke[0]}-P{ke[1]}" for ke in keys]

    plt.figure(figsize=(12, 6))
    plt.boxplot(
        data,
        labels=labels,
        showfliers=False,
        medianprops={"linewidth": 2}
    )
    plt.ylabel("Fisher–Rao forecast error")
    plt.title("Forecast error distributions across 9 schemes")
    plt.grid(alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# ======================================================
# φ(t) extraction
# ======================================================
def extract_phi_timelines(csv_path, start_date="2022-03-01"):
    df = load_fisherrao_as_hyperbolic(csv_path)
    df = df.sort_values("Date").reset_index(drop=True)

    start_date = pd.to_datetime(start_date)
    Lest_list = [14, 30, 60]

    phi_series = {L_est: [] for L_est in Lest_list}
    phi_dates = {L_est: [] for L_est in Lest_list}

    total_rows = len(df)

    for L_est in Lest_list:
        for i in range(total_rows):
            date_val = df["Date"].iloc[i]

            if date_val < start_date:
                continue

            past_count = i + 1

            if past_count < L_est:
                continue

            subset = df.iloc[i - L_est + 1: i + 1]
            est = estimate_window(subset, L_est)

            if est is not None:
                _, phi_hat, _ = est
                phi_series[L_est].append(phi_hat)
                phi_dates[L_est].append(date_val)

    return phi_series, phi_dates


# ======================================================
# φ(t) PLOT
# ======================================================
def plot_phi_timelines(phi_series, phi_dates, save_path=None):
    plt.figure(figsize=(12, 6))

    colors = {14: "tab:blue", 30: "tab:orange", 60: "tab:green"}

    for L_est in sorted(phi_series.keys()):
        plt.plot(
            phi_dates[L_est],
            phi_series[L_est],
            label=f"L_est={L_est}",
            color=colors[L_est],
            linewidth=1.8
        )

    plt.axhline(1.0, color="grey", linestyle="--", alpha=0.6)
    plt.ylabel("φ")
    plt.title("φ for different estimation window sizes")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# ======================================================
# RUN FULL ANALYSIS
# ======================================================
if __name__ == "__main__":
    csv_file = "lognormal_parameters_2022.csv"

    # Save figures in the same folder as the CSV file
    output_folder = os.path.dirname(os.path.abspath(csv_file))

    boxplot_path = os.path.join(output_folder, "forecast_error_boxplots.png")
    phi_path = os.path.join(output_folder, "phi_timelines.png")
    pdf_path = os.path.join(output_folder, "pdf_predictions_3x5.png")

    # ----- 1) Boxplot -----
    results = run_all_schemes(csv_file, start_date="2022-03-01")
    plot_boxplots(results, save_path=boxplot_path)

    # ----- 2) φ(t) plot -----
    phi_series, phi_dates = extract_phi_timelines(csv_file, start_date="2022-03-01")
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
            csv_file,
            start_date,
            window_size,
            k_steps
        )

        pred_list.append((mp, sp, ap, ad, ed))

    plot_all_predictions_3x5(pred_list, save_path=pdf_path)

    print(f"Saved boxplot to: {boxplot_path}")
    print(f"Saved phi plot to: {phi_path}")
    print(f"Saved PDF predictions to: {pdf_path}")
