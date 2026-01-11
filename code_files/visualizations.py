from .imports_and_libraries import *
from .dataset_creation import *


def plot_wave_samples(t, V_clean, V_noisy, w: float=1.0, mu: float=cfg.mu, sigma: float=cfg.noise_std, save_plot: bool=False, show_plot: bool=False) -> None:
    """
    Plot one sine wave example with and without Gaussian noise.

    Parameters
    ----------
    t : array-like
        Time grid of shape (T,).
    V_clean : array-like
        Clean signal values of shape (T,).
    V_noisy : array-like
        Noisy signal values of shape (T,).
    w : float, optional
        True frequency used for title/label. Defaults to 1.0.
    mu : float, optional
        Noise mean used for title/filename. Defaults to `cfg.mu`.
    sigma : float, optional
        Noise std used for title/filename. Defaults to `cfg.noise_std`.
    save_plot : bool, optional
        If True, save figure to `cfg.plots_dir`. Defaults to False.
        Save plot name is like: 'sine_with_noise_mu{mu}_sigma{sigma}_example.png
    show_plot : bool, optional
        If True, display figure window. Defaults to False.

    Returns -> None
    """
    plt.figure(figsize=(8,6))
    plt.plot(t, V_clean, label="clean sine", linewidth=2)
    plt.scatter(t, V_noisy, s=15, alpha=0.7, label="noisy sample", c='black', marker='x')

    plt.xlabel("t")
    plt.ylabel("sin(w*t)")
    plt.title(f"Sine example with noise (w={w:.3f}, mu={mu}, sigma={sigma})")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    if save_plot:
        plt.savefig(cfg.plots_dir / f"sine_with_noise_mu{mu}_sigma{sigma}_example.png", dpi=300)

    if show_plot:
        plt.show()

    plt.close()


def plot_pred_vs_true(y_true, y_pred, test_mse, test_mae, N: int=cfg.num_of_samples, t_disc: int=cfg.discr_of_time, w_min: float=cfg.omega_min, 
                      w_max: float=cfg.omega_max, seed=cfg.seed, sigma: float=cfg.noise_std, folder=cfg.plots_dir, save_plot: bool=False, show_plot: bool=False) -> None:
    """
    Scatter plot of predicted vs. true omega for the single-frequency model.

    The function draws a y=x reference line and prints MSE/MAE in the title.

    Parameters
    ----------
    y_true : torch.Tensor or np.ndarray
        True omega values, shape (N, 1) or (N,). If torch.Tensor, it is expected
        to be on CPU or convertible via `.numpy()`.
    y_pred : torch.Tensor or np.ndarray
        Predicted omega values, shape (N, 1) or (N,).
    test_mse : float
        Test mean squared error displayed in the plot title.
    test_mae : float
        Test mean absolute error displayed in the plot title.
    N : int, optional
        Number of samples (used only for title/filename). Defaults to `cfg.num_of_samples`.
    t_disc : int, optional
        Time discretization length (used only for title/filename). Defaults to `cfg.discr_of_time`.
    w_min : float, optional
        Minimum omega (used only for title/filename). Defaults to `cfg.omega_min`.
    w_max : float, optional
        Maximum omega (used only for title/filename). Defaults to `cfg.omega_max`.
    seed : int, optional
        Random seed identifier (used only for filename). Defaults to `cfg.seed`.
    sigma : float, optional
        Noise std identifier (used only for title/filename). Defaults to `cfg.noise_std`.
    folder : pathlib.Path or str, optional
        Output directory used when `save_plot=True`. Defaults to `cfg.plots_dir`.
    save_plot : bool, optional
        If True, saves PNG into `folder` with filename:
        `T2_N{N}_tdis{t_disc}_w{w_min}-{w_max}_seed{seed}_std{sigma}_PREDvsREAL.png`.
        Defaults to False.
    show_plot : bool, optional
        If True, displays the figure. Defaults to False.

        
    Returns -> None
    """

    plt.figure(figsize=(6,6))
    plt.scatter(y_true.numpy(), y_pred.numpy(), s=14, alpha=0.6)
    mn = min(y_true.min().item(), y_pred.min().item())
    mx = max(y_true.max().item(), y_pred.max().item())
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)
    plt.grid(True, which="both")
    plt.xlabel("True w")
    plt.ylabel("Predicted w")
    plt.title(f"Test N={N}, w=[{w_min}-{w_max}], tdis={t_disc}\nMSE={test_mse:.6f}, MAE={test_mae:.6f}, std={sigma}")
    plt.tight_layout()
    if save_plot:
        plt.savefig(folder / f"T2_N{N}_tdis{t_disc}_w{w_min}-{w_max}_seed{seed}_std{sigma}_PREDvsREAL.png", dpi=300)
    if show_plot:
        plt.show()
    
    plt.close()


def plot_loss_curves(train_mse_hist, val_mse_hist, epochs: int=cfg.epochs, N: int=cfg.num_of_samples, t_disc: int=cfg.discr_of_time, w_min: float=cfg.omega_min, 
                     w_max: float=cfg.omega_max, seed=cfg.seed, folder=cfg.plots_dir, sigma: float=cfg.noise_std, save_plot: bool=False, show_plot: bool=False, 
                     y_limit: float=None, zoom: str="full", name_suf: str="") -> None:
    """
    Plot training and validation MSE curves over epochs.

    Parameters
    ----------
    train_mse_hist : list[float] or array-like
        Training MSE values per epoch (length should be `epochs`).
    val_mse_hist : list[float] or array-like
        Validation MSE values per epoch (length should be `epochs`).
    epochs : int, optional
        Number of epochs shown on x-axis. Defaults to `cfg.epochs`.
    N : int, optional
        Number of samples (used only for title/filename). Defaults to `cfg.num_of_samples`.
    t_disc : int, optional
        Time discretization length (used only for title/filename). Defaults to `cfg.discr_of_time`.
    w_min : float, optional
        Minimum omega (used only for title/filename). Defaults to `cfg.omega_min`.
    w_max : float, optional
        Maximum omega (used only for title/filename). Defaults to `cfg.omega_max`.
    seed : int, optional
        Random seed identifier (used only for filename). Defaults to `cfg.seed`.
    folder : pathlib.Path or str, optional
        Output directory used when `save_plot=True`. Defaults to `cfg.plots_dir`.
    sigma : float, optional
        Noise std identifier (used only for title/filename). Defaults to `cfg.noise_std`.
    save_plot : bool, optional
        If True, saves PNG into `folder` with filename:
        `T2{name_suf}_N{N}_tdis{t_disc}_std{sigma}_w{w_min}-{w_max}_seed{seed}_LOSSf_{zoom}.png`.
        Defaults to False.
    show_plot : bool, optional
        If True, displays the figure. Defaults to False.
    y_limit : float or None, optional
        If provided, sets y-axis upper limit (zoom). Defaults to None (no limit).
    zoom : str, optional
        Label used in saved filename to distinguish zoom settings. Defaults to "full".
    name_suf : str, optional
        Extra suffix inserted after "T2" in the saved filename (e.g. model tag).
        Defaults to "".

    Returns -> None
    """

    epochs_axis = range(1, epochs + 1)
    plt.figure(figsize=(8,5))
    plt.minorticks_on()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.plot(epochs_axis, train_mse_hist, label="Train MSE")
    plt.plot(epochs_axis, val_mse_hist,   label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.ylim(bottom=0, top=y_limit)
    plt.title(f"Training/Validation Loss \n N={N}, w=[{w_min}-{w_max}], tdis={t_disc}, std{sigma}")
    plt.legend()
    plt.tight_layout()

    if save_plot:
        plt.savefig(folder / f"T2{name_suf}_N{N}_tdis{t_disc}_std{sigma}_w{w_min}-{w_max}_seed{seed}_LOSSf_{zoom}.png", dpi=300)
    if show_plot:
        plt.show()

    plt.close()


def plot_val_curves_fixed_N(results, N, folder=cfg.plots_dir, save_plot: bool=False, show_plot: bool=False, y_limit: float=None, zoom: str="full") -> None:
    """
    Plot training and validation MSE curves over epochs.

    Parameters
    ----------
    train_mse_hist : list[float] or array-like
        Training MSE values per epoch (length should be `epochs`).
    val_mse_hist : list[float] or array-like
        Validation MSE values per epoch (length should be `epochs`).
    epochs : int, optional
        Number of epochs shown on x-axis. Defaults to `cfg.epochs`.
    N : int, optional
        Number of samples (used only for title/filename). Defaults to `cfg.num_of_samples`.
    t_disc : int, optional
        Time discretization length (used only for title/filename). Defaults to `cfg.discr_of_time`.
    w_min : float, optional
        Minimum omega (used only for title/filename). Defaults to `cfg.omega_min`.
    w_max : float, optional
        Maximum omega (used only for title/filename). Defaults to `cfg.omega_max`.
    seed : int, optional
        Random seed identifier (used only for filename). Defaults to `cfg.seed`.
    folder : pathlib.Path or str, optional
        Output directory used when `save_plot=True`. Defaults to `cfg.plots_dir`.
    sigma : float, optional
        Noise std identifier (used only for title/filename). Defaults to `cfg.noise_std`.
    save_plot : bool, optional
        If True, saves PNG into `folder` with filename:
        `T2{name_suf}_N{N}_tdis{t_disc}_std{sigma}_w{w_min}-{w_max}_seed{seed}_LOSSf_{zoom}.png`.
        Defaults to False.
    show_plot : bool, optional
        If True, displays the figure. Defaults to False.
    y_limit : float or None, optional
        If provided, sets y-axis upper limit (zoom). Defaults to None (no limit).
    zoom : str, optional
        Label used in saved filename to distinguish zoom settings. Defaults to "full".
    name_suf : str, optional
        Extra suffix inserted after "T2" in the saved filename (e.g. model tag).
        Defaults to "".

    Returns -> None
    """

    subset = [r for r in results if r["N"] == N]
    if not subset:
        print(f"No results for N={N}")
        return

    plt.figure(figsize=(8, 5))
    plt.grid(True, which="both")

    for r in subset:
        epochs_axis = range(1, len(r["val_curve"]) + 1)
        plt.plot(epochs_axis, r["val_curve"], label=f"t_disc={r['t_disc']}")

    plt.xlabel("Epoch")
    plt.ylabel("Val MSE")
    plt.ylim(bottom=0, top=y_limit)
    plt.title(f"Val MSE vs Epoch for N={N}")
    plt.legend()
    plt.tight_layout()

    if save_plot:
        plt.savefig(folder / f"VALcurves_N{N}_{zoom}.png", dpi=300)
    if show_plot:
        plt.show()

    plt.close()


def plot_parallel_hparams( csv_path: str, top_k: int | None=None, renderer: str="browser", dims: list[str] | None=None, color_col: str="best_val", 
                title_prefix: str="Parallel coordinates", show: bool=False, save_path: str | None=None) -> None:
    """
     Plot a Plotly parallel-coordinates chart for hyperparameter search results stored in a CSV.

    The plot is built from a subset of rows:
    - if top_k is None: uses all rows
    - else: uses the top_k rows with the smallest values in `color_col`

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing hyperparameter runs.
    top_k : int or None, optional
        If None, uses all rows. If int, uses the `top_k` rows with the smallest
        values in `color_col`. Defaults to None.
    renderer : str, optional
        Plotly renderer name (e.g. "browser", "notebook"). Defaults to "browser".
    dims : list[str] or None, optional
        Column names used as axes in the parallel plot. If None, defaults to:
        ["d_model", "nhead", "num_layers", "dim_f", color_col].
    color_col : str, optional
        Column used both for coloring and for selecting top_k. Defaults to "best_val".
    title_prefix : str, optional
        Prefix text used in the figure title. Defaults to "Parallel coordinates".
    show : bool, optional
        If True, displays the interactive figure. Defaults to False.
    save_path : str or None, optional
        If provided, saves the figure:
        - to HTML if save_path ends with ".html"
        - otherwise uses Plotly image export (requires kaleido).
        Defaults to None.

    Returns -> None
    """

    pio.renderers.default = renderer

    df = pd.read_csv(csv_path)

    if dims is None:
        dims = ["d_model", "nhead", "num_layers", "dim_f", color_col]

    # choose subset
    if top_k is None:
        subset = df
        suffix = "ALL models"
    else:
        subset = df.nsmallest(top_k, color_col)
        suffix = f"TOP {top_k} models (lowest {color_col})"

    # global range for colour/axis
    global_min = df[color_col].min()
    global_max = df[color_col].max()

    cmin = subset[color_col].min()
    cmax = subset[color_col].max()

    fig = px.parallel_coordinates(subset[dims], dimensions=dims, color=color_col, color_continuous_scale="Viridis", range_color=(cmax, cmin))

    fig.update_layout(title=f"{title_prefix} – {suffix}", width=1800, height=900)

    if save_path is not None:
        if save_path.lower().endswith(".html"):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path, width=1800, height=900, scale=2.0)

    if show:
        fig.show()


#flowt things
@torch.no_grad()
def plot_dataset_vs_learned_marginal(model: nn.Module, device, loader, num_samples_per_x: int=100, bins: int=50, N: int=cfg.num_of_samples, 
                    t_disc: int=cfg.discr_of_time, w_min: float=cfg.omega_min, w_max: float=cfg.omega_max, seed=cfg.seed, folder=cfg.plots_dir, 
                    sigma: float=cfg.noise_std, fl_hid_feat: int=cfg.flow_hidden_features, fl_lay: int=cfg.flow_num_layers,
                    save_plot: bool=False, show_plot: bool=False) -> None:
    """
    Compare dataset target distribution to flow-sampled predictive distribution.

    The function builds two 1D histograms:
    1) All true targets ω from `loader` (concatenated across batches)
    2) Flow samples ω ~ p(ω | x) drawn via `model.sample(xb, num_samples=num_samples_per_x)`

    Parameters
    ----------
    model : torch.nn.Module
        A model implementing `sample(x, num_samples=...)` returning ω samples.
    device : torch.device
        Device used for running the model.
    loader : DataLoader
        DataLoader returning batches (xb, yb), where yb has shape (batch, 1) or (batch,).
    num_samples_per_x : int, optional
        Number of ω samples drawn per input xb item. Defaults to 100.
    bins : int, optional
        Number of histogram bins for both distributions. Defaults to 50.
    N : int, optional
        Dataset size used only for filename metadata. Defaults to `cfg.num_of_samples`.
    t_disc : int, optional
        Time discretization used only for filename metadata. Defaults to `cfg.discr_of_time`.
    w_min : float, optional
        Minimum omega used only for filename metadata. Defaults to `cfg.omega_min`.
    w_max : float, optional
        Maximum omega used only for filename metadata. Defaults to `cfg.omega_max`.
    seed : int, optional
        Seed identifier used only for filename metadata. Defaults to `cfg.seed`.
    folder : pathlib.Path or str, optional
        Output directory used when `save_plot=True`. Defaults to `cfg.plots_dir`.
    sigma : float, optional
        Noise std identifier used only for filename metadata. Defaults to `cfg.noise_std`.
    fl_hid_feat : int, optional
        Flow hidden feature count used only for filename metadata. Defaults to `cfg.flow_hidden_features`.
    fl_lay : int, optional
        Flow layer count used only for filename metadata. Defaults to `cfg.flow_num_layers`.
    save_plot : bool, optional
        If True, saves PNG with filename:
        `dataset_vs_learned_marginal_flow_T2_flowHidFeat{fl_hid_feat}_flowLay{fl_lay}_N{N}_tdis{t_disc}_std{sigma}_w{w_min}-{w_max}_seed{seed}.png`.
        Defaults to False.
    show_plot : bool, optional
        If True, displays the figure. Defaults to False.
    Returns -> None
    """
    model.eval()

    all_targets = []
    all_model_samples = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        all_targets.append(yb.squeeze(-1).cpu().numpy())

        samples = model.sample(xb, num_samples=num_samples_per_x)
        samples = samples.squeeze(-1).cpu().numpy().reshape(-1)
        all_model_samples.append(samples)

    targets = np.concatenate(all_targets)
    flow_samples = np.concatenate(all_model_samples)

    plt.figure(figsize=(8, 5))
    plt.minorticks_on()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.hist( targets, bins=bins, density=True, alpha=0.7, label="dataset ω (targets)")
    plt.hist(flow_samples, bins=bins, density=True, alpha=0.7, label="flow samples ω (model)")

    plt.xlabel("ω")
    plt.ylabel("density")
    plt.title("Dataset vs learned ω distribution (flow head)")
    plt.legend()
    plt.tight_layout()

    if save_plot:
        path = cfg.plots_dir / f"dataset_vs_learned_marginal_flow_T2_flowHidFeat{fl_hid_feat}_flowLay{fl_lay}_N{N}_tdis{t_disc}_std{sigma}_w{w_min}-{w_max}_seed{seed}.png"
        plt.savefig(path, dpi=300)

    if show_plot:
        plt.show()

    plt.close()

@torch.no_grad()
def plot_flow_posterior_one_example(model: nn.Module, device, loader, global_index: int=0, num_samples: int=100000, bins: int=100, num_sigmas: int=3, 
                    N: int=cfg.num_of_samples, t_disc: int=cfg.discr_of_time, w_min: float=cfg.omega_min, w_max: float=cfg.omega_max, 
                    seed=cfg.seed, folder=cfg.plots_dir, fl_hid_feat: int=cfg.flow_hidden_features,
                    fl_lay: int=cfg.flow_num_layers, save_plot: bool=False, show_plot: bool=False) -> None:
    """
    Plot the learned 1D conditional posterior p(ω | x) for one chosen example.

    The function selects one example (by `global_index`) from the dataloader,
    draws many samples ω ~ p(ω | x) using `model.sample`, and plots a histogram
    with vertical lines showing:
    - true ω
    - sample mean μ
    - ±kσ reference lines (for k=1..num_sigmas)

    Parameters
    ----------
    model : torch.nn.Module
        A model implementing `sample(x, num_samples=...)` returning ω samples.
    device : torch.device
        Device used for running the model.
    loader : DataLoader
        DataLoader returning batches (xb, yb). Targets yb must contain ω per sample.
    global_index : int, optional
        Global index into the dataset order provided by `loader` iteration.
        Defaults to 0.
    num_samples : int, optional
        Number of Monte Carlo samples drawn from the flow for the selected x.
        Defaults to 100000.
    bins : int, optional
        Number of histogram bins. Defaults to 100.
    num_sigmas : int, optional
        Number of ±kσ reference lines to draw. Defaults to 3.
    N : int, optional
        Dataset size used only for filename metadata. Defaults to `cfg.num_of_samples`.
    t_disc : int, optional
        Time discretization used only for filename metadata. Defaults to `cfg.discr_of_time`.
    w_min : float, optional
        Minimum omega used only for filename metadata. Defaults to `cfg.omega_min`.
    w_max : float, optional
        Maximum omega used only for filename metadata. Defaults to `cfg.omega_max`.
    seed : int, optional
        Seed identifier used only for filename metadata. Defaults to `cfg.seed`.
    folder : pathlib.Path or str, optional
        Output directory used when `save_plot=True`. Defaults to `cfg.plots_dir`.
    fl_hid_feat : int, optional
        Flow hidden feature count used only for filename metadata. Defaults to `cfg.flow_hidden_features`.
    fl_lay : int, optional
        Flow layer count used only for filename metadata. Defaults to `cfg.flow_num_layers`.
    save_plot : bool, optional
        If True, saves PNG with filename:
        `Probab_density_T2_flowHidFeat{fl_hid_feat}_flowLay{fl_lay}_N{N}_tdis{t_disc}_std{sigma}_w{w_min}-{w_max}_seed{seed}.png`.
        (Uses your existing path logic.) Defaults to False.
    show_plot : bool, optional
        If True, displays the figure. Defaults to False.


    Returns -> None
    """
    model.eval()

    start = 0
    for xb, yb in loader:
        batch_size = xb.size(0)
        end = start + batch_size
        if global_index < end:
            local_idx = global_index - start

            xb = xb.to(device)
            yb = yb.to(device)

            x_one = xb[local_idx : local_idx + 1]
            w_true = yb[local_idx].item()
            break

        start = end
    else:
        raise IndexError(f"global_index {global_index} out of range")

    samples = model.sample(x_one, num_samples=num_samples)
    samples = samples.squeeze().cpu().numpy()

    mu = samples.mean()
    sigma = samples.std()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.minorticks_on()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    ax.hist(samples, bins=bins, density=True, color="tab:blue", alpha=0.8, label="flow samples ω | x")
    ax.axvline(w_true, color="tab:orange", linestyle="--", linewidth=2, label=f"true ω = {w_true:.3f}")
    ax.axvline(mu, color="tab:red", linestyle="-", linewidth=2, label=f"mean μ = {mu:.3f}")


    ax.set_xlim(mu - 0.15, mu + 0.15)
    ax.set_xlabel("ω")
    ax.set_ylabel("'Samples'")
    ax.set_title("Probability for one example of ω")

    sigma_label_added = False
    for k in range(1, num_sigmas + 1):
        left  = mu - k * sigma
        right = mu + k * sigma

        label_sigma = r"±kσ lines" if not sigma_label_added else None
        sigma_label_added = True

        ax.axvline(left,  linestyle="-.", linewidth=1.8, color="tab:brown", alpha=0.7, label=label_sigma)
        ax.axvline(right, linestyle="-.", linewidth=1.8, color="tab:brown", alpha=0.7)

    ks = [x for x in range(-num_sigmas, num_sigmas + 1, 1)]
    tick_positions = [mu + k * sigma for k in ks]
    tick_labels = []
    
    for k in ks:
        if k == 0:
            tick_labels.append(r"μ")
        elif k < 0:
            tick_labels.append(rf"{k}σ")
        else:
            tick_labels.append(rf"+{k}σ")

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)
    ax2.tick_params(axis="x", labelsize=8, pad=2)

    ax.legend()
    fig.tight_layout()

    if save_plot:
        path = cfg.plots_dir / f"Probab_density_T2_flowHidFeat{fl_hid_feat}_flowLay{fl_lay}_N{N}_tdis{t_disc}_std{sigma}_w{w_min}-{w_max}_seed{seed}.png"
        plt.savefig(path, dpi=300)

    if show_plot:
        plt.show()

    plt.close()


def plot_error_vs_true_omega(y_true, y_pred, smooth_window_frac: float = 0.075, N: int=cfg.num_of_samples, t_disc: int=cfg.discr_of_time, w_min: float=cfg.omega_min, 
                    w_max: float=cfg.omega_max, seed=cfg.seed, folder=cfg.plots_dir, sigma: float=cfg.noise_std, fl_hid_feat: int=cfg.flow_hidden_features,
                    fl_lay: int=cfg.flow_num_layers, save_plot: bool = False, show_plot: bool = False) -> None:
    """
    Plot absolute prediction error as a function of the true omega.

    The plot includes:
    - scatter of per-sample absolute error |y_pred - y_true|
    - a smoothed curve computed by sorting by y_true and applying a moving average

    Parameters
    ----------
    y_true : array-like
        True omega values, shape (N,) or (N,1). Converted internally via `np.asarray(...).flatten()`.
    y_pred : array-like
        Predicted omega values, shape compatible with y_true. Flattened the same way.
    smooth_window_frac : float, optional
        Fraction of dataset size used as the moving-average window length.
        The actual window is at least 5 and forced to be odd. Defaults to 0.075.
    N : int, optional
        Dataset size used only for filename metadata. Defaults to `cfg.num_of_samples`.
    t_disc : int, optional
        Time discretization used only for filename metadata. Defaults to `cfg.discr_of_time`.
    w_min : float, optional
        Minimum omega used only for filename metadata. Defaults to `cfg.omega_min`.
    w_max : float, optional
        Maximum omega used only for filename metadata. Defaults to `cfg.omega_max`.
    seed : int, optional
        Seed identifier used only for filename metadata. Defaults to `cfg.seed`.
    folder : pathlib.Path or str, optional
        Output directory used when `save_plot=True`. Defaults to `cfg.plots_dir`.
        (Note: your current code saves to `cfg.plots_dir` directly.)
    sigma : float, optional
        Noise std identifier used only for filename metadata. Defaults to `cfg.noise_std`.
    fl_hid_feat : int, optional
        Flow hidden feature count used only for filename metadata. Defaults to `cfg.flow_hidden_features`.
    fl_lay : int, optional
        Flow layer count used only for filename metadata. Defaults to `cfg.flow_num_layers`.
    save_plot : bool, optional
        If True, saves PNG with filename:
        `T2_error_vs_true_omega__T2_flowHidFeat{fl_hid_feat}_flowLay{fl_lay}_N{N}_tdis{t_disc}_std{sigma}_w{w_min}-{w_max}_seed{seed}.png`.
        Defaults to False.
    show_plot : bool, optional
        If True, displays the figure. Defaults to False.

    Returns -> None
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    err = np.abs(y_pred - y_true)

    sort_idx = np.argsort(y_true)
    y_true_sorted = y_true[sort_idx]
    err_sorted = err[sort_idx]

    n = len(y_true_sorted)
    window = max(5, int(smooth_window_frac * n))
    if window % 2 == 0:
        window += 1

    kernel = np.ones(window) / window
    x_smooth = np.convolve(y_true_sorted, kernel, mode="valid")
    err_smooth = np.convolve(err_sorted, kernel, mode="valid")

    plt.figure(figsize=(6, 5))
    plt.minorticks_on()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.scatter(y_true, err, s=12, alpha=0.4, label="single samples")
    plt.plot(x_smooth, err_smooth, "k-", linewidth=2.0, label="average |error|")

    plt.xlabel("Omega ω")
    plt.ylabel("|predicted ω - true ω|")
    plt.title("Error vs true ω")
    plt.legend()
    plt.tight_layout()

    if save_plot:
        path = cfg.plots_dir / f"T2_error_vs_true_omega__T2_flowHidFeat{fl_hid_feat}_flowLay{fl_lay}_N{N}_tdis{t_disc}_std{sigma}_w{w_min}-{w_max}_seed{seed}.png"
        plt.savefig(path, dpi=300)

    if show_plot:
        plt.show()

    plt.close()

@torch.no_grad()
def plot_uncertainty_vs_error(model: nn.Module, device, loader, num_samples: int=100, save_plot: bool=False, show_plot: bool=False) -> None:
    """
    Plot predictive uncertainty vs absolute error for a flow-based single-frequency model.

    For each input x:
    - draw `num_samples` samples ω ~ p(ω | x)
    - compute predictive mean and std over those samples
    - compute absolute error |mean(ω) - ω_true|
    Then scatter: std(ω) (x-axis) vs |mean-true| (y-axis).

    Parameters
    ----------
    model : torch.nn.Module
        A model implementing `sample(x, num_samples=...)` returning ω samples.
    device : torch.device
        Device used for running the model.
    loader : DataLoader
        DataLoader returning batches (xb, yb), where yb contains ω_true.
    num_samples : int, optional
        Number of Monte Carlo samples per input. Defaults to 100.
    save_plot : bool, optional
        If True, saves PNG into `cfg.plots_dir` with filename `uncertainty_vs_error_flow.png`.
        Defaults to False.
    show_plot : bool, optional
        If True, displays the figure. Defaults to False.

    Returns -> None

    Notes
    -----
    This plot is optional (not used in final versions). For error analysis vs ω, prefer `plot_error_vs_true_omega`.
    """
    model.eval()

    all_std = []
    all_err = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        samples = model.sample(xb, num_samples=num_samples)

        if samples.dim() == 3 and samples.shape[0] == num_samples:
            samples = samples.permute(1, 0, 2)

        samples = samples.squeeze(-1)

        mean_w = samples.mean(dim=1)
        std_w  = samples.std(dim=1)
        true_w = yb.squeeze(-1)

        err = (mean_w - true_w).abs()

        all_std.append(std_w.cpu().numpy())
        all_err.append(err.cpu().numpy())

    all_std = np.concatenate(all_std)
    all_err = np.concatenate(all_err)

    plt.figure(figsize=(6, 6))
    plt.minorticks_on()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.scatter(all_std, all_err, s=12, alpha=0.6)
    plt.xlabel("predictive std(w)")
    plt.ylabel("abs(mean(w) - true w)")
    plt.title("Uncertainty vs absolute error")
    plt.tight_layout()

    if save_plot:
        path = cfg.plots_dir / "uncertainty_vs_error_flow.png"
        plt.savefig(path, dpi=300)

    if show_plot:
        plt.show()

    plt.close()


#for two omegas
def plot_double_wave_sample_general(t, V_noisy, w1: float, w2: float|None=None, mu: float=cfg.mu, sigma: float=cfg.noise_std, signal: str="single", 
                save_plot: bool=False, show_plot: bool=False) -> None:
    """
    Plot one example signal for the two-frequency dataset family.

    This function overlays:
    - a dense "clean" curve built from the analytic formula (using a refined time grid)
    - the provided noisy samples V_noisy evaluated on the original grid t

    Parameters
    ----------
    t : array-like
        Time grid of shape (T,).
    V_noisy : array-like
        Noisy observed values on t, shape (T,). Expected to already include noise.
    w1 : float
        First frequency parameter used in the analytic clean signal.
    w2 : float or None, optional
        Second frequency parameter (required for signal types that use two frequencies).
        Ignored when `signal="single"`. Defaults to None.
    mu : float, optional
        Noise mean metadata used in the plot title/filename. Defaults to `cfg.mu`.
    sigma : float, optional
        Noise std metadata used in the plot title/filename. Defaults to `cfg.noise_std`.
    signal : str, optional
        Analytic signal type used to generate the clean curve. Supported values:
        - "single":  sin(w1 * t)
        - "linear":  sin(w1 * t) + sin(w2 * t)
        - "product" / "nonlinear": sin(w1 * t) + sin(w2 * t) + sin(w1 * t) * sin(w2 * t)
        Defaults to "single".
    save_plot : bool, optional
        If True, saves PNG into `cfg.plots_dir` with a filename depending on `signal`.
        Defaults to False.
    show_plot : bool, optional
        If True, displays the figure. Defaults to False.

    Returns -> None
    """

    t_clean = np.linspace(t.min(), t.max(), len(t) * 100)

    if signal == "linear":
        s1 = np.sin(w1 * t_clean)
        s2 = np.sin(w2 * t_clean)
        V_clean = s1 + s2
        ylabel = "sin(w1*t) + sin(w2*t)"
        title = f"Example (w1={w1:.3f}, w2={w2:.3f}, mu={mu}, sigma={sigma})"
        fname = f"double_sine_linear_mu{mu}_sigma{sigma}_w1{w1:.2f}_w2{w2:.2f}.png"
    elif signal in ("product", "nonlinear"):
        s1 = np.sin(w1 * t_clean)
        s2 = np.sin(w2 * t_clean)
        V_clean = s1 + s2 + s1 * s2
        ylabel = "sin(w1*t) + sin(w2*t) + sin(w1*t)*sin(w2*t)"
        title = f"Example (w1={w1:.3f}, w2={w2:.3f}, mu={mu}, sigma={sigma})"
        fname = f"double_sine_product_mu{mu}_sigma{sigma}_w1{w1:.2f}_w2{w2:.2f}.png"
    elif signal == "single":
        V_clean = np.sin(w1 * t_clean)
        ylabel = "sin(w1*t)"
        title = f"Example (w={w1:.3f}, mu={mu}, sigma={sigma})"
        fname = f"sine_single_mu{mu}_sigma{sigma}_w{w1:.2f}.png"
    else:
        raise ValueError("signal must be 'linear', 'product'/'nonlinear' or 'single'")

    plt.figure(figsize=(8, 6))

    plt.plot(t_clean, V_clean, label="clean mixture", linewidth=2)
    plt.scatter(t, V_noisy, s=15, alpha=0.7, label="noisy sample", c="black", marker="x")

    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.title(f"{title}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    if save_plot:
        plt.savefig(cfg.plots_dir / fname, dpi=300)

    if show_plot:
        plt.show()

    plt.close()

def plot_waves_clean_and_signal_points(i:int=0, wave_type: str="linear", noise: bool=False, save_plot: bool=False, show_plot: bool=False) -> None:
    """
    Plot one dataset example (clean curve + discrete signal points) for two-frequency settings.

    The function generates a dataset using the project’s dataset creation utilities and
    plots the i-th sample as:
    - dense clean curve (analytic)
    - discrete sampled points (with optional noise)

    Parameters
    ----------
    i : int, optional
        Index of the example to plot. Defaults to 0.
    wave_type : str, optional
        Signal type used for dataset generation (e.g. "linear", "product", ...).
        Defaults to "linear".
    noise : bool, optional
        If True, uses the noisy dataset variant; otherwise clean points. Defaults to False.
    save_plot : bool, optional
        If True, saves the figure into `cfg.plots_dir`. Defaults to False.
    show_plot : bool, optional
        If True, displays the figure. Defaults to False.

    Returns -> None
    """

    if wave_type == "linear":
        V_np, tar_np, t_np = make_double_sine_dataset(noise=noise)
        w1, w2 = tar_np[i]
    elif wave_type == "product":
        V_np, tar_np, t_np = make_double_sine_nonlinear_dataset(noise=noise)
        w1, w2 = tar_np[i]
    elif wave_type == "single":
        V_np, tar_np, t_np = make_sine_dataset(noise=noise)
        w1 = float(tar_np[i])
        w2 = None
    V_noisy = V_np[i]
    
    plot_double_wave_sample_general(t_np, V_noisy, w1=w1, w2=w2, signal=wave_type, save_plot=save_plot, show_plot=show_plot)


@torch.no_grad()
def plot_flow_posterior_double_example(model: nn.Module, device,loader, global_index: int=0, num_samples: int=100000, bins: int=100,
        num_sigmas: int=3, N: int=cfg.num_of_samples, t_disc: int=cfg.discr_of_time, w_min: float=cfg.omega_min, w_max: float=cfg.omega_max,
        seed=cfg.seed, sigma: float=cfg.noise_std, folder=cfg.plots_dir, fl_hid_feat: int=cfg.flow_hidden_features, fl_lay: int=cfg.flow_num_layers, 
        save_plot: bool=False, show_plot: bool=False) -> None:
    """
    Plot 1D marginal posteriors for one example in the two-frequency flow model.

    Workflow:
    - select one example by `global_index` from the loader
    - sort the true pair into (w_1_true, w_2_true) where w_1_true <= w_2_true
    - draw many samples from the model flow via `model.sample(x_one, num_samples=...)`
    - sort each sampled pair into (w_1, w_2)
    - plot two histograms side-by-side: posterior of w_1 and posterior of w_2,
      each with true value, sample mean μ, and ±kσ reference lines

    Parameters
    ----------
    model : torch.nn.Module
        A model implementing `sample(x, num_samples=...)` returning samples of shape (1, num_samples, 2)
        or compatible.
    device : torch.device
        Device used for inference.
    loader : DataLoader
        DataLoader returning (xb, yb), with yb containing true pairs (batch, 2).
    global_index : int, optional
        Global index into the dataset order produced by iterating `loader`. Defaults to 0.
    num_samples : int, optional
        Number of Monte Carlo samples drawn for the selected example. Defaults to 100000.
    bins : int, optional
        Number of histogram bins per marginal. Defaults to 100.
    num_sigmas : int, optional
        Number of ±kσ reference lines to draw. Defaults to 3.
    N : int, optional
        Dataset size used only for title/filename metadata. Defaults to `cfg.num_of_samples`.
    t_disc : int, optional
        Time discretization used only for title/filename metadata. Defaults to `cfg.discr_of_time`.
    w_min : float, optional
        Minimum omega used only for title/filename metadata. Defaults to `cfg.omega_min`.
    w_max : float, optional
        Maximum omega used only for title/filename metadata. Defaults to `cfg.omega_max`.
    seed : int, optional
        Seed identifier used only for filename metadata. Defaults to `cfg.seed`.
    folder : pathlib.Path or str, optional
        Output directory used when `save_plot=True`. Defaults to `cfg.plots_dir`.
    sigma : float, optional
        Noise std identifier used only for title/filename metadata. Defaults to `cfg.noise_std`.
    fl_hid_feat : int, optional
        Flow hidden feature count used only for filename metadata. Defaults to `cfg.flow_hidden_features`.
    fl_lay : int, optional
        Flow layer count used only for filename metadata. Defaults to `cfg.flow_num_layers`.
    save_plot : bool, optional
        If True, saves PNG with filename:
        `Probab_density_twofreq_idx{global_index}_FlowHid{fl_hid_feat}_FlowLay{fl_lay}_N{N}_tdis{t_disc}_std{sigma}_w{w_min}-{w_max}_seed{seed}.png`.
        Defaults to False.
    show_plot : bool, optional
        If True, displays the figure. Defaults to False.

    Returns -> None
    """

    model.eval()

    start = 0
    for xb, yb in loader:
        batch_size = xb.size(0)
        end = start + batch_size
        if global_index < end:
            local_idx = global_index - start

            xb = xb.to(device)
            yb = yb.to(device)

            x_one = xb[local_idx : local_idx + 1]
            w_true_pair = yb[local_idx]
            break
        start = end

    #sort the true pair into (w_1_true, w_2_true)
    w_true_np = torch.sort(w_true_pair).values.cpu().numpy()
    w_low_true, w_high_true = float(w_true_np[0]), float(w_true_np[1])

    #sample from flow, sort into (w_1, w_2)
    samples = model.sample(x_one, num_samples=num_samples)
    samples = samples.squeeze(0).cpu().numpy()

    if samples.ndim == 3:
        samples = samples[0]

    samples_sorted = np.sort(samples, axis=1)
    samples_1 = samples_sorted[:, 0]
    samples_2 = samples_sorted[:, 1]

    def plot_1d(ax, samples_1d, w_true, name):
        mu = samples_1d.mean()
        sigma = samples_1d.std()

        ax.minorticks_on()
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        ax.hist(samples_1d, bins=bins, density=True, color="tab:blue", alpha=0.8, label=f"flow samples {name} | x")
        ax.axvline(w_true, color="tab:orange", linestyle="--", linewidth=2, label=f"true {name} = {w_true:.3f}")
        ax.axvline(mu, color="tab:red", linestyle="-", linewidth=2, label=f"mean μ = {mu:.3f}")

        #±kσ lines
        sigma_label_added = False
        ks = range(1, num_sigmas + 1)
        for k in ks:
            left = mu - k * sigma
            right = mu + k * sigma
            label_sigma = r"±kσ lines" if not sigma_label_added else None
            sigma_label_added = True
            ax.axvline(left,  linestyle="-.", linewidth=1.8, color="tab:brown", alpha=0.7, label=label_sigma)
            ax.axvline(right, linestyle="-.", linewidth=1.8, color="tab:brown", alpha=0.7)

        span = max(5 * sigma, 0.02)
        ax.set_xlim(mu - span, mu + span)

        ax.set_xlabel(name)
        ax.set_ylabel("'samples'")
        ax.legend(fontsize=8)

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ticks = [mu + k * sigma for k in range(-num_sigmas, num_sigmas + 1)]
        tick_labels = []
        for k in range(-num_sigmas, num_sigmas + 1):
            if k == 0:
                tick_labels.append(r"μ")
            elif k < 0:
                tick_labels.append(rf"{k}σ")
            else:
                tick_labels.append(rf"+{k}σ")
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(tick_labels)
        ax2.tick_params(axis="x", labelsize=8, pad=2)

    fig, (ax_low, ax_high) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    plot_1d(ax_low,  samples_1,  w_low_true,  "w_1")
    plot_1d(ax_high, samples_2, w_high_true, "w_2")

    fig.suptitle(f"Probability for one example of ω_1 and ω_2 \nN={N}, ω=[{w_min}-{w_max}], tdis={t_disc}, std={sigma}", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_plot:
        plt.savefig(folder/ f"Probab_density_twofreq_idx{global_index}_FlowHid{fl_hid_feat}_FlowLay{fl_lay}_N{N}_tdis{t_disc}_std{sigma}_w{w_min}-{w_max}_seed{seed}.png", dpi=300)

    if show_plot:
        plt.show()

    plt.close(fig)

def plot_pred_vs_true_double(y_true, y_pred, test_mse, test_mae, N: int=cfg.num_of_samples, t_disc: int=cfg.discr_of_time, w_min: float=cfg.omega_min,
            w_max: float=cfg.omega_max, seed=cfg.seed, sigma: float=cfg.noise_std, folder=cfg.plots_dir, save_plot: bool=False, show_plot: bool=False) -> None:
    """
    Scatter plot of predicted vs true frequency pairs for the two-frequency task.

    The function sorts both y_true and y_pred row-wise into (w_1, w_2) with w_1 <= w_2
    and produces two scatter plots:
    - predicted w_1 vs true w_1
    - predicted w_2 vs true w_2
    Each subplot includes a y=x reference line.

    Parameters
    ----------
    y_true : torch.Tensor or np.ndarray
        True frequency pairs, shape (N, 2).
    y_pred : torch.Tensor or np.ndarray
        Predicted frequency pairs, shape (N, 2).
    test_mse : float
        Test mean squared error displayed in the title.
    test_mae : float
        Test mean absolute error displayed in the title.
    N : int, optional
        Dataset size used only for title/filename metadata. Defaults to `cfg.num_of_samples`.
    t_disc : int, optional
        Time discretization used only for title/filename metadata. Defaults to `cfg.discr_of_time`.
    w_min : float, optional
        Minimum omega used only for title/filename metadata. Defaults to `cfg.omega_min`.
    w_max : float, optional
        Maximum omega used only for title/filename metadata. Defaults to `cfg.omega_max`.
    seed : int, optional
        Seed identifier used only for filename metadata. Defaults to `cfg.seed`.
    sigma : float, optional
        Noise std identifier used only for title/filename metadata. Defaults to `cfg.noise_std`.
    folder : pathlib.Path or str, optional
        Output directory used when `save_plot=True`. Defaults to `cfg.plots_dir`.
    save_plot : bool, optional
        If True, saves PNG into `folder` (uses your current naming convention).
        Defaults to False.
    show_plot : bool, optional
        If True, displays the figure. Defaults to False.
    
    Returns -> None
    """

    if hasattr(y_true, "detach"):
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
    else:
        y_true_np = np.asarray(y_true)
        y_pred_np = np.asarray(y_pred)

    #sorting each pair by value
    y_true_np = np.sort(y_true_np, axis=1)
    y_pred_np = np.sort(y_pred_np, axis=1)

    w1_true, w1_pred = y_true_np[:, 0], y_pred_np[:, 0]
    w2_true, w2_pred = y_true_np[:, 1], y_pred_np[:, 1]

    mae_w1 = np.mean(np.abs(w1_pred - w1_true))
    mae_w2 = np.mean(np.abs(w2_pred - w2_true))

    mn = min(y_true_np.min(), y_pred_np.min())
    mx = max(y_true_np.max(), y_pred_np.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    ax1, ax2 = axes

    # small internal helper so both panels look the same
    def scatter_panel(ax, x_true, y_pred, label, mae):
        ax.scatter(x_true, y_pred, s=14, alpha=0.6)
        ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)
        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(f"{label} – MAE={mae:.4f}")
        ax.grid(True, which="both")

    scatter_panel(ax1, w1_true, w1_pred, "w_1", mae_w1)
    scatter_panel(ax2, w2_true, w2_pred, "w_2", mae_w2)

    fig.suptitle(
        f"Double-sine – Test N={N}, w=[{w_min}-{w_max}], tdis={t_disc}\n"
        f"Overall MSE={test_mse:.6f}, MAE={test_mae:.6f}, std={sigma}", fontsize=11)

    try:
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    except Exception:
        pass

    if save_plot:
        fig.savefig(folder / f"T3_double_sorted_N{N}_tdis{t_disc}_w{w_min}-{w_max}_seed{seed}_std{sigma}_PREDvsREAL2.png", dpi=300)

    if show_plot:
        plt.show()

    plt.close(fig)

def plot_freq_space_true_vs_pred(y_true, y_pred, test_mse, test_mae, N: int = cfg.num_of_samples, t_disc: int = cfg.discr_of_time, w_min: float = cfg.omega_min,
    w_max: float = cfg.omega_max, seed=cfg.seed, sigma: float = cfg.noise_std, folder=cfg.plots_dir, save_plot: bool = False, show_plot: bool = False) -> None:
    """
    Plot both true and predicted (w_1, w_2) pairs in the same frequency-space figure.

    The function sorts pairs row-wise into (w_1, w_2) with w_1 <= w_2, then plots:
    - true points in (w_1, w_2) plane
    - predicted points in (w_1, w_2) plane

    Parameters
    ----------
    y_true : torch.Tensor or np.ndarray
        True frequency pairs, shape (N, 2).
    y_pred : torch.Tensor or np.ndarray
        Predicted frequency pairs, shape (N, 2).
    test_mse : float
        Test mean squared error displayed in the title.
    test_mae : float
        Test mean absolute error displayed in the title.
    N : int, optional
        Dataset size used only for title/filename metadata. Defaults to `cfg.num_of_samples`.
    t_disc : int, optional
        Time discretization used only for title/filename metadata. Defaults to `cfg.discr_of_time`.
    w_min : float, optional
        Minimum omega used only for title/filename metadata. Defaults to `cfg.omega_min`.
    w_max : float, optional
        Maximum omega used only for title/filename metadata. Defaults to `cfg.omega_max`.
    seed : int, optional
        Seed identifier used only for filename metadata. Defaults to `cfg.seed`.
    sigma : float, optional
        Noise std identifier used only for title/filename metadata. Defaults to `cfg.noise_std`.
    folder : pathlib.Path or str, optional
        Output directory used when `save_plot=True`. Defaults to `cfg.plots_dir`.
    save_plot : bool, optional
        If True, saves PNG into `folder` (uses your current naming convention).
        Defaults to False.
    show_plot : bool, optional
        If True, displays the figure. Defaults to False.
    
    Returns -> None
    """

    if hasattr(y_true, "detach"):
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
    else:
        y_true_np = np.asarray(y_true)
        y_pred_np = np.asarray(y_pred)

    #sort each pair by value, not by error
    y_true_np = np.sort(y_true_np, axis=1)
    y_pred_np = np.sort(y_pred_np, axis=1)

    w1_true, w1_pred = y_true_np[:, 0], y_pred_np[:, 0]  # w_1
    w2_true, w2_pred = y_true_np[:, 1], y_pred_np[:, 1]  # w_2

    #component-wise MAE
    mae_w1 = np.mean(np.abs(w1_pred - w1_true))
    mae_w2 = np.mean(np.abs(w2_pred - w2_true))

    mn = min(y_true_np.min(), y_pred_np.min())
    mx = max(y_true_np.max(), y_pred_np.max())

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.scatter(w1_true, w2_true, s=18, alpha=0.6, label="True", color="tab:blue")
    ax.scatter(w1_pred, w2_pred, s=18, alpha=0.6, label="Pred", color="tab:orange")

    ax.set_xlabel("w_1")
    ax.set_ylabel("w_2")
    ax.set_title(f"Frequency space (MAE w1={mae_w1:.4f}, w2={mae_w2:.4f})")
    ax.grid(True, which="both")
    ax.set_xlim(w_min, w_max)
    ax.set_ylim(w_min, w_max)
    ax.set_aspect("equal", "box")
    ax.legend()

    fig.suptitle(f"Double-sine – Test N={N}, w=[{w_min}-{w_max}], tdis={t_disc}\nOverall MSE={test_mse:.6f}, MAE={test_mae:.6f}, std={sigma}", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])

    if save_plot:
        fig.savefig(folder / f"T3_double_frequencyspace_N{N}_tdis{t_disc}_w{w_min}-{w_max}_seed{seed}_std{sigma}_PREDvsREAL.png", dpi=300)

    if show_plot:
        plt.show()

    plt.close(fig)

#Contour visualizations for 2 omegas
def plot_analytic_contours_sin2(t0: float=1.0, w_min: float=cfg.omega_min, w_max: float=cfg.omega_max, n_points: int=1000, levels=None, save_plot: bool=False,
        show_plot: bool=True, folder=cfg.plots_dir, triple_t: bool=False, dt: float=0.1, n_levels: int=40, cmap: str="viridis", signal: str="linear", 
        fullcolor: bool=True, fixed_range: bool=True) -> None:
    """
    Plot analytic contour maps in (w1, w2) space for several synthetic signal formulas.

    The function evaluates an analytic function F(w1, w2; t) on a grid:
    w1 in [w_min, w_max], w2 in [w_min, w_max], with `n_points` resolution.

    Supported analytic signals (controlled by `signal`):
    - "linear":            F = sin(t*w1) + sin(t*w2)
    - "product":           F = sin(t*w1) + sin(t*w2) + sin(t*w1)*sin(t*w2)
    - "nonlinear_sq":      F = sin(t*w1^2) + sin(t*w2^2) + sin(t*w1^2)*sin(t*w2^2)
    - "nonlinear_sinprod": F = sin(t*w1) + sin(t*w2) + sin(t*w1*w2)

    Parameters
    ----------
    t0 : float, optional
        Base time value used in the analytic formula. Defaults to 1.0.
    w_min : float, optional
        Minimum frequency for grid. Defaults to `cfg.omega_min`.
    w_max : float, optional
        Maximum frequency for grid. Defaults to `cfg.omega_max`.
    n_points : int, optional
        Number of grid points along each axis (w1 and w2). Defaults to 1000.
    levels : array-like or None, optional
        Explicit contour levels. If None, levels are generated using `n_levels`.
        Defaults to None.
    save_plot : bool, optional
        If True, saves the figure into `folder`. Defaults to False.
    show_plot : bool, optional
        If True, displays the figure. Defaults to True.
    folder : pathlib.Path or str, optional
        Output directory used when saving. Defaults to `cfg.plots_dir`.
    triple_t : bool, optional
        If True, plots contours for three time values: [t0-dt, t0, t0+dt].
        If False, plots only at t0. Defaults to False.
    dt : float, optional
        Time offset used when `triple_t=True`. Defaults to 0.1.
    n_levels : int, optional
        Number of automatically generated contour levels when `levels is None`.
        Defaults to 40.
    cmap : str, optional
        Matplotlib colormap name used for filled contour. Defaults to "viridis".
    signal : str, optional
        Analytic formula selector. Must be one of:
        {"linear", "product", "nonlinear_sq", "nonlinear_sinprod"}.
        Defaults to "linear".
    fullcolor : bool, optional
        If True, uses `contourf` (filled). If False, uses line contours only.
        Defaults to True.
    fixed_range : bool, optional
        If True, uses a fixed theoretical value range per `signal`.
        If False, uses min/max of the computed grid at the first time value.
        Defaults to True.
    
    Returns -> None
    """

    w1 = np.linspace(w_min, w_max, n_points)
    w2 = np.linspace(w_min, w_max, n_points)
    W1, W2 = np.meshgrid(w1, w2, indexing="xy")

    def compute_F(tt):
        if signal == "linear":
            return np.sin(tt * W1) + np.sin(tt * W2), "F(w1,w2)=sin(t·w1)+sin(t·w2)", (-2.0, 2.0)

        if signal == "product":
            F = np.sin(tt * W1) + np.sin(tt * W2) + np.sin(tt * W1) * np.sin(tt * W2)
            return F, "F=sin(t·w1)+sin(t·w2)+sin(t·w1)·sin(t·w2)", (-3.0, 3.0)

        if signal == "nonlinear_sq":
            F = np.sin(tt * (W1**2)) + np.sin(tt * (W2**2)) + np.sin(tt * (W1**2)) * np.sin(tt * (W2**2))
            return F, "F=sin(t·w1²)+sin(t·w2²)+sin(t·w1²)·sin(t·w2²)", (-3.0, 3.0)

        if signal == "nonlinear_sinprod":
            F = np.sin(tt * W1) + np.sin(tt * W2) + np.sin(tt * (W1 * W2))
            return F, "F=sin(t·w1)+sin(t·w2)+sin(t·w1·w2)", (-3.0, 3.0)

        raise ValueError(f"Unknown signal='{signal}'. Try: linear, product, nonlinear_sq, nonlinear_sinprod")

    t_list = [t0]
    if triple_t:
        t_list = [t0 - dt, t0, t0 + dt]

    ncols = len(t_list)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6), sharex=True, sharey=True)
    if ncols == 1:
        axes = [axes]

    _, suptitle, (theo_vmin, theo_vmax) = compute_F(t_list[0])

    if fixed_range:
        vmin, vmax = theo_vmin, theo_vmax
    else:
        F0, _, _ = compute_F(t_list[0])
        vmin, vmax = float(np.min(F0)), float(np.max(F0))

    if levels is None:
        levels = np.linspace(vmin, vmax, n_levels)

    last_cf = None
    for ax, tt in zip(axes, t_list):
        F, _, _ = compute_F(tt)

        if fullcolor:
            last_cf = ax.contourf(W1, W2, F, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        elif not fullcolor:
            last_cf = ax.contour(W1, W2, F, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"t= {tt:.4g}")
        ax.set_xlabel("w1")
        ax.set_ylabel("w2")
        ax.set_xlim(w_min, w_max)
        ax.set_ylim(w_min, w_max)
        ax.set_aspect("equal", "box")

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.015, 0.70])
    cbar = fig.colorbar(last_cf, cax=cax)
    cbar.set_label("F(w1,w2)")
    cbar.set_ticks(np.linspace(vmin, vmax, 9))

    fig.suptitle(f"Contours: {suptitle}", y=1.02)

    if save_plot:
        suffix = f"_triple_dt{dt}" if triple_t else ""
        fig.savefig(folder / f"analytic_contour_{signal}_t{t0:.2f}{suffix}_w{w_min}-{w_max}.png", dpi=300)

    if show_plot:
        plt.show()

    plt.close(fig)
