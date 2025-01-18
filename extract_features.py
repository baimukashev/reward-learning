import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression
from sklearn.model_selection import KFold
from src.util.helper import (
    save_array_plot,
    save_selected_basis_functions,
    plot_regression_results,
    load_data,
    normalize_trajs_std,
    fit_kde,
    compute_log_probabilities,
    create_all_symbolic_basis_functions,
    compute_numerical_values_for_all_trajs_optimized,
    find_feature_trajs,
    filter_outliers,
)


def select_top_features(
    X: np.ndarray,
    y: np.ndarray,
    Xe: np.ndarray,
    ye: np.ndarray,
    all_variables: list,
    feats_names: list,
    score_threshold: float = 0.7,
    n_selected: int = 20,
    env_name: str = None,
    save: bool = None,
    verbose: bool = False,
    folder_path: str = None,
):
    """
    Select top features based on an F-test with cross-validation.

    1) We split X (and y) with KFold (n_splits=10).
    2) Perform F-test (f_regression) on the training split.
    3) Normalize absolute F-values, pick features above 'score_threshold'.
    4) Collect feature indices across folds -> pick the most frequent ones.

    :param X: (N, d) entire data.
    :param y: (N,) entire target/log-prob array.
    :param Xe: (Ne, d) expert-only data (used for final regression tests).
    :param ye: (Ne,) expert-only target array.
    :param all_variables: List of symbolic variables of features .
    :param feats_names: List of sympy expressions or string names.
    :param score_threshold: Normalized threshold for F-values
    :param n_selected: Number of top features to keep after cross-validation.
    :param env_name: Name of environment for saving.
    :param save: If True, save the selected basis functions to file.
    :param verbose: If True, show extra logs and a regression result plot.
    :param folder_path: Directory path for saving plots or data.

    :return: list of indices of the top features in feats_names.
    """
    original_feats_names = feats_names.copy()
    num_feats = len(feats_names)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    selected_features = []

    # Cross-validation to accumulate top features
    for train_index, test_index in kf.split(X):
        X_train, _ = X[train_index], X[test_index]
        y_train, _ = y[train_index], y[test_index]

        f_corr_cv, _ = f_regression(X_train, y_train)
        f_corr_cv_orig = f_corr_cv.copy()
        f_corr_cv = np.abs(f_corr_cv)
        f_corr_cv = (f_corr_cv / np.max(f_corr_cv)).astype(float)

        ind_top_cv = [i for i in range(num_feats) if f_corr_cv[i] > score_threshold]
        selected_features.extend(ind_top_cv)

    # Count how often each feature index was selected across folds
    selected_features_counts = pd.Series(selected_features).value_counts()

    # Take the top 'n_selected' indices
    top_features_indices = selected_features_counts.head(n_selected).index.to_list()
    important_features = [feats_names[i] for i in top_features_indices]

    sign_importance = [f_corr_cv_orig[i] for i in top_features_indices]
    print("Top important features based on F-test and cross-validation:")
    norm_factor = np.max(f_corr_cv_orig) if np.max(f_corr_cv_orig) != 0 else 1.0
    for feature, i_idx, s_val in zip(important_features, top_features_indices, sign_importance):
        print(i_idx, feature, s_val / norm_factor)

    # Re-map to original indexing if feats_names got modified
    ind_top_original = [original_feats_names.index(feat) for feat in important_features]
    selected_basis = [original_feats_names[i] for i in ind_top_original]
    
    # Optionally save the chosen features
    if save:
        print(f"\nSelected basis functions: {selected_basis}")
        save_selected_basis_functions(selected_basis, all_variables, env_name)

    # (Optional) Evaluate how well these features can linearly predict y
    if verbose:
        r2_train, r2_test = plot_regression_results(Xe, ye, ind_top_original, verbose, folder_path)
        print(f"\n({len(top_features_indices)}) / ({len(feats_names)}) features "
              f"- R2 train/test: {r2_train:.3f}/{r2_test:.3f}")

    return top_features_indices


def extract_features():
    """Extract features used as components for the reward model."""
    for (
        env_name,
        dims,
        score_threshold,
        remove_outliers,
        n_selected,
        num_chunks,
        add_non_expert,
        crop_last,
        n_agent,
    ) in [
        ("HalfCheetah-v4", 17, 0.6, False, 12, 40, False, False, 50),  
        # ("Walker2d-v4", 17, 0.3, True, 16, 150, True, True, 50),
        # ("Ant-v4", 27, 0.6, True, 20, 150, True, False, 20),
        # ("Hopper-v4", 11, 0.6, False, 10, 99, True, False, 20),
    ]:
        np.random.seed(42)

        data_path = "final"
        use_marginal = True
        norm_type = "max"  # 'std', 'max', 'none'
        normalize = True
        verbose = False
        save = True
        gamma = 1

        print(f"\n----------Extracting features for ----{env_name}")
        folder_path = f"tmp/{env_name}/"
        os.makedirs(folder_path, exist_ok=True)
        
        trajs, rewards, expert_ts, non_expert_trajs, non_expert_ts = load_data(
            env_name, normalize, num_chunks, gamma
        )
        trajs, means, stds = normalize_trajs_std(trajs, norm_type=norm_type)
        print("Trajs size: ", len(trajs))
        non_expert_trajs, _, _ = normalize_trajs_std(non_expert_trajs, norm_type=norm_type)

        # load or compute logP_tau
        try:
            logP_tau = np.load(f"tmp/{env_name}/logP_tau_{data_path}_{num_chunks}.npy")
            print("Loaded logP_tau from file")
        except Exception as e:
            print("...Computing logP_tau")
            kde_states, kde_succ_states = fit_kde(trajs)
            logP_tau = compute_log_probabilities(
                trajs,
                expert_ts,
                num_chunks,
                kde_states,
                kde_succ_states,
                use_marginal=use_marginal,
            )
            np.save(f"tmp/{env_name}/logP_tau_{data_path}_{num_chunks}.npy", logP_tau)
        
        # create symbolic basis functions
        all_basis_functions, all_variables = create_all_symbolic_basis_functions(
            dims=dims,
            means=means,
            stds=stds,
        )
        feats = compute_numerical_values_for_all_trajs_optimized(
            all_basis_functions, all_variables, trajs
        )
        
        if verbose:
            save_array_plot(logP_tau, f"{folder_path}/logP_tau.png")
        
        mu_tau = find_feature_trajs(len(all_basis_functions), feats, gamma=gamma)
        mu_tau_expert, logP_tau_expert = filter_outliers(
            mu_tau, logP_tau, remove_outliers, crop_last
        )
        if verbose:
            save_array_plot(logP_tau_expert, f"{folder_path}/logP_tau_sorted.png")

        if add_non_expert:
            feats = compute_numerical_values_for_all_trajs_optimized(
                all_basis_functions, all_variables, non_expert_trajs
            )
            mu_tau_agent = find_feature_trajs(
                len(all_basis_functions), feats, gamma=gamma
            )
            # add non-expert data
            logP_tau_agent = np.random.uniform(
                min(logP_tau_expert), min(logP_tau_expert) * (1.1), n_agent
            )
            mu_tau_agent = mu_tau_agent[: len(logP_tau_agent)]
            mu_tau_combined = np.vstack([mu_tau_expert, mu_tau_agent])
            logP_tau_combined = np.hstack([logP_tau_expert, logP_tau_agent])
        else:
            mu_tau_combined = mu_tau_expert
            logP_tau_combined = logP_tau_expert

        select_top_features(
            mu_tau_combined,
            logP_tau_combined,
            mu_tau_expert,
            logP_tau_expert,
            all_variables,
            all_basis_functions,
            score_threshold,
            n_selected=n_selected,
            env_name=env_name,
            save=save,
            verbose=verbose,
            folder_path=None,
        )

if __name__ == "__main__":
    extract_features()