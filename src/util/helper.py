import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sympy import lambdify
import joblib
from joblib import Parallel, delayed
from sklearn.neighbors import KernelDensity
from itertools import combinations_with_replacement
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split


def reverse_range(y):
    y_min = np.min(y)
    y_max = np.max(y)
    y_reversed = y_min + y_max - y
    return y_reversed


def normalize_data_point(data_point, min_vals, max_vals):
    data_point = np.clip(data_point, min_vals, max_vals)
    normalized_data_point = 2 * (data_point - min_vals) / (max_vals - min_vals) - 1
    return normalized_data_point


def save_array_plot(array, filename):
    plt.figure(figsize=(16, 8))
    plt.plot(array, ".")
    plt.savefig(filename)


def filter_outliers(mu_tau, logP_tau, remove_outliers, last=True):
    if remove_outliers:
        if last:
            sorted_indices_x = np.argsort(logP_tau)[:-20]
        else:
            sorted_indices_x = np.argsort(logP_tau)[20:]
        sorted_logPtau = logP_tau[sorted_indices_x]
        sorted_mu_tau = mu_tau[sorted_indices_x, :]

        # shuffle sorted arrays
        indices = np.arange(len(sorted_logPtau))
        np.random.shuffle(indices)
        sorted_mu_tau = sorted_mu_tau[indices]
        sorted_logPtau = sorted_logPtau[indices]
        return np.array(sorted_mu_tau), np.array(sorted_logPtau).squeeze()
    return mu_tau, logP_tau


def normalize_trajs_std(trajs, norm_type="none"):
    stacked_trajs = np.vstack(trajs)
    if norm_type == "none":
        normalized_trajs = stacked_trajs
    elif norm_type == "std":
        sc = StandardScaler()
        stacked_trajs = sc.fit_transform(stacked_trajs)
        normalized_trajs = stacked_trajs
    elif norm_type == "max":
        sc = MinMaxScaler((-1, 1))
        stacked_trajs = sc.fit_transform(stacked_trajs)
        normalized_trajs = stacked_trajs
    else:
        raise ValueError(f"Invalid norm_type: {norm_type}")

    # Reshape normalized data back into the original list structure
    normalized_trajs_list = []
    start_idx = 0
    for traj in trajs:
        end_idx = start_idx + len(traj)
        normalized_trajs_list.append(normalized_trajs[start_idx:end_idx])
        start_idx = end_idx
    return normalized_trajs_list, None, None


def load_data(env_name, normalize, num_chunks, gamma):

    data_path = f"data/{env_name}/"
    expert_trajs = np.load(data_path + "expert_trajs.npy")
    expert_ts = np.load(data_path + "expert_ts.npy")
    expert_rs = np.load(data_path + "expert_rs.npy")

    non_expert_trajs = np.load(data_path + "non_expert_trajs.npy")
    non_expert_ts = np.load(data_path + "non_expert_ts.npy")

    traj_chunks = divide_into_chunks(expert_trajs, expert_ts, num_chunks)
    reward_chunks = divide_1d_array_into_chunks(expert_rs, expert_ts, num_chunks)

    non_expert_chunks = divide_into_chunks(non_expert_trajs, non_expert_ts, num_chunks)

    rewards = []
    for ep in reward_chunks:
        discounted_reward = 0
        for t in reversed(range(len(ep))):
            discounted_reward = ep[t] + gamma * discounted_reward
        rewards.append(discounted_reward)

    return traj_chunks, rewards, expert_ts, non_expert_chunks, non_expert_ts


def divide_1d_array_into_chunks(data_array, time_indexes, num_chunks):
    """
    Divide a 1D array into chunks based on sequential time indexes.

    Parameters:
    - data_array: The 1D array of shape (N,).
    - time_indexes: The array representing sequential time indexes for each point.

    Returns:
    - chunks: A list of chunks, where each chunk is a 1D array.
    """
    chunks = []
    start_index = 0
    counter = 0
    for i in range(1, len(time_indexes)):
        if time_indexes[i] < time_indexes[i - 1]:
            # If the time index resets, create a new chunk
            chunks.append(data_array[start_index:i])
            counter += 1
            if counter == num_chunks:
                break
            start_index = i

    # Add the last chunk
    if counter < num_chunks:
        chunks.append(data_array[start_index:])
    return chunks


def divide_into_chunks(data_array, time_indexes, num_chunks):
    """
    Divide a 2D array into chunks based on sequential time indexes.

    Parameters:
    - data_array: The 2D array of shape (N, d).
    - time_indexes: The array representing sequential time indexes for each point.

    Returns:
    - chunks: A list of chunks, where each chunk is a 2D array.
    """
    chunks = []
    start_index = 0
    counter = 0

    for i in range(1, len(time_indexes)):
        if time_indexes[i] < time_indexes[i - 1]:
            # If the time index resets, create a new chunk
            chunks.append(data_array[start_index:i, :])
            start_index = i
            counter += 1
            if counter == num_chunks:
                break
    return chunks


def create_succ_points(trajs):
    result = []
    indices = []
    for traj in trajs:
        index = 0
        for i in range(len(traj) - 1):
            consecutive_points = np.concatenate((traj[i], traj[i + 1]), axis=0)
            result.append(consecutive_points)
            indices.append(index)
            index += 1
    return np.array(result), np.array(indices)


def find_feature_trajs(num_feats, feats, gamma):
    feats_trajs = np.zeros((len(feats), num_feats))
    for k, traj in enumerate(feats):
        for i, feat in enumerate(traj):
            feat_discounted = feat * (gamma**i)
            feats_trajs[k] += feat_discounted
    return feats_trajs


def compute_prob_marginal(n_batches, trajs, expert_ts, num_chunks, kde_states):
    data = np.vstack(trajs)
    n_jobs = -1  # Use all cores
    batch_size = int(np.ceil(len(data) / n_batches))
    results = Parallel(n_jobs=n_jobs)(
        delayed(score_subset)(kde_states, data[i : i + batch_size])
        for i in range(0, len(data), batch_size)
    )
    log_ps = np.concatenate(results)
    logP_taui = divide_1d_array_into_chunks(log_ps, expert_ts, num_chunks)
    logP_tau = [traj.sum() for traj in logP_taui]
    return np.array(logP_tau)


def compute_prob_seq(logP_tau, n_batches, trajs, num_chunks, kde_succ_states):
    data, time_ts = create_succ_points(trajs)
    n_jobs = -1  # Use all available cores
    batch_size = int(np.ceil(len(data) / n_batches))  # For example, 10 batches
    results = Parallel(n_jobs=n_jobs)(
        delayed(score_subset)(kde_succ_states, data[i : i + batch_size])
        for i in range(0, len(data), batch_size)
    )
    log_pss1 = np.concatenate(results)
    logP_taui = divide_1d_array_into_chunks(log_pss1, time_ts, num_chunks)
    logP_tau2 = [traj.sum() for traj in logP_taui]
    logP_tau2 = np.array(logP_tau2)
    return logP_tau2 - logP_tau


def fit_states(trajs):
    data = np.vstack(trajs)
    kde = KernelDensity(kernel="gaussian", bandwidth="silverman").fit(data)
    # kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(data)
    return kde


def fit_succ_states(trajs):
    data, _ = create_succ_points(trajs)
    kde = KernelDensity(kernel="gaussian", bandwidth="silverman").fit(data)
    # kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(data)
    return kde


def fit_kde(trajs):
    kde_states = fit_states(trajs)
    kde_succ_states = fit_succ_states(trajs)
    return kde_states, kde_succ_states


def score_subset(kde, subset):
    return kde.score_samples(subset)


def compute_log_probabilities(
    trajs, expert_ts, num_chunks, kde_states, kde_succ_states, use_marginal
):
    n_batches = 50
    logP_tau = compute_prob_marginal(
        n_batches, trajs, expert_ts, num_chunks, kde_states
    )
    if not use_marginal:
        logP_tau = compute_prob_seq(
            logP_tau, n_batches, trajs, num_chunks, kde_succ_states
        )
    return logP_tau


def create_all_symbolic_basis_functions(dims=3, means=None, stds=None):
    variables = sp.symbols([f"f_{i}" for i in range(dims)])
    basis_functions = []
    # Add first, second, and third-degree polynomial combinations
    basis_functions += [var for var in variables]
    basis_functions += [x * y for x, y in combinations_with_replacement(variables, 2)]
    basis_functions += [
        x * y * z for x, y, z in combinations_with_replacement(variables, 3)
    ]
    print(f"Total number of unique basis functions: {len(basis_functions)}")
    return basis_functions, variables


def select_symbolic_basis_functions(all_basis_functions, indices):
    selected_basis = [all_basis_functions[i] for i in indices]
    return selected_basis


def compute_numerical_values(selected_basis, variables, x_values):
    substitution_dict = {variables[i]: x_values[i] for i in range(len(variables))}
    evaluated_basis = [func.subs(substitution_dict) for func in selected_basis]
    return evaluated_basis


def compute_with_precompiled_functions(precompiled_functions, x_values):
    evaluated_values = [func(*x_values) for func in precompiled_functions]
    return evaluated_values


def precompile_basis_functions(selected_basis, variables):
    precompiled_functions = [
        lambdify(variables, func, "numpy") for func in selected_basis
    ]
    return precompiled_functions


def compute_numerical_values_for_all_trajs_optimized(selected_basis, variables, trajs):
    precompiled_functions = precompile_basis_functions(selected_basis, variables)

    feats = []
    for traj in trajs:
        num_samples = traj.shape[0]
        num_features = len(precompiled_functions)
        feat = np.zeros((num_samples, num_features))
        for i, func in enumerate(precompiled_functions):
            feat[:, i] = func(*traj.T)
        feats.append(feat)
    return feats


def compute_numerical_values_for_all_trajs(selected_basis, variables, trajs):
    feats = []
    for traj in trajs:
        num_samples = traj.shape[0]
        num_features = len(selected_basis)
        feat = np.zeros((num_samples, num_features))

        for idx, curr_point in enumerate(traj):
            # Create a substitution dictionary for the current point
            substitution_dict = {
                variables[i]: curr_point[i] for i in range(len(variables))
            }
            feat_values = [func.subs(substitution_dict) for func in selected_basis]
            feat[idx, :] = np.array(feat_values, dtype=float)
        feats.append(feat)
    return feats


def load_and_precompile_functions_joblib(filepath):
    data = joblib.load(filepath)
    selected_basis = data["basis"]
    variables = data["variables"]
    precompiled_functions = [
        lambdify(variables, func, "numpy") for func in selected_basis
    ]
    return precompiled_functions


def save_symbolic_basis_functions_joblib(
    selected_basis, variables, filepath="symbolic_expressions.joblib"
):
    data = {"basis": selected_basis, "variables": variables}
    joblib.dump(data, filepath)
    print(f"Saved symbolic expressions for reward to -- {filepath}")


def plot_regression_results(X, y, ind_top, verbose, folder_path):
    # CREATE LABELS
    X = np.array(X)
    y = np.array(y).squeeze()

    regr = linear_model.LassoCV()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 / 5, random_state=85, shuffle=False
    )
    selected_train = X_train[:, ind_top].astype(np.float32)
    selected_test = X_test[:, ind_top].astype(np.float32)

    regr.fit(selected_train, y_train.astype(np.float32))
    y_pred_train = regr.predict(selected_train)
    y_pred_test = regr.predict(selected_test)

    if verbose:
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].set_title("Train data - R score: %.2f" % r2_score(y_train, y_pred_train))
        ax[1].set_title("Test data - R score: %.2f" % r2_score(y_test, y_pred_test))

        sorted_indices_train = np.argsort(y_train)
        sorted_indices_test = np.argsort(y_test)

        ax[0].plot(y_train[sorted_indices_train], "o", color="black", label="y_train")
        ax[0].plot(
            y_pred_train[sorted_indices_train], color="green", label="y_pred_train"
        )
        ax[0].legend()

        ax[1].plot(y_test[sorted_indices_test], "o", color="black", label="y_test")
        ax[1].plot(y_pred_test[sorted_indices_test], color="green", label="y_pred_test")
        ax[1].legend()
        # plt.savefig(f'{folder_path}/regression_results.png')
        plt.show()

    correlations = np.corrcoef(selected_train, y_train, rowvar=False)[-1, :-1]
    signs = np.sign(correlations)
    print("    Correlations of features with the target:", signs)

    return r2_score(y_train, y_pred_train), r2_score(y_test, y_pred_test)


def save_selected_basis_functions(selected_basis, all_variables, env_name):
    filepath = f"tmp/{env_name}/{env_name}_basis.joblib"
    save_symbolic_basis_functions_joblib(
        selected_basis, all_variables, filepath=filepath
    )
