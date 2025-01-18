import numpy as np
from sklearn.preprocessing import Normalizer
from .helper import (
    normalize_data_point,
    load_and_precompile_functions_joblib,
    compute_with_precompiled_functions,
)
from .data_config import ENVIRONMENT_CONFIGS

PRECOMPILED_FUNC = None


def select_feat_extractor(env_name, states, cfg):
    if env_name == "Walker2d-v4":
        feature_expectations = walker_feat_extract(states, cfg)
    elif env_name == "HalfCheetah-v4":
        feature_expectations = cheetah_feat_extract(states, cfg)
    elif env_name == "Hopper-v4":
        feature_expectations = hopper_feat_extract(states, cfg)
    elif env_name == "Ant-v4":
        feature_expectations = ant_feat_extract(states, cfg)
    else:
        raise NotImplementedError
    return feature_expectations


def cheetah_feat_extract(states, cfg):
    global PRECOMPILED_FUNC
    env_name = "HalfCheetah-v4"
    min_vals = ENVIRONMENT_CONFIGS[env_name]["min_vals"]
    max_vals = ENVIRONMENT_CONFIGS[env_name]["max_vals"]
    states = normalize_data_point(states, min_vals, max_vals)
    use_norm = cfg["normalize_feats"]
    if use_norm:
        states = Normalizer().fit_transform(states.reshape(1, -1)).squeeze()

    feats = []
    feat_selection = cfg["feats_method"]

    if feat_selection == "first":
        feats = states[:17]

    elif feat_selection == "random":
        feats.append(states[2] ** 2)
        feats.append(states[4] * states[5])
        feats.append(states[6] * states[7])

        feats.append(states[6])
        feats.append(states[3] * states[10])
        feats.append(states[11] * states[12])

        feats.append(states[11])
        feats.append(states[14] * states[15])
        feats.append(states[16] * states[0])

        feats.append(states[11] * states[4])
        feats.append(states[4] ** 2)
        feats.append(states[13])

    elif feat_selection == "manual":
        feats.append(states[0])
        feats.append(states[4])
        feats.append(states[8])

        feats.append(states[9])
        feats.append(states[13])
        feats.append(states[14])

        feats.append(states[13] ** 2)
        feats.append(states[2])
        feats.append(states[4] * states[5])

        feats.append(states[11])
        feats.append(states[14] * states[12])
        feats.append(states[13] * states[0])

    elif feat_selection == "proposed":
        notex = cfg["path_to_basis"]
        if PRECOMPILED_FUNC is None:
            filepath = f"tmp/{env_name}_basis.joblib"
            PRECOMPILED_FUNC = load_and_precompile_functions_joblib(filepath)
        feats = compute_with_precompiled_functions(PRECOMPILED_FUNC, states)
    else:
        NotImplementedError()
    return np.array(feats)


def walker_feat_extract(states, cfg):
    global PRECOMPILED_FUNC
    env_name = "Walker2d-v4"
    min_vals = ENVIRONMENT_CONFIGS[env_name]["min_vals"]
    max_vals = ENVIRONMENT_CONFIGS[env_name]["max_vals"]
    states = normalize_data_point(states, min_vals, max_vals)
    use_norm = cfg["normalize_feats"]
    if use_norm:
        states = Normalizer().fit_transform(states.reshape(1, -1)).squeeze()

    feat_selection = cfg["feats_method"]
    feats = []

    if feat_selection == "first":
        feats = states[:17]

    elif feat_selection == "random":
        feats.append(states[3] ** 2)
        feats.append(states[4] * states[1])
        feats.append(states[2] * states[3])

        feats.append(states[6])
        feats.append(states[3] * states[5])
        feats.append(states[10] * states[2])

        feats.append(states[7])
        feats.append(states[15] * states[1])
        feats.append(states[12])

    elif feat_selection == "manual":
        feats.append(states[0] * states[2] - states[1] * states[3])
        feats.append(states[0])
        feats.append(states[2] ** 2)
        feats.append(states[1] ** 2)

    elif feat_selection == "proposed":
        notex = cfg["path_to_basis"]
        if PRECOMPILED_FUNC is None:
            filepath = f"tmp/{env_name}_basis.joblib"
            PRECOMPILED_FUNC = load_and_precompile_functions_joblib(filepath)
        feats = compute_with_precompiled_functions(PRECOMPILED_FUNC, states)
    else:
        NotImplementedError()
    return np.array(feats)


def ant_feat_extract(states, cfg):
    global PRECOMPILED_FUNC
    env_name = "Ant-v4"
    min_vals = ENVIRONMENT_CONFIGS[env_name]["min_vals"]
    max_vals = ENVIRONMENT_CONFIGS[env_name]["max_vals"]
    states = normalize_data_point(states, min_vals, max_vals)
    use_norm = cfg["normalize_feats"]
    if use_norm:
        states = Normalizer().fit_transform(states.reshape(1, -1)).squeeze()
    feats = []
    feat_selection = cfg["feats_method"]

    if feat_selection == "first":
        feats = states[:27]

    elif feat_selection == "random":
        feats.append(states[5] ** 2)
        feats.append(states[2] * states[5])
        feats.append(states[2] * states[7])

        feats.append(states[7])
        feats.append(states[8] * states[10])
        feats.append(states[5] * states[2])

        feats.append(states[3])
        feats.append(states[12] * states[7])
        feats.append(states[17] * states[0])

        feats.append(states[11] * states[20])
        feats.append(states[4] ** 2)
        feats.append(states[16])

    elif feat_selection == "manual":
        feats.append(states[0])
        feats.append(states[1] ** 2)
        feats.append(states[2])

        feats.append(states[4] ** 2)
        feats.append(states[13])
        feats.append(states[14])

        feats.append(states[15])
        feats.append(states[16] ** 2)
        feats.append(states[17] ** 2)

        feats.append(states[21])
        feats.append(states[24] * states[2])
        feats.append(states[23] * states[1])

    elif feat_selection == "proposed":
        notex = cfg["path_to_basis"]
        if PRECOMPILED_FUNC is None:
            filepath = f"tmp/{env_name}_basis.joblib"
            PRECOMPILED_FUNC = load_and_precompile_functions_joblib(filepath)
        feats = compute_with_precompiled_functions(PRECOMPILED_FUNC, states)
    else:
        NotImplementedError()
    return np.array(feats)


def hopper_feat_extract(states, cfg):
    global PRECOMPILED_FUNC
    env_name = "Hopper-v4"
    min_vals = ENVIRONMENT_CONFIGS[env_name]["min_vals"]
    max_vals = ENVIRONMENT_CONFIGS[env_name]["max_vals"]
    states = normalize_data_point(states, min_vals, max_vals)
    use_norm = cfg["normalize_feats"]
    if use_norm:
        states = Normalizer().fit_transform(states.reshape(1, -1)).squeeze()

    feat_selection = cfg["feats_method"]
    feats = []

    if feat_selection == "first":
        feats = states[:11]

    elif feat_selection == "random":
        feats.append(states[3] ** 2)
        feats.append(states[2] * states[3])
        feats.append(states[7] * states[1])

        feats.append(states[3] * states[1])
        feats.append(states[1] * states[2])
        feats.append(states[9])

        feats.append(states[2] * states[6])
        feats.append(states[5] * states[2])

    elif feat_selection == "manual":
        feats.append(states[2] * states[1])
        feats.append(states[3] ** 2)
        feats.append(states[7])

        feats.append(states[5] * states[6])
        feats.append(states[6])
        feats.append(states[6])

        feats.append(states[3] * states[5])
        feats.append(states[0])

    elif feat_selection == "proposed":
        notex = cfg["path_to_basis"]
        if PRECOMPILED_FUNC is None:
            filepath = f"tmp/{env_name}_basis.joblib"
            PRECOMPILED_FUNC = load_and_precompile_functions_joblib(filepath)
        feats = compute_with_precompiled_functions(PRECOMPILED_FUNC, states)
    else:
        NotImplementedError()
    return np.array(feats)
