import numpy as np

# Min and max values for each environment generated by running 
# random actions for multiple episodes to get approximate the min and max values
# for states to normalize
ENVIRONMENT_CONFIGS = {
    "HalfCheetah-v4": {
        "min_vals": 1.5 * np.array([
            -0.600, -3.220, -0.644, -0.858, -0.555, -1.072,
            -1.108, -0.681, -3.315, -3.527, -6.308, -20.155,
            -24.883, -23.288, -22.412, -25.717, -26.089
        ]),
        "max_vals": 1.5 * np.array([
            0.378, 3.802, 0.910, 0.865, 0.873, 0.814,
            1.017, 0.664, 3.221, 3.184, 6.993, 19.490,
            23.201, 20.072, 25.405, 26.938, 23.360
        ]),
    },
    "Walker2d-v4": {
        "min_vals": 1.5 * np.array([
            0.312, -1.000, -1.212, -2.367, -1.187, -1.279,
            -2.346, -1.167, -2.909, -6.356, -10.000, -10.000,
            -10.000, -10.000, -10.000, -10.000, -10.000
        ]),
        "max_vals": 1.5 * np.array([
            1.328, 0.121, 0.262, 0.178, 1.153, 0.243,
            0.224, 1.167, 1.365, 1.912, 10.000, 10.000,
            10.000, 10.000, 10.000, 10.000, 10.000
        ]),
    },
    "Ant-v4": {
        "min_vals": 1.5 * np.array([
            0.136, -1.000, -1.000, -1.000, -1.000, -0.686,
            -0.100, -0.680, -1.361, -0.680, -1.357, -0.678,
            -0.099, -3.635, -4.142, -3.867, -10.950, -8.932,
            -8.740, -16.185, -14.044, -15.993, -18.374,
            -16.869, -18.456, -16.225, -13.949
        ]),
        "max_vals": 1.3 * np.array([
            1.000, 1.000, 1.000, 1.000, 1.000, 0.681,
            1.351, 0.683, 0.100, 0.682, 0.100, 0.681,
            1.364, 4.081, 4.221, 5.764, 8.832, 8.763,
            9.109, 16.911, 18.407, 16.541, 14.151,
            16.197, 14.252, 16.509, 17.901
        ]),
    },
    "Hopper-v4": {
        "min_vals": 1.5 * np.array([
            0.444, -0.200, -0.706, -0.803, -0.712, -1.905,
            -1.631, -7.472, -8.452, -8.945, -7.972
        ]),
        "max_vals": 1.3 * np.array([
            1.324, 0.197, 0.039, 0.039, 0.852, 1.839,
            0.946, 4.782, 4.710, 5.279, 8.642
        ]),
    },
}