import numpy as np
from scipy.spatial.distance import cdist


def find_pareto_front(data):
    """Returns the indices of the sequences that are in the pareto front,
    considering energy gap, energy of forward topology, and sequence repeat lengths."""
    return is_pareto_efficient_indexed(data)


def is_pareto_efficient_indexed(costs, return_mask=False):
    """
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask, False to return integer indices of efficient points.
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.

    This code is from username Peter at
    https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for

    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs <= costs[next_point_index], axis=1)
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def normalize(data_column):
    data_max = data_column.max()
    data_min = data_column.min()
    return (data_column - data_max) / (data_min - data_max)


def find_nearest_utopian_pt(data, pareto_front, weights=None):
    """data should have columns that are values of individual parameters."""
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    data = data[pareto_front, :]

    data_shape = data.shape
    num_params = data_shape[1]
    if weights is None:
        weights = np.ones(num_params).reshape(-1)
    else:
        weights = np.array(weights).reshape(-1)

    pareto_points = np.empty(data_shape)
    for i in range(data.shape[1]):
        if len(set(data[:, i])) > 1:
            pareto_points[:, i] = normalize(data[:, i])
        else:
            pareto_points[:, i] = 1
    dists = cdist(pareto_points, np.ones(num_params).reshape(1, -1), 'wminkowski', p=2., w=weights)
    nearest_utopian_pt = pareto_front[(dists == min(dists)).flatten()]
    if len(nearest_utopian_pt) > 1:
        return nearest_utopian_pt[0]
    else:
        return int(nearest_utopian_pt)

