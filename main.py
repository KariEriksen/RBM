from rbm import non_interaction_case, weak_interaction_case, strong_interaction_case # noqa: 401

"""case(monte_carlo_cycles, number of particles,
        number of dimensions, number of hidden nodes)"""

# non_interaction_case(10000, 1, 2, 2)
# weak_interaction_case(10000, 2, 2, 3)
strong_interaction_case(10, 2, 3, 3)
