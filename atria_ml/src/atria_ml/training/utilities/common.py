from codename import codename


def random_experiment_name() -> str:
    """
    Generates a random experiment name based on the current timestamp.

    Returns:
        str: A string representing the current timestamp.
    """

    return "-".join(codename())
