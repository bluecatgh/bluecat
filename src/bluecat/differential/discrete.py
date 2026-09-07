def diff(t, x):

    if len(t) != len(x):
        raise ValueError("Time and signal arrays must have the same length.")

    v = []

    for k in range(1, len(t)):
        # Compute discrete derivative between consecutive data points
        derivative = (x[k] - x[k - 1]) / (t[k] - t[k - 1])
        v.append(derivative)

    return v
