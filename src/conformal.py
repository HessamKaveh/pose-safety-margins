"""Uncertainty radii for 2D keypoints.

Every method returns, for each (person, joint), a radius r such that the true joint is claimed to lie
within r pixels of the predicted one. Conformal methods learn a quantile on a calibration set; with
exchangeable data split conformal prediction guarantees  P(error <= r) >= 1 - alpha  on new data.

A  heatmap-gaussian   the model's own uncertainty: heatmap spread sigma read as an isotropic 2D Gaussian,
                      r = sigma * sqrt(-2 ln alpha). No calibration data used.
B  cp-fixed           split CP per joint on the scale-normalised error e / h (h = person box height),
                      r = q_j * h. Same relative radius for every person.
C  cp-adaptive        split CP per joint on the normalised score e / sigma ("normalised conformal"),
                      r = q_j * sigma. Radius grows where the heatmap is spread out.
D  cp-mondrian        like C, but a separate quantile for each joint x confidence tercile (Mondrian CP),
                      giving coverage within each confidence group, not only on average.
W  cp-whole-body      C with one score per person: the max over its labelled joints, so that *all* joints
                      of a person are covered at the same time with probability >= 1 - alpha.
"""
import numpy as np

METHODS = ["heatmap-gaussian", "cp-fixed", "cp-adaptive", "cp-mondrian"]


def cp_quantile(scores, alpha):
    """Finite-sample conformal quantile: the ceil((n+1)(1-alpha))-th smallest score."""
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return np.partition(scores, k - 1)[k - 1]


class Calibrator:
    def __init__(self, method, alpha):
        self.method, self.alpha = method, alpha

    def fit(self, d):
        """d: dict of flat arrays over labelled (person, joint) pairs of the calibration set."""
        m, a = self.method, self.alpha
        self.q = {}
        if m == "cp-fixed":
            for j in range(17):
                s = d["err"][d["joint"] == j] / d["h"][d["joint"] == j]
                self.q[j] = cp_quantile(s, a)
        elif m == "cp-adaptive":
            for j in range(17):
                k = d["joint"] == j
                self.q[j] = cp_quantile(d["err"][k] / d["sigma"][k], a)
        elif m == "cp-mondrian":
            self.edges = {}
            for j in range(17):
                k = d["joint"] == j
                e = np.quantile(d["conf"][k], [1 / 3, 2 / 3])
                self.edges[j] = e
                b = np.digitize(d["conf"][k], e)
                s = d["err"][k] / d["sigma"][k]
                for g in range(3):
                    self.q[j, g] = cp_quantile(s[b == g], a)
        return self

    def radius(self, d):
        m, a = self.method, self.alpha
        if m == "heatmap-gaussian":
            return d["sigma"] * np.sqrt(-2 * np.log(a))
        r = np.empty(len(d["err"]), np.float64)
        for j in range(17):
            k = d["joint"] == j
            if m == "cp-fixed":
                r[k] = self.q[j] * d["h"][k]
            elif m == "cp-adaptive":
                r[k] = self.q[j] * d["sigma"][k]
            else:
                b = np.digitize(d["conf"][k], self.edges[j])
                r[k] = np.choose(b, [self.q[j, 0], self.q[j, 1], self.q[j, 2]]) * d["sigma"][k]
        return r


def whole_body(cal, test, alpha):
    """Simultaneous coverage of all labelled joints of a person (score = max over joints of e / sigma)."""
    s_cal = np.zeros(cal["n_persons"]); np.maximum.at(s_cal, cal["person"], cal["err"] / cal["sigma"])
    q = cp_quantile(s_cal[np.unique(cal["person"])], alpha)
    return q * test["sigma"]
