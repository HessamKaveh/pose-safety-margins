import numpy as np
import pytest

from conformal import METHODS, Calibrator, cp_quantile, whole_body


def test_cp_quantile_finite_sample_rank():
    # n = 9, alpha = 0.1 -> k = ceil(10 * 0.9) = 9 -> the largest score
    s = np.arange(1, 10, dtype=float)
    assert cp_quantile(s, 0.1) == 9.0
    # n = 19, alpha = 0.1 -> k = ceil(20 * 0.9) = 18, independent of input order
    s = np.random.default_rng(0).permutation(np.arange(1, 20, dtype=float))
    assert cp_quantile(s, 0.1) == 18.0


def test_cp_quantile_edge_cases():
    assert cp_quantile(np.array([]), 0.1) == np.inf
    # n = 5, alpha = 0.1 -> k = ceil(6 * 0.9) = 6 > n -> infinite radius
    assert cp_quantile(np.arange(5, dtype=float), 0.1) == np.inf


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_cp_quantile_marginal_coverage(alpha):
    rng = np.random.default_rng(1)
    covs = []
    for _ in range(200):
        cal, test = rng.exponential(size=200), rng.exponential(size=1000)
        covs.append((test <= cp_quantile(cal, alpha)).mean())
    m = np.mean(covs)
    assert 1 - alpha <= m + 0.01
    assert m <= 1 - alpha + 1 / 201 + 0.01


def _synthetic(n_persons, rng):
    """Flat per-(person, joint) arrays in the format Calibrator expects; errors scale with sigma."""
    person = np.repeat(np.arange(n_persons), 17)
    joint = np.tile(np.arange(17), n_persons)
    sigma = rng.uniform(0.5, 5.0, len(person))
    err = sigma * np.sqrt(rng.chisquare(2, len(person)))
    return dict(err=err, sigma=sigma, conf=1.0 / sigma + rng.normal(0, 0.05, len(person)), joint=joint,
                person=person, h=rng.uniform(50, 300, n_persons)[person], n_persons=n_persons)


@pytest.mark.parametrize("method", METHODS)
def test_calibrator_coverage(method):
    rng = np.random.default_rng(2)
    alpha = 0.1
    cal, test = _synthetic(2000, rng), _synthetic(2000, rng)
    r = Calibrator(method, alpha).fit(cal).radius(test)
    assert r.shape == test["err"].shape and np.all(r > 0)
    cov = (test["err"] <= r).mean()
    if method == "heatmap-gaussian":
        # errors are exactly Rayleigh(sigma), for which sigma * sqrt(-2 ln alpha) is the exact 1 - alpha quantile
        assert abs(cov - (1 - alpha)) < 0.02
    else:
        assert cov >= 1 - alpha - 0.015


def test_whole_body_simultaneous_coverage():
    rng = np.random.default_rng(3)
    alpha = 0.1
    cal, test = _synthetic(3000, rng), _synthetic(3000, rng)
    r = whole_body(cal, test, alpha)
    covered = test["err"] <= r
    all_joints = np.ones(test["n_persons"], bool)
    np.logical_and.at(all_joints, test["person"], covered)
    assert all_joints.mean() >= 1 - alpha - 0.02
    # simultaneous coverage needs a wider radius than per-joint coverage
    assert np.median(r) > np.median(Calibrator("cp-adaptive", alpha).fit(cal).radius(test))
