import numpy as np
from tinygrad import Tensor
from tinyops.ml.nmf import nmf

def test_nmf_parity():
    """
    Tests NMF parity with scikit-learn by comparing reconstruction error.
    NMF output is not deterministic, so we compare the error rather than W and H directly.
    """
    from sklearn.decomposition import NMF

    # Set a seed for reproducibility
    np.random.seed(42)
    Tensor.manual_seed(42)

    n_samples, n_features, n_components = 20, 15, 5
    X_np = np.random.rand(n_samples, n_features).astype(np.float32)
    X_tg = Tensor(X_np)

    # scikit-learn NMF
    # Using 'nndsvd' for initialization is more stable for tests
    # but since our implementation is random, we stick to random for a fair comparison.
    # scikit-learn's random is different, so results won't be identical.
    model = NMF(n_components=n_components, init='random', random_state=42, max_iter=200, tol=1e-4, solver='mu')
    W_sklearn = model.fit_transform(X_np)
    H_sklearn = model.components_
    sklearn_error = np.linalg.norm(X_np - W_sklearn @ H_sklearn, 'fro')

    # tinyops NMF
    W_tinyops, H_tinyops = nmf(X_tg, n_components=n_components, max_iter=200, tol=1e-4)
    tinyops_error = (X_tg - W_tinyops @ H_tinyops).pow(2).sum().sqrt().item()

    # The errors should be very close, given the same algorithm but different random initializations.
    # We'll assert they are within a reasonable tolerance of each other.
    np.testing.assert_allclose(tinyops_error, sklearn_error, rtol=1e-1)
