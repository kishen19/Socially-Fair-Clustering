from __future__ import division, absolute_import

import pytest
from sklearn.datasets import load_iris
import numpy as np

import coresets


class TestCoresets(object):

    @pytest.fixture
    def gen_data(self):
        X, _ = load_iris(return_X_y=True)
        return X

    def test_kmeans(self):
        # test that outlier has largest sensitivity
        X = np.random.rand(100, 2) * 0.1
        X[-1] = np.array([1, 1])

        coreset_gen = coresets.KMeansCoreset(X, n_clusters=1)
        assert np.alltrue(coreset_gen.p[-1] >= coreset_gen.p)

    def test_kmeans_lightweight(self, gen_data):
        X = gen_data
        lightweight_coreset_gen = coresets.KMeansLightweightCoreset(X)
        coreset_size = 50
        C, w = lightweight_coreset_gen.generate_coreset(coreset_size)
        # test that coreset has correct dimensions
        assert C.shape[0] == coreset_size
        assert C.shape[1] == X.shape[1]
        assert w.shape[0] == coreset_size

    def test_kmeans_lightweight_uniform(self):
        X = np.ones((100, 10))
        lightweight_coreset_gen = coresets.KMeansLightweightCoreset(X)
        coreset_size = 50
        C, w = lightweight_coreset_gen.generate_coreset(coreset_size)
        # test that coreset has correct dimensions
        assert C.shape[0] == coreset_size
        assert C.shape[1] == X.shape[1]
        assert w.shape[0] == coreset_size

        # assert that every point has weight 2
        assert np.allclose(w, np.ones(coreset_size) * 2)
