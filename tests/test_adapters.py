"""Tests for adapters module."""
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from adapts.adapters import IdentityTransformer, MultichannelProjector


class TestIdentityTransformer:
    """Test suite for IdentityTransformer."""

    def test_init(self):
        """Test initialization of IdentityTransformer."""
        transformer = IdentityTransformer()
        assert transformer is not None

    def test_fit_returns_self(self):
        """Test that fit returns self."""
        transformer = IdentityTransformer()
        X = np.random.rand(10, 5)
        result = transformer.fit(X)
        assert result is transformer

    def test_transform_no_modification(self):
        """Test that transform returns unmodified data."""
        transformer = IdentityTransformer()
        X = np.random.rand(10, 5)
        X_transformed = transformer.transform(X)
        assert_array_almost_equal(X, X_transformed)

    def test_inverse_transform_no_modification(self):
        """Test that inverse_transform returns unmodified data."""
        transformer = IdentityTransformer()
        X = np.random.rand(10, 5)
        X_inverse = transformer.inverse_transform(X)
        assert_array_almost_equal(X, X_inverse)

    def test_forward_no_modification(self):
        """Test that forward returns unmodified data."""
        transformer = IdentityTransformer()
        X = np.random.rand(10, 5)
        X_forward = transformer.forward(X)
        assert_array_almost_equal(X, X_forward)

    def test_transform_with_different_shapes(self):
        """Test transform with different input shapes."""
        transformer = IdentityTransformer()
        
        # 1D array
        X_1d = np.random.rand(10)
        assert_array_almost_equal(X_1d, transformer.transform(X_1d))
        
        # 2D array
        X_2d = np.random.rand(10, 5)
        assert_array_almost_equal(X_2d, transformer.transform(X_2d))
        
        # 3D array
        X_3d = np.random.rand(10, 5, 3)
        assert_array_almost_equal(X_3d, transformer.transform(X_3d))


class TestMultichannelProjector:
    """Test suite for MultichannelProjector."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        projector = MultichannelProjector(
            num_channels=5,
            new_num_channels=3,
            device="cpu"
        )
        assert projector.num_channels == 5
        assert projector.new_num_channels == 3
        assert projector.patch_window_size_ == 1

    def test_init_with_patch_window(self):
        """Test initialization with patch window size."""
        projector = MultichannelProjector(
            num_channels=5,
            new_num_channels=3,
            patch_window_size=4,
            device="cpu"
        )
        assert projector.patch_window_size_ == 4

    def test_init_with_pca_projector(self):
        """Test initialization with PCA base projector."""
        projector = MultichannelProjector(
            num_channels=5,
            new_num_channels=3,
            base_projector="pca",
            device="cpu"
        )
        assert projector.base_projector == "pca"
        assert projector.base_projector_ is not None

    def test_init_with_svd_projector(self):
        """Test initialization with SVD base projector."""
        projector = MultichannelProjector(
            num_channels=5,
            new_num_channels=3,
            base_projector="svd",
            device="cpu"
        )
        assert projector.base_projector == "svd"

    def test_fit_shape_validation(self):
        """Test fit with correct shape validation."""
        projector = MultichannelProjector(
            num_channels=5,
            new_num_channels=3,
            device="cpu"
        )
        # Shape: (num_samples, num_channels, seq_len)
        X = np.random.rand(10, 5, 32)
        projector.fit(X)

    def test_fit_wrong_channels(self):
        """Test fit raises error with wrong number of channels."""
        projector = MultichannelProjector(
            num_channels=5,
            new_num_channels=3,
            device="cpu"
        )
        # Wrong number of channels (3 instead of 5)
        X = np.random.rand(10, 3, 32)
        with pytest.raises(AssertionError):
            projector.fit(X)

    def test_transform_basic(self):
        """Test basic transform functionality with identity transformer."""
        # With default (no base_projector), it uses IdentityTransformer
        # which doesn't actually reduce dimensions
        projector = MultichannelProjector(
            num_channels=5,
            new_num_channels=5,  # Must match num_channels when no projector
            device="cpu"
        )
        X = np.random.rand(10, 5, 32)
        projector.fit(X)
        X_transformed = projector.transform(X)
        
        # Check output shape
        assert X_transformed.shape[0] == 10  # num_samples
        assert X_transformed.shape[1] == 5   # channels (unchanged with identity)
        assert X_transformed.shape[2] == 32  # seq_len

    def test_transform_with_pca(self):
        """Test transform with PCA projector."""
        projector = MultichannelProjector(
            num_channels=5,
            new_num_channels=3,
            base_projector="pca",
            device="cpu"
        )
        X = np.random.rand(20, 5, 64)
        projector.fit(X)
        X_transformed = projector.transform(X)
        
        assert X_transformed.shape[0] == 20
        assert X_transformed.shape[1] == 3

    def test_fit_transform_pipeline(self):
        """Test complete fit and transform pipeline with identity transformer."""
        # With default (no base_projector), channels remain unchanged
        projector = MultichannelProjector(
            num_channels=8,
            new_num_channels=8,  # Must match when no projector
            device="cpu"
        )
        X_train = np.random.rand(50, 8, 128)
        X_test = np.random.rand(10, 8, 128)
        
        projector.fit(X_train)
        X_test_transformed = projector.transform(X_test)
        
        assert X_test_transformed.shape[0] == 10
        assert X_test_transformed.shape[1] == 8
        assert X_test_transformed.shape[2] == 128
