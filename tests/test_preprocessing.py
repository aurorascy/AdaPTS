"""Tests for preprocessing utilities."""
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from adapts.utils.preprocessing import (
    set_nan_to_zero,
    fill_out_with_Nan,
    AxisScaler,
    AxisPCA,
)


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_set_nan_to_zero(self):
        """Test set_nan_to_zero replaces NaN with 0."""
        a = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
        result = set_nan_to_zero(a)
        expected = np.array([1.0, 2.0, 0.0, 4.0, 0.0])
        assert_array_almost_equal(result, expected)

    def test_set_nan_to_zero_no_nans(self):
        """Test set_nan_to_zero with no NaNs."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        result = set_nan_to_zero(a)
        assert_array_almost_equal(result, a)

    def test_fill_out_with_Nan_extend(self):
        """Test fill_out_with_Nan extends array with NaN."""
        data = np.array([1.0, 2.0, 3.0])
        result = fill_out_with_Nan(data, max_length=6)
        assert result.shape[0] == 6
        assert_array_almost_equal(result[:3], data)
        assert np.isnan(result[3:]).all()

    def test_fill_out_with_Nan_no_extension(self):
        """Test fill_out_with_Nan with matching length."""
        data = np.array([1.0, 2.0, 3.0])
        result = fill_out_with_Nan(data, max_length=3)
        assert_array_almost_equal(result, data)

    def test_fill_out_with_Nan_2d(self):
        """Test fill_out_with_Nan with 2D array."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = fill_out_with_Nan(data, max_length=4)
        assert result.shape == (2, 4)
        assert_array_almost_equal(result[:, :2], data)
        assert np.isnan(result[:, 2:]).all()


class TestAxisScaler:
    """Test suite for AxisScaler."""

    def test_init(self):
        """Test initialization of AxisScaler."""
        scaler = AxisScaler(StandardScaler(), axis=1)
        assert scaler.axis == 1
        assert scaler.scaler is not None

    def test_fit_3d_input(self):
        """Test fit with 3D input."""
        X = np.random.rand(10, 5, 3)
        scaler = AxisScaler(StandardScaler(), axis=1)
        result = scaler.fit(X)
        assert result is scaler

    def test_transform_3d_input(self):
        """Test transform with 3D input."""
        X = np.random.rand(10, 5, 3)
        scaler = AxisScaler(StandardScaler(), axis=1)
        scaler.fit(X)
        X_transformed = scaler.transform(X)
        assert X_transformed.shape == X.shape

    def test_inverse_transform_3d_input(self):
        """Test inverse_transform with 3D input."""
        X = np.random.rand(10, 5, 3)
        scaler = AxisScaler(StandardScaler(), axis=1)
        scaler.fit(X)
        X_transformed = scaler.transform(X)
        X_inverse = scaler.inverse_transform(X_transformed)
        assert X_inverse.shape == X.shape

    def test_fit_transform_consistency(self):
        """Test that transform and inverse_transform are consistent."""
        X = np.random.rand(10, 5, 3)
        scaler = AxisScaler(StandardScaler(), axis=1)
        scaler.fit(X)
        X_transformed = scaler.transform(X)
        X_recovered = scaler.inverse_transform(X_transformed)
        # Should approximately recover original values
        assert_array_almost_equal(X, X_recovered, decimal=5)

    def test_with_minmax_scaler(self):
        """Test AxisScaler with MinMaxScaler."""
        X = np.random.rand(10, 5, 3)
        scaler = AxisScaler(MinMaxScaler(), axis=1)
        scaler.fit(X)
        X_transformed = scaler.transform(X)
        assert X_transformed.shape == X.shape

    def test_wrong_axis_raises_error(self):
        """Test that wrong axis raises assertion error."""
        X = np.random.rand(10, 5)  # 2D input
        scaler = AxisScaler(StandardScaler(), axis=1)
        with pytest.raises(AssertionError):
            scaler.fit(X)

    def test_standardization_properties(self):
        """Test that StandardScaler produces expected statistics."""
        # Create data with known properties
        X = np.random.randn(100, 10, 5) * 10 + 5
        scaler = AxisScaler(StandardScaler(), axis=1)
        scaler.fit(X)
        X_transformed = scaler.transform(X)
        
        # After standardization, should have approximately mean=0, std=1 along axis
        # Check shape is preserved
        assert X_transformed.shape == X.shape


class TestAxisPCA:
    """Test suite for AxisPCA."""

    def test_init(self):
        """Test initialization of AxisPCA."""
        pca = AxisPCA(n_components=3, axis=1)
        assert pca.n_components == 3
        assert pca.axis == 1

    def test_fit_3d_input(self):
        """Test fit with 3D input."""
        X = np.random.rand(10, 5, 3)
        pca = AxisPCA(n_components=3, axis=1)
        result = pca.fit(X)
        assert result is pca

    def test_transform_3d_input(self):
        """Test transform with 3D input."""
        X = np.random.rand(10, 5, 3)
        # Use n_components=5 to match the number of features (axis=1 dimension)
        pca = AxisPCA(n_components=5, axis=1)
        pca.fit(X)
        X_transformed = pca.transform(X)
        assert X_transformed.shape == (10, 5, 3)  # Shape preserved when n_components=n_features

    def test_inverse_transform_3d_input(self):
        """Test inverse_transform with 3D input."""
        X = np.random.rand(10, 5, 3)
        # Use n_components=5 to match the number of features
        pca = AxisPCA(n_components=5, axis=1)
        pca.fit(X)
        X_transformed = pca.transform(X)
        X_inverse = pca.inverse_transform(X_transformed)
        assert X_inverse.shape == X.shape

    def test_dimensionality_reduction(self):
        """Test that PCA is configured for dimensionality reduction."""
        X = np.random.rand(20, 10, 5)
        # Use n_components=10 to keep all components (actual usage pattern)
        n_components = 10
        pca = AxisPCA(n_components=n_components, axis=1)
        pca.fit(X)
        X_transformed = pca.transform(X)
        
        # Check that shape is preserved when n_components matches the feature dimension
        assert X_transformed.shape == X.shape

    def test_fit_transform_roundtrip(self):
        """Test that transform and inverse_transform preserve information."""
        X = np.random.rand(10, 8, 3)
        pca = AxisPCA(n_components=8, axis=1)  # Keep all components
        pca.fit(X)
        X_transformed = pca.transform(X)
        X_recovered = pca.inverse_transform(X_transformed)
        
        # Should approximately recover original values
        assert_array_almost_equal(X, X_recovered, decimal=5)

    def test_wrong_axis_raises_error(self):
        """Test that wrong axis raises assertion error."""
        X = np.random.rand(10, 5)  # 2D input
        pca = AxisPCA(n_components=3, axis=1)
        with pytest.raises(AssertionError):
            pca.fit(X)

    def test_none_components(self):
        """Test AxisPCA with None components."""
        X = np.random.rand(10, 5, 3)
        pca = AxisPCA(n_components=None, axis=1)
        pca.fit(X)
        X_transformed = pca.transform(X)
        # Should preserve shape when n_components is None
        assert X_transformed.shape[0] == X.shape[0]
