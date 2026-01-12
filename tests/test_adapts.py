"""Tests for ADAPTS main class."""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from adapts.adapts import ADAPTS
from adapts.adapters import IdentityTransformer, MultichannelProjector


class TestADAPTS:
    """Test suite for ADAPTS class."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        adapter = Mock()
        iclearner = Mock()
        
        adapts = ADAPTS(
            adapter=adapter,
            iclearner=iclearner,
            n_features=5,
            n_components=3,
        )
        
        assert adapts.n_features == 5
        assert adapts.n_components == 3
        assert adapts.adapter is adapter
        assert adapts.iclearner is iclearner

    def test_init_with_scaler_preprocessing(self):
        """Test initialization with scaler preprocessing enabled."""
        adapter = Mock()
        iclearner = Mock()
        
        adapts = ADAPTS(
            adapter=adapter,
            iclearner=iclearner,
            n_features=5,
            n_components=3,
            scaler_in_preprocessing=True,
        )
        
        assert adapts.scaler is not None
        # Should have MinMax and Standard scalers
        assert len(adapts.scaler) == 2

    def test_init_with_pca_preprocessing(self):
        """Test initialization with PCA and scaler preprocessing."""
        adapter = Mock()
        iclearner = Mock()
        
        adapts = ADAPTS(
            adapter=adapter,
            iclearner=iclearner,
            n_features=5,
            n_components=3,
            pca_in_preprocessing=True,
            scaler_in_preprocessing=True,
        )
        
        assert adapts.scaler is not None
        # Should have MinMax, Standard scalers and PCA
        assert len(adapts.scaler) == 3

    def test_init_without_preprocessing(self):
        """Test initialization without preprocessing."""
        adapter = Mock()
        iclearner = Mock()
        
        adapts = ADAPTS(
            adapter=adapter,
            iclearner=iclearner,
            n_features=5,
            n_components=3,
            scaler_in_preprocessing=False,
        )
        
        # Should use IdentityTransformer
        assert isinstance(adapts.scaler, IdentityTransformer)

    def test_fit_adapter(self):
        """Test fit_adapter method."""
        adapter = Mock()
        adapter.fit = Mock()
        iclearner = Mock()
        
        adapts = ADAPTS(
            adapter=adapter,
            iclearner=iclearner,
            n_features=5,
            n_components=3,
            scaler_in_preprocessing=False,
        )
        
        X = np.random.rand(10, 5, 32)
        adapts.fit_adapter(X)
        
        # Verify adapter.fit was called
        adapter.fit.assert_called_once()

    def test_transform(self):
        """Test transform method."""
        adapter = Mock()
        adapter.transform = Mock(return_value=np.random.rand(10, 3, 32))
        iclearner = Mock()
        
        adapts = ADAPTS(
            adapter=adapter,
            iclearner=iclearner,
            n_features=5,
            n_components=3,
            scaler_in_preprocessing=False,
        )
        
        X = np.random.rand(10, 5, 32)
        result = adapts.transform(X)
        
        # Verify adapter.transform was called
        adapter.transform.assert_called_once()
        assert result.shape == (10, 3, 32)

    def test_inverse_transform(self):
        """Test inverse_transform method."""
        # Create a mock adapter that returns data with correct structure
        adapter = Mock()
        # inverse_transform should return data with proper shape
        adapter.inverse_transform = Mock(return_value=np.random.rand(10, 5, 32))
        adapter.base_projector_ = Mock(spec=[])  # Empty spec means no 'likelihood' attribute
        
        iclearner = Mock()
        
        adapts = ADAPTS(
            adapter=adapter,
            iclearner=iclearner,
            n_features=5,
            n_components=3,
            scaler_in_preprocessing=False,
        )
        
        X = np.random.rand(10, 3, 32)
        result = adapts.inverse_transform(X)
        
        # Verify adapter.inverse_transform was called
        adapter.inverse_transform.assert_called_once()
        assert result.shape == (10, 5, 32)


class TestADAPTSIntegration:
    """Integration tests for ADAPTS with real components."""

    def test_adapts_with_identity_transformer(self):
        """Test ADAPTS with IdentityTransformer."""
        adapter = MultichannelProjector(
            num_channels=5,
            new_num_channels=5,
            device="cpu"
        )
        iclearner = Mock()
        
        adapts = ADAPTS(
            adapter=adapter,
            iclearner=iclearner,
            n_features=5,
            n_components=5,
            scaler_in_preprocessing=False,
        )
        
        X = np.random.rand(10, 5, 32)
        adapts.fit_adapter(X)
        X_transformed = adapts.transform(X)
        
        assert X_transformed.shape == X.shape

    def test_adapts_with_dimensionality_reduction(self):
        """Test ADAPTS with dimensionality reduction."""
        adapter = MultichannelProjector(
            num_channels=8,
            new_num_channels=4,
            base_projector="pca",
            device="cpu"
        )
        iclearner = Mock()
        
        adapts = ADAPTS(
            adapter=adapter,
            iclearner=iclearner,
            n_features=8,
            n_components=4,
            scaler_in_preprocessing=False,
        )
        
        X = np.random.rand(20, 8, 64)
        adapts.fit_adapter(X)
        X_transformed = adapts.transform(X)
        
        # Should reduce from 8 to 4 channels
        assert X_transformed.shape[0] == 20
        assert X_transformed.shape[1] == 4

    def test_adapts_transform_inverse_transform_consistency(self):
        """Test that transform and inverse_transform are consistent."""
        adapter = MultichannelProjector(
            num_channels=5,
            new_num_channels=5,
            device="cpu"
        )
        iclearner = Mock()
        
        adapts = ADAPTS(
            adapter=adapter,
            iclearner=iclearner,
            n_features=5,
            n_components=5,
            scaler_in_preprocessing=False,
        )
        
        X = np.random.rand(10, 5, 32)
        adapts.fit_adapter(X)
        X_transformed = adapts.transform(X)
        X_recovered = adapts.inverse_transform(X_transformed)
        
        # Should approximately recover original shape
        assert X_recovered.shape == X.shape
