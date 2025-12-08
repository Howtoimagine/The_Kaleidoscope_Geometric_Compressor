"""
Test E8 Lattice Core
"""

import numpy as np
import pytest
from e8zip.core.e8_lattice import E8Lattice, nearest_e8_point, get_e8_roots


class TestE8Lattice:
    """Tests for E8 lattice operations."""

    def test_root_count(self):
        """E8 should have exactly 240 roots."""
        lattice = E8Lattice()
        assert len(lattice.roots) == 240

    def test_root_norms(self):
        """All E8 roots should have norm √2."""
        lattice = E8Lattice()
        norms = np.linalg.norm(lattice.roots, axis=1)
        expected_norm = np.sqrt(2)
        np.testing.assert_allclose(norms, expected_norm, rtol=1e-10)

    def test_nearest_root(self):
        """Test finding nearest root."""
        lattice = E8Lattice()

        # A root should be its own nearest root
        root = lattice.roots[0]
        idx, nearest, dist = lattice.nearest_root(root)

        np.testing.assert_allclose(nearest, root)
        assert dist < 1e-10

    def test_nearest_lattice_point(self):
        """Test finding nearest lattice point."""
        lattice = E8Lattice()

        # Test with a simple integer point
        point = np.array([1.0, 1.0, 0, 0, 0, 0, 0, 0])
        idx, nearest, dist = lattice.nearest_lattice_point(point)

        # Should return exactly the input (it's already a lattice point)
        np.testing.assert_allclose(nearest, point)
        assert dist < 1e-10

    def test_project_to_8d(self):
        """Test projection to 8D."""
        lattice = E8Lattice()

        # Short vector should be zero-padded
        short = np.array([1.0, 2.0, 3.0])
        projected = lattice._ensure_8d(short)

        assert len(projected) == 8
        np.testing.assert_equal(projected[:3], short)
        np.testing.assert_equal(projected[3:], 0)

        # Long vector should be truncated
        long = np.arange(16, dtype=float)
        projected = lattice._ensure_8d(long)

        assert len(projected) == 8
        np.testing.assert_equal(projected, long[:8])

    def test_golden_ratio_weight(self):
        """Test golden ratio weighting."""
        lattice = E8Lattice()

        # Weight at k=k_star should be 1 + A (when phase=0)
        weight = lattice.golden_ratio_weight(0.05, amplitude=0.02, k_star=0.05, phase=0)
        assert abs(weight - 1.02) < 0.01

    def test_convenience_functions(self):
        """Test module-level convenience functions."""
        roots = get_e8_roots()
        assert len(roots) == 240

        point = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        idx, nearest, dist = nearest_e8_point(point)

        assert isinstance(idx, int)
        assert len(nearest) == 8


class TestE8Cosets:
    """Test E8 coset structure."""

    def test_coset_types(self):
        """Test that both coset types are represented in roots."""
        lattice = E8Lattice()

        # Count integer and half-integer roots
        integer_roots = 0
        half_integer_roots = 0

        for root in lattice.roots:
            if np.allclose(root % 0.5, 0) and not np.allclose(root % 1, 0):
                half_integer_roots += 1
            elif np.allclose(root % 1, 0) or np.allclose(np.abs(root), 1):
                integer_roots += 1

        # 112 integer-type, 128 half-integer-type
        assert integer_roots == 112
        assert half_integer_roots == 128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
