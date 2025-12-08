"""
Hyperbolic Geometry Module

Implements the Poincaré Ball model of hyperbolic space.
Hyperbolic geometry has exponential distance growth near the boundary,
making it ideal for representing hierarchical data structures.

The hippocampus uses hyperbolic geometry for spatial representation!
(Nature 2022 - Evolution optimized neural circuits for hyperbolic space)

Mathematical Foundation:
- Poincaré Ball: Unit ball with hyperbolic metric
- Möbius operations: Hyperbolic analogues of Euclidean operations
- Geodesics: "Straight lines" in curved space
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

# Constants
EPSILON = 1e-10
DEFAULT_CURVATURE = -1.0


class PoincareBall:
    """
    The Poincaré ball model of hyperbolic space.

    The entire hyperbolic n-space is represented inside a unit ball.
    The boundary represents points at infinity.

    Key property: Exponential distance growth near boundary enables
    efficient representation of hierarchical structures.
    """

    def __init__(self, dim: int = 8, curvature: float = DEFAULT_CURVATURE):
        """
        Initialize Poincaré Ball.

        Args:
            dim: Dimension of the space (default: 8 for E8)
            curvature: Negative curvature magnitude (default: -1.0)
        """
        self.dim = dim
        self.c = abs(curvature)  # Curvature magnitude
        self.max_norm = 1.0 - EPSILON  # Stay inside ball

    def project_to_ball(self, x: np.ndarray) -> np.ndarray:
        """Project Euclidean point to Poincaré ball."""
        x = np.asarray(x, dtype=np.float64)
        norm = np.linalg.norm(x)

        if norm >= self.max_norm:
            return x / norm * self.max_norm
        return x

    def distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Compute hyperbolic distance in Poincaré ball.

        d(u,v) = (1/√c) * acosh(1 + 2c * ||u-v||² / ((1-c||u||²)(1-c||v||²)))
        """
        u = self.project_to_ball(u)
        v = self.project_to_ball(v)

        diff_norm_sq = np.sum((u - v) ** 2)
        u_norm_sq = np.sum(u**2)
        v_norm_sq = np.sum(v**2)

        denominator = (1 - self.c * u_norm_sq) * (1 - self.c * v_norm_sq)
        if denominator < EPSILON:
            denominator = EPSILON

        x = 1 + 2 * self.c * diff_norm_sq / denominator

        # acosh(x) = log(x + sqrt(x²-1))
        if x < 1:
            x = 1

        dist = np.arccosh(x) / np.sqrt(self.c)
        return float(dist)

    def mobius_add(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Möbius addition in Poincaré ball.

        This is the hyperbolic equivalent of vector addition.

        u ⊕ v = ((1 + 2c<u,v> + c||v||²)u + (1 - c||u||²)v) /
                (1 + 2c<u,v> + c²||u||²||v||²)
        """
        u = self.project_to_ball(u)
        v = self.project_to_ball(v)

        u_norm_sq = np.sum(u**2)
        v_norm_sq = np.sum(v**2)
        uv_dot = np.dot(u, v)

        numerator = (1 + 2 * self.c * uv_dot + self.c * v_norm_sq) * u + (
            1 - self.c * u_norm_sq
        ) * v
        denominator = 1 + 2 * self.c * uv_dot + self.c**2 * u_norm_sq * v_norm_sq

        if abs(denominator) < EPSILON:
            denominator = EPSILON

        result = numerator / denominator
        return self.project_to_ball(result)

    def geodesic(self, u: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
        """
        Compute point along geodesic from u to v at parameter t.

        t=0 gives u, t=1 gives v.

        The geodesic is computed using the exponential map.
        """
        u = self.project_to_ball(u)
        v = self.project_to_ball(v)

        if t <= 0:
            return u.copy()
        if t >= 1:
            return v.copy()

        # Compute the tangent vector at u pointing toward v
        # First, compute the log map (inverse of exp map)
        minus_u = -u
        v_minus_u = self.mobius_add(minus_u, v)

        # Scale the tangent vector by t
        tangent_norm = np.linalg.norm(v_minus_u)
        if tangent_norm < EPSILON:
            return u.copy()

        # Apply exponential map at u with scaled tangent
        scaled_tangent = v_minus_u * t

        return self.exp_map(u, scaled_tangent)

    def exp_map(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Exponential map: Move from x in direction v.

        exp_x(v) = x ⊕ tanh(√c * ||v|| / (1 - c||x||²)) * v / (√c * ||v||)
        """
        x = self.project_to_ball(x)

        x_norm_sq = np.sum(x**2)
        v_norm = np.linalg.norm(v)

        if v_norm < EPSILON:
            return x.copy()

        sqrt_c = np.sqrt(self.c)
        lambda_x = 1.0 / (1 - self.c * x_norm_sq + EPSILON)

        arg = sqrt_c * lambda_x * v_norm
        coeff = np.tanh(np.clip(arg, -10, 10)) / (sqrt_c * v_norm + EPSILON)

        direction = v * coeff

        return self.mobius_add(x, direction)

    def log_map(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Logarithmic map: Compute tangent vector at x pointing to y.

        log_x(y) = (1 - c||x||²) / √c * atanh(√c * ||-x ⊕ y||) * (-x ⊕ y) / ||-x ⊕ y||
        """
        x = self.project_to_ball(x)
        y = self.project_to_ball(y)

        minus_x = -x
        diff = self.mobius_add(minus_x, y)

        diff_norm = np.linalg.norm(diff)
        if diff_norm < EPSILON:
            return np.zeros_like(x)

        x_norm_sq = np.sum(x**2)
        sqrt_c = np.sqrt(self.c)
        lambda_x = 1.0 / (1 - self.c * x_norm_sq + EPSILON)

        arg = sqrt_c * diff_norm
        arg = np.clip(arg, -1 + EPSILON, 1 - EPSILON)

        coeff = np.arctanh(arg) / (sqrt_c * lambda_x * diff_norm + EPSILON)

        return diff * coeff

    def centroid(self, points: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the hyperbolic centroid (Fréchet mean) of points.

        Uses iterative optimization in the tangent space.
        """
        points = np.array([self.project_to_ball(p) for p in points])
        n = len(points)

        if n == 0:
            return np.zeros(self.dim)
        if n == 1:
            return points[0].copy()

        if weights is None:
            weights = np.ones(n) / n
        else:
            weights = np.asarray(weights)
            weights = weights / weights.sum()

        # Initialize with Euclidean weighted mean
        mean = np.sum(points * weights[:, np.newaxis], axis=0)
        mean = self.project_to_ball(mean)

        # Gradient descent in tangent space
        for _ in range(100):
            # Compute gradient (sum of log maps)
            gradient = np.zeros(self.dim)
            for i, p in enumerate(points):
                gradient += weights[i] * self.log_map(mean, p)

            grad_norm = np.linalg.norm(gradient)
            if grad_norm < 1e-6:
                break

            # Step along gradient
            step_size = 0.1
            mean = self.exp_map(mean, gradient * step_size)

        return mean


class LorentzModel:
    """
    The Lorentz (hyperboloid) model of hyperbolic space.

    An alternative to the Poincaré ball with different numerical properties.
    Uses Minkowski inner product: <x,y> = -x₀y₀ + x₁y₁ + ... + xₙyₙ
    """

    def __init__(self, dim: int = 8):
        """
        Initialize Lorentz model.

        Args:
            dim: Dimension of the hyperbolic space (actual vectors are dim+1)
        """
        self.dim = dim

    def to_lorentz(self, poincare: np.ndarray) -> np.ndarray:
        """Convert Poincaré ball point to Lorentz model."""
        poincare = np.asarray(poincare)
        norm_sq = np.sum(poincare**2)

        # Ensure we're inside the ball
        if norm_sq >= 1 - EPSILON:
            poincare = poincare / np.sqrt(norm_sq) * (1 - EPSILON)
            norm_sq = np.sum(poincare**2)

        time_component = (1 + norm_sq) / (1 - norm_sq)
        space_components = 2 * poincare / (1 - norm_sq)

        return np.concatenate([[time_component], space_components])

    def to_poincare(self, lorentz: np.ndarray) -> np.ndarray:
        """Convert Lorentz model point to Poincaré ball."""
        lorentz = np.asarray(lorentz)

        # lorentz[0] is the time component
        return lorentz[1:] / (1 + lorentz[0])

    def minkowski_inner(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Minkowski inner product."""
        return -x[0] * y[0] + np.dot(x[1:], y[1:])

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute hyperbolic distance in Lorentz model."""
        inner = self.minkowski_inner(x, y)
        # Clamp for numerical stability
        inner = max(inner, -1.0)
        return float(np.arccosh(-inner))


# Convenience functions
_default_ball = PoincareBall(dim=8)


def hyperbolic_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Compute hyperbolic distance between two points."""
    return _default_ball.distance(u, v)


def geodesic_point(u: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
    """Get point along geodesic from u to v at parameter t."""
    return _default_ball.geodesic(u, v, t)


def hyperbolic_centroid(points: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute hyperbolic centroid of points."""
    return _default_ball.centroid(points, weights)
