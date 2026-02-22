"""
Blast Radius Assessment Engine.

This module provides impact assessment capabilities for detected incidents,
quantifying the scope of business impact across customers, revenue, orders,
and operational metrics.

Components:
    BlastRadiusMapper: BFS/DFS graph traversal to map affected entities
    ImpactScorer: Quantitative impact scoring and severity classification
    BlastRadiusVisualizer: Cytoscape.js JSON generation for frontend graphs

Example:
    >>> from api.engine.blast_radius import BlastRadiusMapper
    >>> mapper = BlastRadiusMapper(storage=storage)
    >>> blast_radius = mapper.compute_blast_radius(incident)
    >>> print(f"Customers affected: {blast_radius.customers_affected}")
"""

from .impact_scorer import ImpactScorer
from .mapper import BlastRadiusMapper
from .visualizer import BlastRadiusVisualizer

__all__ = [
    "BlastRadiusMapper",
    "ImpactScorer",
    "BlastRadiusVisualizer",
]
