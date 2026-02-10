"""
Core Business Reliability Engine.

Detection: Statistical + ML hybrid anomaly detection
RCA: Root cause analysis using graph-based causal inference
Blast Radius: Impact propagation through entity graphs
Monitors: SLO evaluation and health scoring
"""

# Engine modules:
# Detection:
# - detection/statistical.py: Z-score, IQR, MAD detectors
# - detection/changepoint.py: Ruptures integration
# - detection/ml_detector.py: LightGBM classifier
# - detection/ensemble.py: Weighted ensemble scoring
#
# RCA:
# - rca/graph_builder.py: NetworkX entity graphs
# - rca/temporal_correlation.py: Cross-correlation analysis
# - rca/causal_ranker.py: PageRank-based ranking
# - rca/explainer.py: Natural language explanations
#
# Blast Radius:
# - blast_radius/mapper.py: Graph traversal
# - blast_radius/impact_scorer.py: Impact quantification
# - blast_radius/visualizer.py: Cytoscape JSON generation
#
# Monitors:
# - monitors/slo_evaluator.py: SLO compliance
# - monitors/alert_router.py: Alert routing
# - monitors/health_scorer.py: Composite health scoring
