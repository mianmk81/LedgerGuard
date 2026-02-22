import { AlertTriangle, ChevronDown, ChevronRight, DollarSign, FileText, GitFork, Info, Loader2, TrendingDown, TrendingUp, Users, Zap } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../api/client'
import ConfidenceBadge from '../components/common/ConfidenceBadge'
import LoadingState from '../components/common/LoadingState'
import SeverityBadge from '../components/common/SeverityBadge'
import Graph3D from '../components/graph/Graph3D'

/**
 * Derives the impact category for a single RCA path node based on its z-score.
 * Mirrors the IMPACT_COLORS convention used in Graph3D.
 */
function nodeImpact(zscore) {
  if (zscore == null) return 'neutral'
  if (zscore < -2) return 'negative'
  if (zscore > 2) return 'positive'
  return 'neutral'
}

function explainZScore(zscore) {
  const abs = Math.abs(zscore || 0)
  if (abs >= 7) return 'Extremely unusual — this almost never happens under normal conditions'
  if (abs >= 5) return 'Very unusual — a strong signal that something is wrong'
  if (abs >= 3) return 'Significantly outside normal range — warrants immediate attention'
  if (abs >= 2) return 'Outside normal range — worth investigating'
  return 'Within normal range'
}

function explainSeverity(severity) {
  const s = (severity || '').toLowerCase()
  if (s === 'critical') return 'This needs immediate action — it could seriously impact your business'
  if (s === 'high') return 'This is a significant problem that should be addressed soon'
  if (s === 'medium') return 'This is a moderate issue — keep an eye on it and plan a fix'
  if (s === 'low') return 'Minor issue — not urgent but worth noting'
  return ''
}

function explainConfidence(confidence) {
  const c = (confidence || '').toLowerCase().replace(/_/g, ' ')
  if (c === 'very high') return 'Multiple detection methods agree — this is almost certainly a real issue, not a false alarm'
  if (c === 'high') return 'Strong evidence this is a real issue'
  if (c === 'medium') return 'Moderate evidence — could be real but may need more data to confirm'
  if (c === 'low') return 'Weak signal — might be noise, worth monitoring'
  return ''
}

function explainMetric(metricName, value, baseline) {
  const name = (metricName || '').replace(/_/g, ' ')
  if (value == null || baseline == null) return ''
  const pctChange = baseline !== 0 ? Math.abs((value - baseline) / baseline * 100) : 0
  const direction = value > baseline ? 'higher' : 'lower'
  return `${name} is currently ${pctChange.toFixed(0)}% ${direction} than normal (${baseline.toLocaleString(undefined, {maximumFractionDigits: 1})} is typical)`
}

/**
 * Converts a causal path object (from the RCA API) into a Cytoscape-format
 * graph that Graph3D can consume via its cytoscapeToForceGraph converter.
 *
 * Each node in path.nodes becomes a graph node. Edges connect them
 * sequentially: node[0] -> node[1] -> node[2] -> ...
 *
 * Node sizing is proportional to contribution_score so the most influential
 * metric in the chain appears largest in the 3-D view.
 */
function buildPathGraph(path) {
  if (!path?.nodes?.length) return null

  const elements = {
    nodes: path.nodes.map((node, idx) => {
      const contribution = node.contribution_score ?? 0
      // Scale size: minimum 20, maximum 80, proportional to contribution
      const size = 20 + Math.round(contribution * 60)
      // All nodes in a causal chain are part of an incident — always red
      const label = (node.metric_name || `node_${idx}`)
        .replace(/_/g, ' ')
        .replace(/\b\w/g, (l) => l.toUpperCase())

      return {
        data: {
          id: node.metric_name || `node_${idx}`,
          label,
          size,
          impact: 'negative',
          primary: idx === 0, // first node is the root cause
          metric_zscore: node.metric_zscore,
          contribution_score: node.contribution_score,
          metric_value: node.metric_value,
          metric_baseline: node.metric_baseline,
        },
      }
    }),
    edges: path.nodes.slice(0, -1).map((node, idx) => {
      const sourceId = node.metric_name || `node_${idx}`
      const targetId = path.nodes[idx + 1].metric_name || `node_${idx + 1}`
      return {
        data: {
          id: `edge_${idx}_${idx + 1}`,
          source: sourceId,
          target: targetId,
          label: 'causes',
          impact: 'negative',
        },
      }
    }),
  }

  return { elements }
}

/**
 * Merges all RCA causal paths into a single chain graph for the main 3D view.
 * Shows the full causal chain(s): root cause → ... → incident metric.
 * Each path contributes its nodes and edges; overlapping metrics are deduplicated
 * with the highest contribution_score winning.
 */
function buildMergedCausalGraph(causalChain) {
  if (!causalChain?.paths?.length) return null

  const nodeMap = new Map() // id -> { data }
  const edgeSet = new Set() // "source→target"

  for (const path of causalChain.paths) {
    if (!path?.nodes?.length) continue
    for (let i = 0; i < path.nodes.length; i++) {
      const node = path.nodes[i]
      const id = node.metric_name || `node_${i}`
      const existing = nodeMap.get(id)
      const contribution = node.contribution_score ?? 0
      if (!existing || (existing.data.contribution_score ?? 0) < contribution) {
        const label = (node.metric_name || `node_${i}`)
          .replace(/_/g, ' ')
          .replace(/\b\w/g, (l) => l.toUpperCase())
        const size = 20 + Math.round(contribution * 60)
        nodeMap.set(id, {
          data: {
            id,
            label,
            size,
            impact: 'negative',
            primary: i === 0,
            metric_zscore: node.metric_zscore,
            contribution_score: node.contribution_score,
            metric_value: node.metric_value,
            metric_baseline: node.metric_baseline,
          },
        })
      }
      if (i > 0) {
        const prevId = path.nodes[i - 1].metric_name || `node_${i - 1}`
        const currId = path.nodes[i].metric_name || `node_${i}`
        edgeSet.add(`${prevId}→${currId}`)
      }
    }
  }

  const nodes = Array.from(nodeMap.values())
  const edges = Array.from(edgeSet).map((key) => {
    const [source, target] = key.split('→')
    return {
      data: {
        id: `edge_${source}_${target}`,
        source,
        target,
        label: 'causes',
        impact: 'negative',
      },
    }
  })

  if (nodes.length < 2 || edges.length === 0) return null
  return { elements: { nodes, edges } }
}

/**
 * Renders a breadcrumb-style text trail of the causal path:
 *   Root Metric  ->  Next  ->  ...  ->  Incident Metric
 */
function PathBreadcrumb({ nodes }) {
  if (!nodes?.length) return null

  return (
    <div className="flex flex-wrap items-center gap-1 text-xs text-gray-600 mb-3">
      {nodes.map((node, idx) => {
        const label = (node.metric_name || '')
          .replace(/_/g, ' ')
          .replace(/\b\w/g, (l) => l.toUpperCase())
        const impact = nodeImpact(node.metric_zscore)
        const colorMap = {
          negative: 'text-red-600 font-semibold',
          positive: 'text-emerald-600 font-semibold',
          neutral: 'text-indigo-600 font-semibold',
        }
        return (
          <span key={idx} className="flex items-center gap-1">
            <span className={colorMap[impact]}>{label}</span>
            {idx < nodes.length - 1 && (
              <ChevronRight className="w-3 h-3 text-gray-400 flex-shrink-0" />
            )}
          </span>
        )
      })}
    </div>
  )
}

/**
 * A single interactive causal path card. Clicking the header toggles an
 * expanded view that shows the 3-D mini-graph of the chain.
 */
function CausalPathCard({ path, idx, isSelected, onToggle }) {
  const pathGraph = useMemo(() => buildPathGraph(path), [path])
  const scorePercent = ((path.overall_score ?? 0) * 100).toFixed(0)

  return (
    <div
      className={[
        'rounded-lg border transition-all duration-200',
        isSelected
          ? 'border-primary-400 border-l-4 bg-white shadow-sm'
          : 'border-gray-200 bg-gray-50 opacity-80 hover:opacity-100 hover:border-gray-300',
      ].join(' ')}
    >
      {/* Clickable header row */}
      <button
        type="button"
        className="w-full text-left px-4 py-3 flex items-center justify-between group focus:outline-none"
        onClick={() => onToggle(idx)}
        aria-expanded={isSelected}
        aria-controls={`rca-graph-${idx}`}
      >
        <div className="flex items-center gap-2 min-w-0">
          {/* Graph icon — subtle visual affordance that this row is interactive */}
          <GitFork
            className={[
              'w-4 h-4 flex-shrink-0 transition-colors',
              isSelected ? 'text-primary-500' : 'text-gray-400 group-hover:text-primary-400',
            ].join(' ')}
          />
          <span
            className={[
              'text-sm font-medium',
              isSelected ? 'text-primary-700' : 'text-gray-700',
            ].join(' ')}
          >
            Causal Path #{idx + 1}
          </span>
          {path.nodes?.length > 0 && (
            <span className="hidden sm:inline text-xs text-gray-400 truncate max-w-xs">
              — {path.nodes.length} metric{path.nodes.length !== 1 ? 's' : ''}
            </span>
          )}
        </div>

        <div className="flex items-center gap-3 flex-shrink-0 ml-3">
          <span
            className={[
              'text-sm font-semibold',
              isSelected ? 'text-primary-600' : 'text-gray-600',
            ].join(' ')}
          >
            Score: {scorePercent}%
          </span>
          {isSelected ? (
            <ChevronDown className="w-4 h-4 text-primary-500" />
          ) : (
            <ChevronRight className="w-4 h-4 text-gray-400 group-hover:text-primary-400 transition-colors" />
          )}
        </div>
      </button>

      {/* Node list — always visible as a compact summary */}
      {path.nodes?.length > 0 && (
        <div className="px-4 pb-3 space-y-2">
          {path.nodes.map((node, nIdx) => {
            const impact = nodeImpact(node.metric_zscore)
            const borderColorMap = {
              negative: 'border-red-400',
              positive: 'border-emerald-400',
              neutral: 'border-primary-300',
            }
            return (
              <div key={nIdx} className={`pl-4 border-l-2 ${borderColorMap[impact]}`}>
                <div className="flex items-center justify-between flex-wrap gap-x-4 gap-y-0.5">
                  <span className="font-medium text-sm text-gray-900">
                    {(node.metric_name || '').replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                  </span>
                  <span className="text-xs text-gray-500">
                    z:&nbsp;{node.metric_zscore?.toFixed(1)}&#963;&nbsp;&nbsp;contribution:&nbsp;
                    {((node.contribution_score ?? 0) * 100).toFixed(0)}%
                  </span>
                </div>
                <p className="text-xs text-gray-500 mt-0.5">
                  Value:&nbsp;{node.metric_value?.toFixed(2)}&nbsp;
                  <span className="text-gray-400">(baseline: {node.metric_baseline?.toFixed(2)})</span>
                </p>
              </div>
            )
          })}
        </div>
      )}

      {/* Expanded section — breadcrumb + 3-D mini-graph */}
      {isSelected && (
        <div
          id={`rca-graph-${idx}`}
          className="border-t border-primary-100 px-4 pt-4 pb-4"
        >
          <p className="text-xs font-semibold text-primary-700 uppercase tracking-wider mb-2">
            Causal Chain &mdash; Path #{idx + 1}
          </p>
          <PathBreadcrumb nodes={path.nodes} />
          {pathGraph ? (
            <Graph3D
              graphData={pathGraph}
              height={300}
              title={`Causal Chain — Path #${idx + 1}`}
              legendMode="forecast"
            />
          ) : (
            <div className="py-8 text-center text-sm text-gray-400">
              Not enough nodes to render a graph for this path.
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function IncidentDetail() {
  const { id: incidentId } = useParams()
  const [loading, setLoading] = useState(true)
  const [data, setData] = useState(null)
  const [cascadeData, setCascadeData] = useState(null)
  const [graphLayer, setGraphLayer] = useState('entity')
  const [selectedPathIdx, setSelectedPathIdx] = useState(null)
  const [error, setError] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [analyzeStatus, setAnalyzeStatus] = useState('')

  const fetchIncidentDetail = useCallback(async () => {
    try {
      const [incidentResult, cascadeResult] = await Promise.all([
        api.incidents.get(incidentId),
        api.cascades.getBlastRadius(incidentId).catch(() => null),
      ])
      setData(incidentResult)
      setCascadeData(cascadeResult)
    } catch (err) {
      console.error('Failed to fetch incident:', err)
      setError('Incident not found')
    } finally {
      setLoading(false)
    }
  }, [incidentId])

  const handleAnalyze = useCallback(async () => {
    setAnalyzing(true)
    setAnalyzeStatus('Finding root cause...')
    try {
      await api.incidents.analyze(incidentId)
      setAnalyzeStatus('Refreshing results...')
      await fetchIncidentDetail()
      setAnalyzeStatus('')
    } catch (err) {
      console.error('Analysis failed:', err)
      setAnalyzeStatus('Analysis failed. Please try again.')
    } finally {
      setAnalyzing(false)
    }
  }, [incidentId, fetchIncidentDetail])

  useEffect(() => {
    fetchIncidentDetail()
  }, [fetchIncidentDetail])

  /**
   * Toggle selection: clicking an already-selected path deselects it.
   */
  const handlePathToggle = useCallback((idx) => {
    setSelectedPathIdx((prev) => (prev === idx ? null : idx))
  }, [])

  // Must be before any early returns (rules of hooks)
  const mergedCausalGraph = useMemo(
    () => buildMergedCausalGraph(data?.causal_chain),
    [data?.causal_chain]
  )

  if (loading) {
    return <LoadingState message="Loading incident details..." />
  }

  if (error || !data) {
    return (
      <div className="text-center py-12">
        <AlertTriangle className="w-12 h-12 text-gray-400 mx-auto mb-3" />
        <p className="text-gray-600">{error || 'Incident not found'}</p>
      </div>
    )
  }

  const incident = data.incident
  const causalChain = data.causal_chain
  const blastRadius = data.blast_radius

  if (!incident) return <div>Incident data missing</div>

  const typeDisplay = (incident.incident_type || '').replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())
  const totalAtRisk = blastRadius
    ? (blastRadius.estimated_revenue_exposure || 0) + (blastRadius.estimated_refund_exposure || 0)
    : 0

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{typeDisplay} Incident</h1>
          {totalAtRisk > 0 && (
            <p className="mt-2 text-lg font-semibold text-red-600">
              ${totalAtRisk.toLocaleString()} at risk
            </p>
          )}
          <div className="flex items-center space-x-3 mt-2">
            <SeverityBadge severity={incident.severity} />
            <ConfidenceBadge confidence={incident.confidence} />
            <span className="text-sm text-gray-600">
              Detected {new Date(incident.detected_at).toLocaleString()}
            </span>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          {!causalChain && (
            <button
              onClick={handleAnalyze}
              disabled={analyzing}
              className="btn bg-amber-600 hover:bg-amber-700 text-white flex items-center space-x-2 disabled:opacity-50"
            >
              {analyzing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
              <span>{analyzing ? analyzeStatus || 'Analyzing...' : 'Analyze Root Cause'}</span>
            </button>
          )}
          <Link
            to={`/incidents/${incidentId}/postmortem`}
            className="btn btn-primary flex items-center space-x-2"
          >
            <FileText className="w-4 h-4" />
            <span>View Postmortem</span>
          </Link>
        </div>
      </div>

      {/* Plain-English severity explanation */}
      {explainSeverity(incident.severity) && (
        <div className={`px-4 py-3 rounded-lg text-sm ${
          incident.severity === 'critical' ? 'bg-red-50 text-red-800 border border-red-200' :
          incident.severity === 'high' ? 'bg-orange-50 text-orange-800 border border-orange-200' :
          'bg-yellow-50 text-yellow-800 border border-yellow-200'
        }`}>
          <p className="font-medium">{explainSeverity(incident.severity)}</p>
          {explainConfidence(incident.confidence) && (
            <p className="mt-1 opacity-80">{explainConfidence(incident.confidence)}</p>
          )}
        </div>
      )}

      {/* Metric Summary — leading indicators */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card">
          <p className="text-xs text-gray-500 uppercase">Primary Metric</p>
          <p className="text-lg font-bold text-gray-900 mt-1">
            {(incident.primary_metric || '').replace(/_/g, ' ')}
          </p>
          {incident.primary_metric_value != null && incident.primary_metric_baseline != null && (
            <p className="text-xs text-gray-500 mt-1">
              {explainMetric(incident.primary_metric, incident.primary_metric_value, incident.primary_metric_baseline)}
            </p>
          )}
        </div>
        <div className="card">
          <p className="text-xs text-gray-500 uppercase">How Unusual</p>
          <p className="text-lg font-bold text-red-600 mt-1">
            {incident.primary_metric_zscore?.toFixed(1)}x away from normal
          </p>
          <p className="text-xs text-gray-500 mt-1">
            {explainZScore(incident.primary_metric_zscore)}
          </p>
        </div>
        <div className="card">
          <p className="text-xs text-gray-500 uppercase">Current Value</p>
          <p className="text-lg font-bold text-gray-900 mt-1">
            {incident.primary_metric_value?.toLocaleString(undefined, { maximumFractionDigits: 2 })}
          </p>
        </div>
        <div className="card">
          <p className="text-xs text-gray-500 uppercase">Normal Value</p>
          <p className="text-lg font-bold text-gray-500 mt-1">
            {incident.primary_metric_baseline?.toLocaleString(undefined, { maximumFractionDigits: 2 })}
          </p>
        </div>
      </div>

      {/* Incident Explanation — what caused it, what it caused, what it will cause */}
      {(causalChain?.paths?.length > 0 || blastRadius?.narrative || blastRadius?.downstream_incidents_triggered?.length > 0 || (blastRadius && (blastRadius.customers_affected > 0 || blastRadius.orders_affected > 0 || blastRadius.estimated_revenue_exposure > 0)) || (blastRadius?.estimated_churn_exposure ?? 0) > 0 || incident?.primary_metric) && (
        <div className="card border-l-4 border-l-primary-500 bg-gradient-to-r from-slate-50 to-white">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Info className="w-5 h-5 text-primary-600" />
            Incident explanation
          </h2>
          <div className="space-y-5">
            {/* What caused this incident */}
            {(causalChain?.paths?.length > 0 || incident?.primary_metric) && (
              <div>
                <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wider mb-2 flex items-center gap-2">
                  <TrendingDown className="w-4 h-4 text-amber-600" />
                  What caused this
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  {causalChain?.paths?.length > 0 ? (
                    <>
                      Root cause traced to{' '}
                      <span className="font-semibold text-amber-800">
                        {(causalChain.paths[0].nodes[0]?.metric_name || '')
                          .replace(/_/g, ' ')
                          .replace(/\b\w/g, (l) => l.toUpperCase())}
                      </span>
                      {causalChain.paths[0].nodes.length > 1 && (
                        <>
                          , propagating through{' '}
                          {causalChain.paths[0].nodes
                            .slice(1, -1)
                            .map((n) => (n.metric_name || '').replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()))
                            .join(' → ')}
                          , to the incident metric{' '}
                          <span className="font-semibold text-gray-900">
                            {(causalChain.paths[0].nodes[causalChain.paths[0].nodes.length - 1]?.metric_name || '')
                              .replace(/_/g, ' ')
                              .replace(/\b\w/g, (l) => l.toUpperCase())}
                          </span>
                        </>
                      )}
                      . Expand the causal paths below to see the full chain and contribution scores.
                    </>
                  ) : (
                    <>
                      Primary metric{' '}
                      <span className="font-semibold text-amber-800">
                        {(incident.primary_metric || '').replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                      </span>
                      {' '}deviated significantly (z-score: {incident.primary_metric_zscore?.toFixed(1) ?? '—'}).
                      Run analysis to generate root cause paths.
                    </>
                  )}
                </p>
              </div>
            )}

            {/* What it has caused — current impact */}
            {(blastRadius?.narrative || (blastRadius && (blastRadius.customers_affected > 0 || blastRadius.orders_affected > 0 || blastRadius.estimated_revenue_exposure > 0))) && (
              <div>
                <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wider mb-2 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-red-600" />
                  What it has caused
                </h3>
                {blastRadius.narrative ? (
                  <p className="text-gray-700 leading-relaxed">{blastRadius.narrative}</p>
                ) : (
                  <p className="text-gray-700 leading-relaxed">
                    This incident has affected{' '}
                    {blastRadius.customers_affected > 0 && (
                      <span className="font-medium">{blastRadius.customers_affected.toLocaleString()} customers</span>
                    )}
                    {blastRadius.customers_affected > 0 && blastRadius.orders_affected > 0 && ', '}
                    {blastRadius.orders_affected > 0 && (
                      <span className="font-medium">{blastRadius.orders_affected.toLocaleString()} orders</span>
                    )}
                    {(blastRadius.estimated_revenue_exposure > 0 || blastRadius.estimated_refund_exposure > 0) && (
                      <>
                        , with{' '}
                        <span className="font-semibold text-red-700">
                          $
                          {(
                            (blastRadius.estimated_revenue_exposure || 0) +
                            (blastRadius.estimated_refund_exposure || 0)
                          ).toLocaleString()}{' '}
                          at risk
                        </span>
                      </>
                    )}
                    .
                  </p>
                )}
              </div>
            )}

            {/* What it will cause — downstream / cascade risk */}
            {(blastRadius?.downstream_incidents_triggered?.length > 0 || (blastRadius?.estimated_churn_exposure ?? 0) > 0) && (
              <div>
                <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wider mb-2 flex items-center gap-2">
                  <GitFork className="w-4 h-4 text-purple-600" />
                  What it will cause
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  {blastRadius?.downstream_incidents_triggered?.length > 0 && (
                    <>
                      This incident has triggered{' '}
                      <span className="font-semibold text-purple-800">
                        {blastRadius.downstream_incidents_triggered.length} downstream incident
                        {blastRadius.downstream_incidents_triggered.length !== 1 ? 's' : ''}
                      </span>
                      , creating cascade risk.{' '}
                    </>
                  )}
                  {(blastRadius?.estimated_churn_exposure ?? 0) > 0 && (
                    <>
                      <span className="font-semibold text-purple-800">
                        {blastRadius.estimated_churn_exposure.toLocaleString()} customers
                      </span>{' '}
                      {blastRadius?.downstream_incidents_triggered?.length > 0 ? 'are also ' : 'are '}at risk of churning.{' '}
                    </>
                  )}
                  Without intervention, the impact will spread. Early action can prevent downstream revenue loss and support load.
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Dual-layer 3D Graph: Entity impact + Causal metric chain */}
      {(cascadeData?.graph || cascadeData?.causal_graph || mergedCausalGraph) && (
        <div className="card p-0 overflow-hidden w-full">
          <div className="flex border-b border-gray-200">
            <button
              type="button"
              onClick={() => setGraphLayer('entity')}
              className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                graphLayer === 'entity'
                  ? 'bg-primary-50 text-primary-700 border-b-2 border-primary-600'
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              Entity impact
            </button>
            <button
              type="button"
              onClick={() => setGraphLayer('causal')}
              className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                graphLayer === 'causal'
                  ? 'bg-primary-50 text-primary-700 border-b-2 border-primary-600'
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              Causal metric chain
            </button>
          </div>
          {graphLayer === 'entity' && cascadeData?.graph && (
            <Graph3D graphData={cascadeData.graph} height={520} title="3D Entity Blast Radius" />
          )}
          {graphLayer === 'causal' && (mergedCausalGraph || cascadeData?.causal_graph) && (
            <Graph3D
              graphData={mergedCausalGraph || cascadeData.causal_graph}
              height={520}
              title="3D Causal Metric Chain — Root cause → incident"
              legendMode="forecast"
            />
          )}
          {graphLayer === 'causal' && !mergedCausalGraph && !cascadeData?.causal_graph && (
            <div className="py-16 text-center text-gray-500">
              No causal chain data yet. Run analysis to generate RCA paths.
            </div>
          )}
        </div>
      )}

      {/* Blast Radius Impact — $ at risk front and center with premium design */}
      {blastRadius && (
        <div className="card-premium">
          <h2 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
            <div className="w-1 h-6 bg-gradient-to-b from-red-500 to-purple-600 rounded-full" />
            Impact at a glance
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-5">
            <div className="group relative overflow-hidden p-5 bg-gradient-to-br from-red-500 to-red-600 rounded-2xl shadow-glow-danger hover:shadow-xl transition-all duration-300 hover:scale-105">
              <div className="absolute top-0 right-0 w-24 h-24 bg-white/10 rounded-full -mr-12 -mt-12 blur-2xl" />
              <div className="relative">
                <div className="flex items-center space-x-2 mb-3">
                  <div className="p-2 bg-white/20 rounded-lg backdrop-blur-sm">
                    <Users className="w-5 h-5 text-white" />
                  </div>
                  <span className="text-xs font-bold text-red-100 uppercase tracking-wider">Customers</span>
                </div>
                <p className="text-3xl font-black text-white tracking-tight">
                  {blastRadius.customers_affected?.toLocaleString()}
                </p>
              </div>
            </div>

            <div className="group relative overflow-hidden p-5 bg-gradient-to-br from-orange-500 to-orange-600 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105">
              <div className="absolute top-0 right-0 w-24 h-24 bg-white/10 rounded-full -mr-12 -mt-12 blur-2xl" />
              <div className="relative">
                <div className="flex items-center space-x-2 mb-3">
                  <div className="p-2 bg-white/20 rounded-lg backdrop-blur-sm">
                    <DollarSign className="w-5 h-5 text-white" />
                  </div>
                  <span className="text-xs font-bold text-orange-100 uppercase tracking-wider">Revenue Risk</span>
                </div>
                <p className="text-3xl font-black text-white tracking-tight">
                  ${blastRadius.estimated_revenue_exposure?.toLocaleString()}
                </p>
              </div>
            </div>

            <div className="group relative overflow-hidden p-5 bg-gradient-to-br from-amber-500 to-yellow-600 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105">
              <div className="absolute top-0 right-0 w-24 h-24 bg-white/10 rounded-full -mr-12 -mt-12 blur-2xl" />
              <div className="relative">
                <div className="flex items-center space-x-2 mb-3">
                  <div className="p-2 bg-white/20 rounded-lg backdrop-blur-sm">
                    <DollarSign className="w-5 h-5 text-white" />
                  </div>
                  <span className="text-xs font-bold text-yellow-100 uppercase tracking-wider">Refund Risk</span>
                </div>
                <p className="text-3xl font-black text-white tracking-tight">
                  ${blastRadius.estimated_refund_exposure?.toLocaleString()}
                </p>
              </div>
            </div>

            <div className="group relative overflow-hidden p-5 bg-gradient-to-br from-purple-500 to-purple-600 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105">
              <div className="absolute top-0 right-0 w-24 h-24 bg-white/10 rounded-full -mr-12 -mt-12 blur-2xl" />
              <div className="relative">
                <div className="flex items-center space-x-2 mb-3">
                  <div className="p-2 bg-white/20 rounded-lg backdrop-blur-sm">
                    <TrendingUp className="w-5 h-5 text-white" />
                  </div>
                  <span className="text-xs font-bold text-purple-100 uppercase tracking-wider">Churn Risk</span>
                </div>
                <p className="text-3xl font-black text-white tracking-tight">
                  {blastRadius.estimated_churn_exposure}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Root Cause Analysis — interactive causal paths */}
      {causalChain && causalChain.paths?.length > 0 && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">Root Cause Analysis</h2>
            <p className="text-xs text-gray-400 hidden sm:block">
              Click a path to visualise the causal chain
            </p>
          </div>
          <div className="space-y-3">
            {causalChain.paths.map((path, idx) => (
              <CausalPathCard
                key={idx}
                path={path}
                idx={idx}
                isSelected={selectedPathIdx === idx}
                onToggle={handlePathToggle}
              />
            ))}
          </div>
        </div>
      )}

      {/* Detection Methods */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-3">Detection Details</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-xs text-gray-500 uppercase">Methods</p>
            <div className="flex flex-wrap gap-1 mt-1">
              {incident.detection_methods?.map((method) => (
                <span
                  key={method}
                  className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-700"
                >
                  {method.replace(/_/g, ' ')}
                </span>
              ))}
            </div>
          </div>
          <div>
            <p className="text-xs text-gray-500 uppercase">Evidence Events</p>
            <p className="text-lg font-bold text-gray-900 mt-1">{incident.evidence_event_count}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 uppercase">Data Quality</p>
            <p className="text-lg font-bold text-gray-900 mt-1">
              {(incident.data_quality_score * 100).toFixed(0)}%
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500 uppercase">Status</p>
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800 capitalize mt-1">
              {incident.status}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
