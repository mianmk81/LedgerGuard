import { useState, useEffect, useMemo, useCallback } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { api } from '../api/client'
import LoadingState from '../components/common/LoadingState'
import Sparkline from '../components/common/Sparkline'
import SeverityBadge from '../components/common/SeverityBadge'
import Graph3D from '../components/graph/Graph3D'
import { TrendingUp, TrendingDown, Minus, AlertTriangle, AlertCircle, Activity, Shield, BarChart3, Users, ChevronRight, ChevronDown, ChevronUp, X, Zap, Clock, ArrowRight } from 'lucide-react'

const DOMAIN_ICONS = {
  financial: BarChart3,
  operational: Activity,
  customer: Users,
}

const GRADE_COLORS = {
  A: 'text-green-600 bg-green-50 border-green-200',
  B: 'text-blue-600 bg-blue-50 border-blue-200',
  C: 'text-yellow-600 bg-yellow-50 border-yellow-200',
  D: 'text-orange-600 bg-orange-50 border-orange-200',
  F: 'text-red-600 bg-red-50 border-red-200',
}

const INITIAL_METRICS_COUNT = 6

const formatMetricValue = (value) => {
  if (typeof value !== 'number') return '\u2014'
  if (value < 1 && value > 0) return (value * 100).toFixed(1) + '%'
  return value.toLocaleString(undefined, { maximumFractionDigits: 1 })
}

export default function HealthDashboard() {
  const location = useLocation()
  const [loading, setLoading] = useState(true)
  const [dashboardMetrics, setDashboardMetrics] = useState(null)
  const [healthScore, setHealthScore] = useState(null)
  const [incidents, setIncidents] = useState([])
  const [reportsData, setReportsData] = useState(null)
  const [futureScoreData, setFutureScoreData] = useState(null)
  const [warningsData, setWarningsData] = useState(null)
  const [futureScoreDetailOpen, setFutureScoreDetailOpen] = useState(false)
  const [showAllMetrics, setShowAllMetrics] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchDashboardData()
  }, [])

  useEffect(() => {
    if (location.hash === '#future-score') {
      const el = document.getElementById('future-score')
      el?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [futureScoreData, location.hash])

  // React-specialist agent: useCallback for stable function references
  const fetchDashboardData = useCallback(async () => {
    try {
      setError(null)
      const [metricsData, healthData, reportsRes, futureRes, incidentsData, warningsRes] = await Promise.all([
        api.metrics.getDashboard('30d'),
        api.metrics.getHealthScore(),
        api.dashboard.reports(5).catch(() => null),
        api.dashboard.futureScore(30).catch(() => null),
        api.incidents.list({ page_size: 5 }).catch(() => []),
        api.warnings.list().catch(() => null),
      ])

      setDashboardMetrics(metricsData)
      setHealthScore(healthData)
      const incList = reportsRes?.incidents ?? (Array.isArray(incidentsData) ? incidentsData : incidentsData?.data ?? [])
      setIncidents(Array.isArray(incList) ? incList : [])
      setReportsData(reportsRes?.data ?? reportsRes)
      setFutureScoreData(futureRes?.data ?? futureRes)
      const warnList = warningsRes?.data?.warnings ?? warningsRes?.warnings ?? []
      setWarningsData(Array.isArray(warnList) ? warnList : [])
    } catch (err) {
      console.error('Failed to fetch dashboard data:', err)
      setError(err.message || 'Failed to load dashboard data')
    } finally {
      setLoading(false)
    }
  }, [])

  if (loading) {
    return <LoadingState message="Loading dashboard..." />
  }

  if (error) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Dashboard</h1>
        </div>
        <div className="card border-red-200 bg-red-50/50">
          <div className="flex items-center space-x-3">
            <AlertCircle className="w-8 h-8 text-red-500 flex-shrink-0" />
            <div>
              <p className="font-medium text-red-800">Failed to load dashboard</p>
              <p className="text-sm text-red-600 mt-1">{error}</p>
              <button
                onClick={() => { setLoading(true); fetchDashboardData() }}
                className="mt-3 btn btn-secondary text-sm"
              >
                Retry
              </button>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Hero line — judges see this in 5 seconds */}
      <div className="rounded-2xl bg-gradient-to-r from-primary-600 via-primary-600 to-primary-700 px-6 py-5 text-white shadow-hover border border-primary-500/20 transition-shadow duration-300 hover:shadow-xl">
        <p className="text-xl font-bold tracking-tight md:text-2xl text-white/95">
          Know before it hurts.
        </p>
        <p className="mt-1.5 text-primary-100 text-sm md:text-base">
          We predict when you&apos;ll run out of cash — and trace exactly what&apos;s causing it. Your business health, decoded.
        </p>
        {futureScoreData && (
          <a
            href="#future-score"
            className="inline-flex items-center gap-2 mt-4 px-4 py-2 rounded-xl bg-white/15 hover:bg-white/25 border border-white/30 text-white font-medium text-sm transition-all"
          >
            See forecast — where your business is headed <ChevronDown className="w-4 h-4" />
          </a>
        )}
      </div>

      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Your business health, decoded</h1>
        <p className="text-slate-600 mt-1">Know before it hurts — financial anomalies, explained</p>
      </div>

      {/* Business Reliability Score - HERO VERSION */}
      {healthScore && (
        <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-primary-600 via-primary-700 to-primary-800 shadow-2xl border border-primary-500/20 animate-scale-in">
          {/* Animated background gradient */}
          <div className="absolute inset-0 bg-gradient-to-r from-primary-500/20 to-transparent animate-pulse" />

          <div className="relative px-8 py-10">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/10 backdrop-blur-sm border border-white/20 mb-4">
                  <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
                  <span className="text-xs font-semibold text-white/90 uppercase tracking-wider">Live Score</span>
                </div>
                <h2 className="text-2xl font-bold text-white mb-2">Business Reliability Score</h2>
                <p className="text-primary-100 max-w-md">
                  {healthScore.explanation || 'Real-time composite health across financial, operational, and customer domains'}
                </p>
              </div>

              <div className="text-right flex items-center gap-6">
                {/* Main Score - HUGE and prominent */}
                <div className="relative">
                  <div className="absolute -inset-4 bg-white/5 rounded-3xl blur-xl" />
                  <div className="relative bg-white/10 backdrop-blur-md rounded-2xl px-8 py-6 border border-white/20 shadow-2xl">
                    <div className="text-7xl font-black text-white tracking-tight leading-none">
                      {healthScore.overall_score?.toFixed(0) ?? '—'}
                    </div>
                    <div className="text-sm font-medium text-primary-100 mt-2 text-center">out of 100</div>
                  </div>
                </div>

                {/* Grade Badge - Premium style */}
                {healthScore.overall_grade && (
                  <div className="relative">
                    <div className="absolute -inset-2 bg-white/10 rounded-2xl blur-lg" />
                    <div className="relative w-20 h-20 rounded-2xl bg-white shadow-xl flex items-center justify-center border-4 border-white/30">
                      <span className="text-4xl font-black bg-gradient-to-br from-primary-600 to-primary-800 bg-clip-text text-transparent">
                        {healthScore.overall_grade}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>

          {/* Domain Breakdown */}
          {healthScore.domains && (
            <div className="grid grid-cols-3 gap-4 mt-6">
              {Object.entries(healthScore.domains).map(([domain, data]) => {
                const Icon = DOMAIN_ICONS[domain] || Activity
                return (
                  <div key={domain} className="p-4 bg-white/10 backdrop-blur-sm rounded-xl border border-white/20">
                    <div className="flex items-center space-x-2 mb-2">
                      <Icon className="w-4 h-4 text-primary-200" />
                      <span className="text-sm font-medium text-white/90 capitalize">{domain}</span>
                    </div>
                    <div className="flex items-baseline space-x-2">
                      <span className="text-2xl font-bold text-white">
                        {data.score?.toFixed(0) ?? '—'}
                      </span>
                      {data.grade && (
                        <span
                          className={`text-xs font-medium px-1.5 py-0.5 rounded ${
                            GRADE_COLORS[data.grade] || ''
                          }`}
                        >
                          {data.grade}
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-primary-200 mt-1">
                      {data.explanation || `${data.metrics_available ?? 0}/${data.metrics_total ?? 0} metrics`}
                    </div>
                  </div>
                )
              })}
            </div>
          )}

          {/* Trend */}
          {healthScore.trend && (
            <div className="mt-4 flex items-center space-x-2 text-sm">
              {healthScore.trend.direction === 'improving' ? (
                <TrendingUp className="w-4 h-4 text-emerald-400" />
              ) : healthScore.trend.direction === 'degrading' ? (
                <TrendingDown className="w-4 h-4 text-rose-400" />
              ) : (
                <Minus className="w-4 h-4 text-white/50" />
              )}
              <span className="text-primary-100">
                Trend: {healthScore.trend.direction} vs previous period
              </span>
            </div>
          )}
          </div>
        </div>
      )}

      {/* Predicted Issues — What our ML models see coming */}
      {warningsData && warningsData.length > 0 && (
        <div className="card animate-fade-in border-amber-200/60 bg-gradient-to-br from-amber-50/30 to-white">
          <div className="flex items-center gap-3 mb-5">
            <div className="p-2.5 rounded-xl bg-amber-100 border border-amber-200">
              <Zap className="w-5 h-5 text-amber-600" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-900">Predicted Issues</h2>
              <p className="text-sm text-slate-500">Our models detected {warningsData.length} issue{warningsData.length > 1 ? 's' : ''} likely to happen soon</p>
            </div>
          </div>

          <div className="space-y-4">
            {warningsData.slice(0, 4).map((w) => {
              const severityColors = {
                critical: 'border-red-300 bg-red-50/60',
                high: 'border-orange-300 bg-orange-50/60',
                medium: 'border-amber-200 bg-amber-50/40',
                low: 'border-slate-200 bg-slate-50/40',
              }
              const severityDotColors = {
                critical: 'bg-red-500',
                high: 'bg-orange-500',
                medium: 'bg-amber-500',
                low: 'bg-slate-400',
              }
              const cardStyle = severityColors[w.severity] || severityColors.medium
              const dotColor = severityDotColors[w.severity] || severityDotColors.medium
              const daysInt = w.days_to_threshold != null ? Math.round(w.days_to_threshold) : null

              return (
                <div key={w.warning_id} className={`p-4 rounded-xl border ${cardStyle} transition-all hover:shadow-md`}>
                  {/* Header row: incident name + severity + timeline */}
                  <div className="flex items-start justify-between gap-3 mb-2">
                    <div className="flex items-center gap-2.5 min-w-0">
                      <div className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${dotColor}`} />
                      <h3 className="font-semibold text-slate-900 truncate">
                        {w.incident_label || (w.incident_type || 'Issue').replace(/_/g, ' ')}
                      </h3>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <SeverityBadge severity={w.severity} />
                      {daysInt != null && (
                        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-white/80 border border-slate-200 text-xs font-semibold text-slate-700">
                          <Clock className="w-3 h-3" />
                          {daysInt <= 1 ? 'Tomorrow' : `~${daysInt} days`}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Plain-English prediction */}
                  <p className="text-sm text-slate-700 mb-3 leading-relaxed">
                    {w.prediction_summary}
                  </p>

                  {/* Causal chain visualization */}
                  {w.forward_chain && w.forward_chain.length > 1 && (
                    <div className="flex items-center flex-wrap gap-1.5 mb-3">
                      <span className="text-xs font-medium text-slate-500 mr-1">Root cause chain:</span>
                      {w.forward_chain.map((step, i) => (
                        <span key={i} className="flex items-center gap-1">
                          <span className={`px-2 py-0.5 rounded-md text-xs font-medium ${
                            i === 0
                              ? 'bg-red-100 text-red-800 border border-red-200'
                              : i === w.forward_chain.length - 1
                                ? 'bg-amber-100 text-amber-800 border border-amber-200'
                                : 'bg-slate-100 text-slate-700 border border-slate-200'
                          }`}>
                            {step.replace(/_/g, ' ')}
                          </span>
                          {i < w.forward_chain.length - 1 && (
                            <ArrowRight className="w-3 h-3 text-slate-400 flex-shrink-0" />
                          )}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Prevention */}
                  {w.prevention_steps && (
                    <div className="flex items-start gap-2 p-2.5 rounded-lg bg-green-50/80 border border-green-200/60">
                      <Shield className="w-3.5 h-3.5 text-green-600 mt-0.5 flex-shrink-0" />
                      <p className="text-xs text-green-800 leading-relaxed">{w.prevention_steps}</p>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Reports Card — Current/recent reports + score + 3D causal (what caused, how, where, when) */}
      {reportsData && (
        <div className="card animate-fade-in">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-slate-900">Current & Recent Reports</h2>
            <Link to="/incidents" className="text-sm text-primary-600 hover:text-primary-700 font-medium flex items-center gap-1">
              View all <ChevronRight className="w-4 h-4" />
            </Link>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="space-y-3">
              {reportsData.score && (
                <div className="flex items-center gap-3">
                  <div className="text-3xl font-bold text-primary-600">
                    {reportsData.score.overall_score?.toFixed(0) ?? '—'}
                  </div>
                  <div className={`text-xl font-bold px-2 py-1 rounded ${
                    GRADE_COLORS[reportsData.score.overall_grade] || GRADE_COLORS.C
                  }`}>
                    {reportsData.score.overall_grade || '—'}
                  </div>
                </div>
              )}
              <p className="text-sm text-slate-500">Health score from current metrics</p>
              {reportsData.incidents?.length > 0 ? (
                <div className="space-y-2">
                  {reportsData.incidents.slice(0, 4).map((inc) => (
                    <Link
                      key={inc.incident_id}
                      to={`/incidents/${inc.incident_id}`}
                      className="block p-3 rounded-xl border border-slate-200 hover:border-primary-300 hover:bg-primary-50/50 hover:shadow-sm transition-all duration-200 text-sm"
                    >
                      <div className="flex justify-between items-center">
                        <span className="font-medium truncate">
                          {(inc.incident_type || '').replace(/_/g, ' ')}
                        </span>
                        <SeverityBadge severity={inc.severity} />
                      </div>
                      <span className="text-xs text-slate-500">
                        {inc.detected_at ? new Date(inc.detected_at).toLocaleDateString() : ''}
                      </span>
                    </Link>
                  ))}
                </div>
              ) : (
                <div className="flex items-center gap-2 text-green-600">
                  <Shield className="w-5 h-5" />
                  <span className="text-sm font-medium">No incidents detected</span>
                </div>
              )}
            </div>
            <div className="lg:col-span-2 w-full min-w-0">
              <Graph3D
                graphData={reportsData.causal_graph}
                height={280}
                title="3D Causal Chain — What caused, how, where, when"
              />
            </div>
          </div>
        </div>
      )}

      {/* Future Score Card — Predicted rating + WHY + drivers + 3D graph */}
      {futureScoreData && (
        <div id="future-score" className="card scroll-mt-24">
          <h2 className="text-lg font-semibold text-slate-900 mb-4">Where your business is headed</h2>

          {/* WHY & WHEN — inline, not hidden in modal */}
          <div className="mb-4 p-4 rounded-xl bg-slate-50 border border-slate-200">
            <h3 className="text-sm font-semibold text-slate-700 mb-2">Why & when</h3>
            <p className="text-slate-800">{futureScoreData.why_summary}</p>
            {futureScoreData.when_hit && (
              <p className="mt-2 text-sm font-medium text-slate-700">{futureScoreData.when_hit}</p>
            )}
          </div>

          {/* Drivers: red = pulling down, green = driving up */}
          {(() => {
            const nodes = futureScoreData.causal_graph?.elements?.nodes || []
            const driversDown = nodes.filter((n) => (n.data?.impact || n.impact) === 'negative')
            const driversUp = nodes.filter((n) => (n.data?.impact || n.impact) === 'positive')
            if (driversDown.length === 0 && driversUp.length === 0) return null
            return (
              <div className="mb-4 grid grid-cols-1 sm:grid-cols-2 gap-4">
                {driversDown.length > 0 && (
                  <div className="p-3 rounded-xl border border-red-200 bg-red-50/50">
                    <p className="text-xs font-semibold text-red-700 uppercase mb-2 flex items-center gap-2">
                      <TrendingDown className="w-3.5 h-3.5" /> Pulling score down
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {driversDown.map((n) => {
                        const label = n.data?.label || n.data?.id || n.id || '?'
                        return (
                          <span key={label} className="px-2 py-1 rounded-lg bg-red-100 text-red-800 text-sm font-medium">
                            {String(label).replace(/_/g, ' ')}
                          </span>
                        )
                      })}
                    </div>
                  </div>
                )}
                {driversUp.length > 0 && (
                  <div className="p-3 rounded-xl border border-green-200 bg-green-50/50">
                    <p className="text-xs font-semibold text-green-700 uppercase mb-2 flex items-center gap-2">
                      <TrendingUp className="w-3.5 h-3.5" /> Driving score up
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {driversUp.map((n) => {
                        const label = n.data?.label || n.data?.id || n.id || '?'
                        return (
                          <span key={label} className="px-2 py-1 rounded-lg bg-green-100 text-green-800 text-sm font-medium">
                            {String(label).replace(/_/g, ' ')}
                          </span>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>
            )
          })()}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="space-y-2">
              <p className="text-sm text-slate-500">Projected in {futureScoreData.projection_days} days</p>
              <div className="flex items-center gap-3">
                <div className={`w-16 h-16 rounded-xl border-2 flex items-center justify-center text-2xl font-bold ${
                  GRADE_COLORS[futureScoreData.projected_grade] || GRADE_COLORS.C
                }`}>
                  {futureScoreData.projected_grade || '—'}
                </div>
                <div>
                  <p className="text-2xl font-bold text-slate-900">
                    {futureScoreData.projected_score?.toFixed(0) ?? '—'}
                  </p>
                  <p className="text-xs text-slate-500">from {futureScoreData.current_grade} today</p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setFutureScoreDetailOpen(true)}
                className="btn btn-secondary text-sm mt-2 w-full flex items-center justify-center gap-2"
              >
                Full detail — path & what to do <ChevronRight className="w-4 h-4" />
              </button>
            </div>
            <div className="lg:col-span-2 w-full min-w-0">
              <Graph3D
                graphData={futureScoreData.causal_graph}
                height={260}
                title="3D Causal Chain — red = down, green = up"
                legendMode="forecast"
              />
            </div>
          </div>

          {/* Future Score Detail Modal */}
          {futureScoreDetailOpen && (
            <div
              className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
              role="dialog"
              aria-modal="true"
              aria-labelledby="future-score-dialog-title"
              onClick={(e) => { if (e.target === e.currentTarget) setFutureScoreDetailOpen(false) }}
              onKeyDown={(e) => { if (e.key === 'Escape') setFutureScoreDetailOpen(false) }}
            >
              <div className="bg-white rounded-xl shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
                <div className="p-6 border-b flex items-center justify-between">
                  <h3 id="future-score-dialog-title" className="text-xl font-semibold text-slate-900">Future Score — Full Detail</h3>
                  <button
                    type="button"
                    onClick={() => setFutureScoreDetailOpen(false)}
                    className="p-2 rounded-lg hover:bg-slate-100"
                    aria-label="Close dialog"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
                <div className="p-6 space-y-6">
                  <div>
                    <h4 className="text-sm font-medium text-slate-500 uppercase mb-2">Why & how</h4>
                    <p className="text-slate-900">{futureScoreData.why_summary}</p>
                  </div>
                  {futureScoreData.when_hit && (
                    <div>
                      <h4 className="text-sm font-medium text-slate-500 uppercase mb-2">When it will affect you</h4>
                      <p className="text-slate-900 font-medium">{futureScoreData.when_hit}</p>
                    </div>
                  )}
                  {futureScoreData.path?.length > 0 && (
                    <div>
                      <h4 className="text-sm font-medium text-slate-500 uppercase mb-2">Path (steps in chain)</h4>
                      <div className="flex flex-wrap gap-1 items-center">
                        {futureScoreData.path.map((step, i) => (
                          <span key={i} className="flex items-center gap-1">
                            <span className="px-2 py-1 bg-slate-100 rounded text-sm">
                              {step.replace(/_/g, ' ')}
                            </span>
                            {i < futureScoreData.path.length - 1 && (
                              <ChevronRight className="w-3 h-3 text-slate-400 flex-shrink-0" />
                            )}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {futureScoreData.prevention_steps && (
                    <div>
                      <h4 className="text-sm font-medium text-slate-500 uppercase mb-2">What to do to stop it</h4>
                      <p className="text-slate-900 bg-green-50 border border-green-200 rounded-lg p-4">
                        {futureScoreData.prevention_steps}
                      </p>
                    </div>
                  )}
                  {futureScoreData.top_warning && (
                    <div className="p-4 bg-amber-50 border border-amber-200 rounded-lg">
                      <p className="text-sm font-medium text-amber-800">Leading indicator</p>
                      <p className="text-sm text-amber-700 mt-1">
                        {futureScoreData.top_warning.metric?.replace(/_/g, ' ')}: current {futureScoreData.top_warning.current_value?.toFixed(2)}
                        → projected {futureScoreData.top_warning.projected_value?.toFixed(2)}
                        {futureScoreData.top_warning.days_to_threshold != null && (
                          <> (~{Math.round(futureScoreData.top_warning.days_to_threshold)} days to threshold)</>
                        )}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Key Metrics Grid — Show 6 initially, expand for more */}
      {dashboardMetrics?.metrics && (
        <div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
            {(showAllMetrics ? dashboardMetrics.metrics : dashboardMetrics.metrics.slice(0, INITIAL_METRICS_COUNT)).map((metric) => (
            <div key={metric.metric_name} className="group relative overflow-hidden bg-gradient-to-br from-white to-slate-50/50 rounded-2xl shadow-card border border-slate-200/60 p-5 transition-all duration-300 hover:shadow-xl hover:-translate-y-1 hover:border-primary-200/60 cursor-pointer">
              {/* Hover gradient overlay */}
              <div className="absolute inset-0 bg-gradient-to-br from-primary-50/0 to-primary-100/0 group-hover:from-primary-50/30 group-hover:to-primary-100/10 transition-all duration-300 rounded-2xl" />

              <div className="relative flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider truncate mb-2">
                    {metric.metric_name.replace(/_/g, ' ')}
                  </p>
                  <p className="text-2xl font-black text-slate-900 tracking-tight">
                    {formatMetricValue(metric.current_value)}
                  </p>
                  <div className="flex items-center space-x-1.5 mt-2">
                    {metric.trend === 'up' ? (
                      <div className="p-1 bg-green-50 rounded-lg">
                        <TrendingUp className="w-3.5 h-3.5 text-green-600" />
                      </div>
                    ) : metric.trend === 'down' ? (
                      <div className="p-1 bg-red-50 rounded-lg">
                        <TrendingDown className="w-3.5 h-3.5 text-red-600" />
                      </div>
                    ) : (
                      <div className="p-1 bg-slate-50 rounded-lg">
                        <Minus className="w-3.5 h-3.5 text-slate-400" />
                      </div>
                    )}
                    <span
                      className={`text-sm font-bold ${
                        metric.change_percent >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}
                    >
                      {metric.change_percent >= 0 ? '+' : ''}
                      {metric.change_percent?.toFixed(1)}%
                    </span>
                  </div>
                </div>
                {metric.sparkline?.length > 0 && (
                  <div className="w-20 h-14 ml-3 opacity-80 group-hover:opacity-100 transition-opacity">
                    <Sparkline data={metric.sparkline} />
                  </div>
                )}
              </div>
              {metric.explanation && (
                <p className="relative text-xs text-slate-400 mt-2 leading-relaxed">{metric.explanation}</p>
              )}
              </div>
            ))}
          </div>
          {dashboardMetrics.metrics.length > INITIAL_METRICS_COUNT && (
            <button
              type="button"
              onClick={() => setShowAllMetrics(!showAllMetrics)}
              className="mt-3 flex items-center gap-1 text-sm text-primary-600 hover:text-primary-700 font-medium"
            >
              {showAllMetrics ? (
                <>
                  <ChevronUp className="w-4 h-4" />
                  Show fewer metrics
                </>
              ) : (
                <>
                  <ChevronDown className="w-4 h-4" />
                  Show {dashboardMetrics.metrics.length - INITIAL_METRICS_COUNT} more metrics
                </>
              )}
            </button>
          )}
        </div>
      )}

      {/* Recent Incidents */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-slate-900">Recent Incidents</h2>
          <Link to="/incidents" className="text-sm text-primary-600 hover:text-primary-700 font-medium hover:underline">
            View all
          </Link>
        </div>

        {incidents.length === 0 ? (
          <div className="text-center py-8">
            <Shield className="w-12 h-12 text-green-400 mx-auto mb-3" />
            <p className="text-slate-600 font-medium">No incidents detected</p>
            <p className="text-sm text-slate-500 mt-1">All systems operating normally</p>
          </div>
        ) : (
          <div className="space-y-3">
            {incidents.map((incident) => (
              <Link
                key={incident.incident_id}
                to={`/incidents/${incident.incident_id}`}
                className="block p-4 border border-slate-200 rounded-xl hover:border-primary-300 hover:bg-primary-50/30 hover:shadow-sm transition-all duration-200"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <AlertTriangle className="w-5 h-5 text-orange-500 flex-shrink-0" />
                    <div>
                      <p className="font-medium text-slate-900">
                        {(incident.incident_type || '').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </p>
                      <p className="text-sm text-slate-600 mt-0.5">
                        {incident.primary_metric?.replace(/_/g, ' ')} — z-score: {incident.primary_metric_zscore?.toFixed(1)}
                      </p>
                    </div>
                  </div>
                  <div className="text-right flex items-center space-x-3">
                    <SeverityBadge severity={incident.severity} />
                    <span className="text-xs text-slate-500">
                      {incident.detected_at ? new Date(incident.detected_at).toLocaleDateString() : ''}
                    </span>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
