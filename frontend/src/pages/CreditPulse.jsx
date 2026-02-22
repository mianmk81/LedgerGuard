import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import LoadingState from '../components/common/LoadingState'
import SeverityBadge from '../components/common/SeverityBadge'
import Graph3D from '../components/graph/Graph3D'
import {
  HeartIcon,
  BarChart3,
  Activity,
  Users,
  TrendingUp,
  TrendingDown,
  Minus,
  ArrowRightIcon,
  ArrowRight,
  AlertCircle,
  CheckCircle,
  AlertTriangle,
  Zap,
  Clock,
  Shield,
} from 'lucide-react'

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

const STATUS_STYLES = {
  healthy: 'text-green-600 bg-green-50 border-green-200',
  degraded: 'text-yellow-600 bg-yellow-50 border-yellow-200',
  warning: 'text-orange-600 bg-orange-50 border-orange-200',
  critical: 'text-red-600 bg-red-50 border-red-200',
  no_data: 'text-gray-500 bg-gray-50 border-gray-200',
}

export default function CreditPulse() {
  const [loading, setLoading] = useState(true)
  const [data, setData] = useState(null)
  const [warnings, setWarnings] = useState([])
  const [error, setError] = useState(null)

  // React-specialist agent: useCallback for stable function references
  const fetchCreditPulse = useCallback(async (lookbackDays = 7) => {
    try {
      setError(null)
      const [creditResult, warningsResult] = await Promise.all([
        api.creditPulse.get(lookbackDays),
        api.warnings.list().catch(() => ({ warnings: [], count: 0 })),
      ])
      setData(creditResult)
      const warnList = warningsResult?.data?.warnings ?? warningsResult?.warnings ?? []
      setWarnings(Array.isArray(warnList) ? warnList : [])
    } catch (err) {
      console.error('Failed to fetch Credit Pulse:', err)
      setError(err.message || 'Failed to load financial health score')
    } finally {
      setLoading(false)
    }
  }, [])

  const handleRetry = useCallback(() => {
    setLoading(true)
    setError(null)
    fetchCreditPulse()
  }, [fetchCreditPulse])

  useEffect(() => {
    fetchCreditPulse()
  }, [fetchCreditPulse])

  if (loading) {
    return <LoadingState message="Loading Credit Pulse..." />
  }

  if (error) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Credit Pulse</h1>
          <p className="text-gray-600 mt-1">Your financial health, explained</p>
        </div>
        <div className="card border-red-200 bg-red-50/50">
          <div className="flex items-center space-x-3">
            <AlertCircle className="w-8 h-8 text-red-500 flex-shrink-0" />
            <div>
              <p className="font-medium text-red-800">Failed to load Credit Pulse</p>
              <p className="text-sm text-red-600 mt-1">{error}</p>
              <button
                onClick={handleRetry}
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

  if (!data) {
    return null
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Credit Pulse</h1>
        <p className="text-gray-600 mt-1">
          Your financial health score — and why it&apos;s what it is
        </p>
      </div>

      {/* Hero Score Card - Premium Financial Health */}
      <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-emerald-500 via-emerald-600 to-teal-700 shadow-xl-premium">
        {/* Animated pulse effect */}
        <div className="absolute inset-0 bg-gradient-to-r from-emerald-400/20 to-transparent animate-pulse" />

        <div className="relative px-8 py-10">
          <div className="flex items-center justify-between flex-wrap gap-6">
            <div className="flex items-center space-x-5">
              <div className="relative">
                <div className="absolute inset-0 bg-white/30 rounded-2xl blur-lg" />
                <div className="relative w-20 h-20 rounded-2xl bg-white/20 backdrop-blur-sm border border-white/30 flex items-center justify-center">
                  <HeartIcon className="w-10 h-10 text-white drop-shadow-lg" />
                </div>
              </div>
              <div>
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/20 backdrop-blur-sm border border-white/30 mb-3">
                  <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                  <span className="text-xs font-bold text-white uppercase tracking-wider">Credit Pulse</span>
                </div>
                <h2 className="text-2xl font-bold text-white drop-shadow-md">Financial Health Score</h2>
                <p className="text-emerald-50 mt-1">
                  {data.explanation || `Based on ${data.lookback_days} days of business data`}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-8">
              {/* Massive Score */}
              <div className="relative">
                <div className="absolute -inset-6 bg-white/10 rounded-3xl blur-2xl" />
                <div className="relative bg-white/15 backdrop-blur-md rounded-3xl px-10 py-8 border border-white/30 shadow-2xl">
                  <div className="text-8xl font-black text-white tracking-tight leading-none drop-shadow-lg">
                    {typeof data.score === 'number' ? data.score.toFixed(0) : '—'}
                  </div>
                  <div className="text-sm font-semibold text-emerald-100 mt-3 text-center">out of 100</div>
                </div>
              </div>

              {/* Grade - Glass morphism */}
              {data.grade && (
                <div className="relative">
                  <div className="absolute -inset-3 bg-white/20 rounded-2xl blur-xl" />
                  <div className="relative w-24 h-24 rounded-2xl bg-white shadow-2xl flex items-center justify-center border-4 border-white/40">
                    <span className="text-5xl font-black bg-gradient-to-br from-emerald-600 to-teal-700 bg-clip-text text-transparent">
                      {data.grade}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>

        {/* Trend */}
        {data.trend && (
          <div className="mt-4 flex items-center space-x-2 text-sm pt-4 border-t border-gray-100">
            {data.trend.direction === 'improving' ? (
              <TrendingUp className="w-4 h-4 text-green-500" />
            ) : data.trend.direction === 'degrading' ? (
              <TrendingDown className="w-4 h-4 text-red-500" />
            ) : (
              <Minus className="w-4 h-4 text-gray-400" />
            )}
            <span className="text-gray-600">
              Trend: {data.trend.direction} vs previous period
            </span>
          </div>
        )}
        </div>
      </div>

      {/* Predicted Issues — Forward Prediction Warnings */}
      {warnings.length > 0 && (
        <div className="card animate-fade-in border-amber-200/60 bg-gradient-to-br from-amber-50/30 to-white">
          <div className="flex items-center gap-3 mb-5">
            <div className="p-2.5 rounded-xl bg-amber-100 border border-amber-200">
              <Zap className="w-5 h-5 text-amber-600" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-gray-900">Predicted Issues</h3>
              <p className="text-sm text-gray-500">
                Our models detected {warnings.length} issue{warnings.length > 1 ? 's' : ''} likely to happen soon
              </p>
            </div>
          </div>
          <div className="space-y-4">
            {warnings.map((w) => {
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
                  <div className="flex items-start justify-between gap-3 mb-2">
                    <div className="flex items-center gap-2.5 min-w-0">
                      <div className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${dotColor}`} />
                      <h4 className="font-semibold text-gray-900 truncate">
                        {w.incident_label || (w.incident_type || 'Issue').replace(/_/g, ' ')}
                      </h4>
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

                  <p className="text-sm text-gray-700 mb-3 leading-relaxed">
                    {w.prediction_summary || w.prevention_steps}
                  </p>

                  {w.forward_chain && w.forward_chain.length > 1 && (
                    <div className="flex items-center flex-wrap gap-1.5 mb-3">
                      <span className="text-xs font-medium text-gray-500 mr-1">Root cause chain:</span>
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
                            <ArrowRight className="w-3 h-3 text-gray-400 flex-shrink-0" />
                          )}
                        </span>
                      ))}
                    </div>
                  )}

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

      {/* Why This Score */}
      {data.why_summary && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Why this score?</h3>
          <p className="text-gray-700 leading-relaxed">{data.why_summary}</p>
        </div>
      )}

      {/* 3D Causal Metric Graph */}
      {data.causal_graph && (
        <div className="rounded-2xl overflow-hidden border border-slate-200/60 shadow-card w-full">
          <div className="px-5 py-3.5 bg-gradient-to-r from-slate-900 to-slate-800 border-b border-slate-700/50">
            <h3 className="text-base font-semibold text-slate-300">How metrics influence each other</h3>
            <p className="text-xs text-slate-500 mt-0.5">Causal flow: supply chain → operations → finance → customer health</p>
          </div>
          <Graph3D graphData={data.causal_graph} height={480} title="3D Causal Metric Flow" />
        </div>
      )}

      {/* Contributing Factors */}
      {data.contributing_factors && data.contributing_factors.length > 0 && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">What&apos;s driving your score</h3>
          <div className="space-y-3">
            {data.contributing_factors.map((factor, idx) => (
              <div
                key={`${factor.metric}-${idx}`}
                className="flex items-center justify-between p-4 bg-gray-50 rounded-lg border border-gray-100 hover:border-gray-200 transition-colors"
              >
                <div className="flex items-center space-x-3">
                  {factor.direction === 'negative' ? (
                    <AlertCircle className="w-5 h-5 text-orange-500 flex-shrink-0" />
                  ) : factor.direction === 'positive' ? (
                    <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
                  ) : (
                    <Minus className="w-5 h-5 text-gray-400 flex-shrink-0" />
                  )}
                  <div>
                    <p className="font-medium text-gray-900">{factor.label}</p>
                    <p className="text-xs text-gray-500 capitalize mt-0.5">{factor.domain}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  {factor.value != null && (
                    <span className="text-sm text-gray-600 font-mono">
                      {typeof factor.value === 'number' && factor.value < 1 && factor.value > 0
                        ? (factor.value * 100).toFixed(1) + '%'
                        : factor.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                    </span>
                  )}
                  {factor.score != null && (
                    <span className="text-sm font-medium text-gray-700">{factor.score} pts</span>
                  )}
                  <span
                    className={`text-xs font-medium px-2 py-0.5 rounded ${
                      STATUS_STYLES[factor.status] || STATUS_STYLES.no_data
                    }`}
                  >
                    {factor.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Domain Breakdown */}
      {data.domains && Object.keys(data.domains).length > 0 && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Score by domain</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(data.domains).map(([domain, domainData]) => {
              const Icon = DOMAIN_ICONS[domain] || Activity
              return (
                <div key={domain} className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <div className="flex items-center space-x-2 mb-2">
                    <Icon className="w-4 h-4 text-gray-500" />
                    <span className="text-sm font-medium text-gray-700 capitalize">{domain}</span>
                  </div>
                  <div className="flex items-baseline space-x-2">
                    <span className="text-2xl font-bold text-gray-900">
                      {domainData.score != null ? domainData.score.toFixed(0) : '—'}
                    </span>
                    {domainData.grade && (
                      <span
                        className={`text-xs font-medium px-1.5 py-0.5 rounded ${
                          GRADE_COLORS[domainData.grade] || ''
                        }`}
                      >
                        {domainData.grade}
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {domainData.metrics_available ?? 0}/{domainData.metrics_total ?? 0} metrics
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Recent Incidents CTA */}
      {data.recent_incidents_count > 0 && (
        <div className="card border-primary-100 bg-primary-50/30">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900">
                {data.recent_incidents_count} recent incident{data.recent_incidents_count !== 1 ? 's' : ''} detected
              </p>
              <p className="text-sm text-gray-600 mt-1">
                Review root causes and blast radius to understand what affected your score
              </p>
            </div>
            <Link
              to="/incidents"
              className="btn btn-primary flex items-center space-x-2"
            >
              View incidents
              <ArrowRightIcon className="w-4 h-4" />
            </Link>
          </div>
        </div>
      )}

      {data.evaluated_at && (
        <p className="text-xs text-gray-500">
          Last evaluated: {new Date(data.evaluated_at).toLocaleString()}
        </p>
      )}
    </div>
  )
}
