import { useState, useEffect, useCallback, useMemo } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import LoadingState from '../components/common/LoadingState'
import SeverityBadge from '../components/common/SeverityBadge'
import ConfidenceBadge from '../components/common/ConfidenceBadge'
import { Search, Filter, Play, CheckCircle } from 'lucide-react'
import toast from 'react-hot-toast'

export default function IncidentsList() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [incidents, setIncidents] = useState([])
  const [analyzing, setAnalyzing] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [filters, setFilters] = useState({
    severity: null,
    status: null,
  })

  useEffect(() => {
    fetchIncidents()
  }, [filters])

  const fetchIncidents = useCallback(async () => {
    setLoading(true)
    try {
      const data = await api.incidents.list(filters)
      setIncidents(Array.isArray(data) ? data : [])
    } catch (error) {
      console.error('Failed to fetch incidents:', error)
    } finally {
      setLoading(false)
    }
  }, [filters])

  const filteredIncidents = useMemo(() => {
    if (!searchQuery.trim()) return incidents
    const q = searchQuery.trim().toLowerCase()
    return incidents.filter((inc) => {
      const type = (inc.incident_type || '').toLowerCase()
      const metric = (inc.primary_metric || '').toLowerCase()
      const status = (inc.status || '').toLowerCase()
      return type.includes(q) || metric.includes(q) || status.includes(q)
    })
  }, [incidents, searchQuery])

  const runAnalysis = useCallback(async () => {
    setAnalyzing(true)
    try {
      const result = await api.analysis.run({
        lookback_days: 30,
        min_zscore: 3.0,
        run_rca: true,
        run_blast_radius: true,
        run_postmortem: true,
      })
      toast.success(`Analysis complete: ${result.incidents_detected} incidents detected`)
      fetchIncidents()
    } catch (error) {
      toast.error('Analysis failed — please try again')
      console.error('Analysis failed:', error)
    } finally {
      setAnalyzing(false)
    }
  }, [fetchIncidents])

  if (loading) {
    return <LoadingState message="Loading incidents..." />
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">What went wrong — and why</h1>
          <p className="text-gray-600 mt-1">Anomalies detected with root cause and impact</p>
        </div>
        <button
          onClick={runAnalysis}
          disabled={analyzing}
          className="btn btn-primary flex items-center space-x-2"
        >
          <Play className="w-4 h-4" />
          <span>{analyzing ? 'Analyzing...' : 'Run Analysis'}</span>
        </button>
      </div>

      {/* Filters */}
      <div className="card">
        <div className="flex flex-wrap items-end gap-4">
          <div className="flex-1 min-w-[200px]">
            <label htmlFor="incident-search" className="block text-xs font-medium text-gray-500 mb-1">Search</label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" aria-hidden="true" />
              <input
                id="incident-search"
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search by type, metric, or status..."
                className="input pl-10"
              />
            </div>
          </div>
          <div>
            <label htmlFor="severity-filter" className="block text-xs font-medium text-gray-500 mb-1">Severity</label>
            <select
              id="severity-filter"
              className="input w-40"
              value={filters.severity || ''}
              onChange={(e) => setFilters({ ...filters, severity: e.target.value || null })}
            >
              <option value="">All Severities</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
          </div>
          <div>
            <label htmlFor="status-filter" className="block text-xs font-medium text-gray-500 mb-1">Status</label>
            <select
              id="status-filter"
              className="input w-40"
              value={filters.status || ''}
              onChange={(e) => setFilters({ ...filters, status: e.target.value || null })}
            >
              <option value="">All Statuses</option>
              <option value="open">Open</option>
              <option value="acknowledged">Acknowledged</option>
              <option value="resolved">Resolved</option>
            </select>
          </div>
        </div>
      </div>

      {/* Incidents Table */}
      {filteredIncidents.length === 0 ? (
        <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-green-50 via-emerald-50/50 to-teal-50 border border-green-200/60 shadow-card py-16">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" style={{ backgroundSize: '200% 100%' }} />
          <div className="relative text-center">
            <div className="inline-flex p-6 bg-green-100/80 rounded-3xl mb-6 shadow-soft">
              <CheckCircle className="w-16 h-16 text-green-600 drop-shadow-sm" />
            </div>
            {searchQuery.trim() ? (
              <>
                <h2 className="text-2xl font-bold text-gray-900 mb-2">No matching incidents</h2>
                <p className="text-gray-600 font-medium mb-1">No results for &ldquo;{searchQuery}&rdquo;</p>
                <p className="text-sm text-gray-500 mb-6">Try a different search term or clear your filters</p>
                <button
                  onClick={() => setSearchQuery('')}
                  className="btn btn-secondary"
                >
                  Clear search
                </button>
              </>
            ) : (
              <>
                <h2 className="text-2xl font-bold text-gray-900 mb-2">All Clear!</h2>
                <p className="text-gray-600 font-medium mb-1">No incidents detected</p>
                <p className="text-sm text-gray-500 mb-6">Your financial operations are running smoothly</p>
                <button
                  onClick={runAnalysis}
                  disabled={analyzing}
                  className="btn btn-primary shadow-lg hover:shadow-xl transition-all duration-200"
                >
                  {analyzing ? 'Analyzing...' : 'Run Analysis'}
                </button>
              </>
            )}
          </div>
        </div>
      ) : (
        <div className="card p-0">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Incident Type
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Primary Metric
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Severity
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Confidence
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Z-Score
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Detected
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filteredIncidents.map((incident) => (
                  <tr
                    key={incident.incident_id}
                    className="hover:bg-gray-50 transition-colors cursor-pointer focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:ring-inset"
                    onClick={() => navigate(`/incidents/${incident.incident_id}`)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        navigate(`/incidents/${incident.incident_id}`)
                      }
                    }}
                    tabIndex={0}
                    role="link"
                    aria-label={`View incident: ${(incident.incident_type || '').replace(/_/g, ' ')} — ${incident.severity} severity`}
                  >
                    <td className="px-6 py-4">
                      <span className="font-medium text-primary-600">
                        {(incident.incident_type || '').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-700">
                      {(incident.primary_metric || '').replace(/_/g, ' ')}
                    </td>
                    <td className="px-6 py-4">
                      <SeverityBadge severity={incident.severity} />
                    </td>
                    <td className="px-6 py-4">
                      <ConfidenceBadge confidence={incident.confidence} />
                    </td>
                    <td className="px-6 py-4 text-sm font-mono">
                      {incident.primary_metric_zscore?.toFixed(1)}σ
                    </td>
                    <td className="px-6 py-4">
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800 border border-blue-200 capitalize">
                        {incident.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-600">
                      {incident.detected_at ? new Date(incident.detected_at).toLocaleDateString() : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
