import { useState, useEffect, useCallback } from 'react'
import { useParams, Link } from 'react-router-dom'
import { api } from '../api/client'
import LoadingState from '../components/common/LoadingState'
import SeverityBadge from '../components/common/SeverityBadge'
import { Download, Printer, ArrowLeft, Clock, AlertTriangle, Shield, CheckCircle } from 'lucide-react'

export default function PostmortemView() {
  const { id: incidentId } = useParams()
  const [loading, setLoading] = useState(true)
  const [postmortem, setPostmortem] = useState(null)
  const [error, setError] = useState(null)

  const fetchPostmortem = useCallback(async () => {
    try {
      const data = await api.incidents.getPostmortem(incidentId, 'json')
      setPostmortem(data)
    } catch (err) {
      console.error('Failed to fetch postmortem:', err)
      setError(err.response?.data?.detail || 'Postmortem not available')
    } finally {
      setLoading(false)
    }
  }, [incidentId])

  useEffect(() => {
    fetchPostmortem()
  }, [fetchPostmortem])

  if (loading) {
    return <LoadingState message="Generating postmortem report..." />
  }

  if (error || !postmortem) {
    return (
      <div className="text-center py-12">
        <AlertTriangle className="w-12 h-12 text-gray-400 mx-auto mb-3" />
        <p className="text-gray-600">{error || 'Postmortem not available'}</p>
        <Link to={`/incidents/${incidentId}`} className="text-primary-600 hover:text-primary-700 mt-2 inline-block">
          Back to incident
        </Link>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Link
            to={`/incidents/${incidentId}`}
            className="btn btn-secondary p-2"
          >
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Incident Postmortem</h1>
            <p className="text-sm text-gray-600 mt-0.5">
              Generated {new Date(postmortem.generated_at).toLocaleString()}
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <button onClick={() => window.print()} className="btn btn-secondary flex items-center space-x-2">
            <Printer className="w-4 h-4" />
            <span>Print</span>
          </button>
        </div>
      </div>

      {/* Postmortem Report */}
      <div className="card max-w-4xl mx-auto print:shadow-none">
        {/* Title & Meta */}
        <div className="border-b border-gray-200 pb-6 mb-6">
          <h2 className="text-xl font-bold text-gray-900">{postmortem.title}</h2>
          <div className="flex items-center space-x-3 mt-3">
            <SeverityBadge severity={postmortem.severity} />
            <span className="inline-flex items-center space-x-1 text-sm text-gray-600">
              <Clock className="w-4 h-4" />
              <span>Duration: {postmortem.duration}</span>
            </span>
            <span className="text-sm text-gray-600 capitalize">Status: {postmortem.status}</span>
          </div>
        </div>

        {/* Executive Summary */}
        <section className="mb-8">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Executive Summary</h3>
          <p className="text-gray-700 leading-relaxed">{postmortem.one_line_summary}</p>
        </section>

        {/* Timeline */}
        {postmortem.timeline?.length > 0 && (
          <section className="mb-8">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Incident Timeline</h3>
            <div className="space-y-4">
              {postmortem.timeline.map((entry, idx) => (
                <div key={idx} className="flex items-start space-x-4">
                  <div className="flex-shrink-0 w-2 h-2 mt-2 bg-primary-600 rounded-full" />
                  <div className="flex-1">
                    <div className="text-sm font-medium text-gray-500">
                      {new Date(entry.timestamp).toLocaleString()}
                    </div>
                    <div className="text-gray-800 mt-0.5">{entry.event_description}</div>
                    {entry.metric_name && (
                      <div className="text-sm text-gray-600 mt-0.5">
                        {entry.metric_name}: {entry.metric_value?.toFixed(2)}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Root Cause */}
        <section className="mb-8">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Root Cause Analysis</h3>
          <p className="text-gray-700 leading-relaxed">{postmortem.root_cause_summary}</p>
        </section>

        {/* Contributing Factors */}
        {postmortem.contributing_factors?.length > 0 && (
          <section className="mb-8">
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Contributing Factors</h3>
            <ul className="space-y-2">
              {postmortem.contributing_factors.map((factor, idx) => (
                <li key={idx} className="flex items-start space-x-2">
                  <AlertTriangle className="w-4 h-4 text-yellow-500 mt-0.5 flex-shrink-0" />
                  <span className="text-gray-700">{factor}</span>
                </li>
              ))}
            </ul>
          </section>
        )}

        {/* Blast Radius Summary */}
        {postmortem.blast_radius && (
          <section className="mb-8">
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Impact Assessment</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="p-3 bg-gray-50 rounded-lg text-center">
                <p className="text-2xl font-bold text-gray-900">
                  {postmortem.blast_radius.customers_affected?.toLocaleString()}
                </p>
                <p className="text-xs text-gray-500 mt-1">Customers Affected</p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg text-center">
                <p className="text-2xl font-bold text-gray-900">
                  ${postmortem.blast_radius.estimated_revenue_exposure?.toLocaleString()}
                </p>
                <p className="text-xs text-gray-500 mt-1">Revenue Exposure</p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg text-center">
                <p className="text-2xl font-bold text-gray-900">
                  ${postmortem.blast_radius.estimated_refund_exposure?.toLocaleString()}
                </p>
                <p className="text-xs text-gray-500 mt-1">Refund Exposure</p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg text-center">
                <p className="text-2xl font-bold text-gray-900 capitalize">
                  {postmortem.blast_radius.blast_radius_severity}
                </p>
                <p className="text-xs text-gray-500 mt-1">Severity</p>
              </div>
            </div>
          </section>
        )}

        {/* Recommendations */}
        <section className="mb-8">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">Recommendations</h3>
          <div className="space-y-2">
            {postmortem.recommendations?.map((rec, idx) => (
              <div key={idx} className="flex items-start space-x-2">
                <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                <span className="text-gray-700">{rec}</span>
              </div>
            ))}
          </div>
        </section>

        {/* Recommended Monitors */}
        {postmortem.monitors?.length > 0 && (
          <section className="mb-8">
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Recommended Monitors</h3>
            <div className="space-y-2">
              {postmortem.monitors.map((monitor, idx) => (
                <div key={idx} className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                  <div className="font-medium text-blue-900">{monitor.name}</div>
                  <div className="text-sm text-blue-700 mt-0.5">
                    {monitor.metric_name} — {monitor.condition} — {monitor.check_frequency}
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Confidence Note */}
        <section className="p-4 bg-gray-50 rounded-lg border border-gray-200">
          <div className="flex items-center space-x-2 mb-2">
            <Shield className="w-4 h-4 text-gray-500" />
            <span className="text-sm font-medium text-gray-700">Confidence & Data Quality</span>
          </div>
          <p className="text-sm text-gray-600">{postmortem.confidence_note}</p>
          <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
            <span>Data Quality: {(postmortem.data_quality_score * 100).toFixed(0)}%</span>
            <span>Algorithm: {postmortem.algorithm_version}</span>
            <span>Run: {postmortem.run_id}</span>
          </div>
        </section>
      </div>
    </div>
  )
}
