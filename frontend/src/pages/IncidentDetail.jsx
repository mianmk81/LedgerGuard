import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { api } from '../api/client'
import LoadingState from '../components/common/LoadingState'
import SeverityBadge from '../components/common/SeverityBadge'
import ConfidenceBadge from '../components/common/ConfidenceBadge'
import { FileText, Share2, Network } from 'lucide-react'

export default function IncidentDetail() {
  const { incidentId } = useParams()
  const [loading, setLoading] = useState(true)
  const [incident, setIncident] = useState(null)

  useEffect(() => {
    fetchIncidentDetail()
  }, [incidentId])

  const fetchIncidentDetail = async () => {
    try {
      const data = await api.incidents.get(incidentId)
      setIncident(data)
    } catch (error) {
      console.error('Failed to fetch incident:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <LoadingState message="Loading incident details..." />
  }

  if (!incident) {
    return <div>Incident not found</div>
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{incident.title}</h1>
          <div className="flex items-center space-x-3 mt-2">
            <SeverityBadge severity={incident.severity} />
            <ConfidenceBadge confidence={incident.confidence} />
            <span className="text-sm text-gray-600">
              Detected {new Date(incident.detected_at).toLocaleString()}
            </span>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Link
            to={`/incidents/${incidentId}/postmortem`}
            className="btn btn-secondary flex items-center space-x-2"
          >
            <FileText className="w-4 h-4" />
            <span>Postmortem</span>
          </Link>
          <button className="btn btn-secondary">
            <Share2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Description */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-3">Description</h2>
        <p className="text-gray-700">{incident.description}</p>
      </div>

      {/* Root Causes */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Root Causes</h2>
        <div className="space-y-3">
          {incident.root_causes.map((cause, idx) => (
            <div key={idx} className="p-4 bg-gray-50 rounded-lg border border-gray-200">
              <div className="flex items-start justify-between mb-2">
                <div className="font-medium text-gray-900">
                  {cause.entity_type.toUpperCase()}: {cause.entity_id}
                </div>
                <div className="text-sm font-medium text-primary-600">
                  {(cause.likelihood * 100).toFixed(0)}% likelihood
                </div>
              </div>
              <p className="text-sm text-gray-700">{cause.explanation}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Affected Entities */}
      <div className="grid grid-cols-2 gap-6">
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-3">Affected Entities</h2>
          <div className="text-3xl font-bold text-primary-600 mb-2">
            {incident.affected_entities.length}
          </div>
          <div className="space-y-2">
            {incident.affected_entities.slice(0, 5).map((entityId) => (
              <div key={entityId} className="text-sm text-gray-600">
                {entityId}
              </div>
            ))}
            {incident.affected_entities.length > 5 && (
              <div className="text-sm text-gray-500">
                +{incident.affected_entities.length - 5} more
              </div>
            )}
          </div>
        </div>

        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-3">Blast Radius</h2>
          <div className="text-3xl font-bold text-primary-600 mb-2">
            Depth: {incident.blast_radius_depth}
          </div>
          <Link
            to={`/incidents/${incidentId}/blast-radius`}
            className="btn btn-secondary flex items-center space-x-2 mt-4"
          >
            <Network className="w-4 h-4" />
            <span>View Graph</span>
          </Link>
        </div>
      </div>

      {/* Timeline */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Timeline</h2>
        <div className="space-y-4">
          {incident.timeline.map((event, idx) => (
            <div key={idx} className="flex items-start space-x-4">
              <div className="flex-shrink-0 w-2 h-2 mt-2 bg-primary-600 rounded-full" />
              <div className="flex-1">
                <div className="text-sm text-gray-600">
                  {new Date(event.timestamp).toLocaleString()}
                </div>
                <div className="font-medium text-gray-900">{event.event}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
