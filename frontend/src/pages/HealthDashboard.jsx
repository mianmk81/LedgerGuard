import { useState, useEffect } from 'react'
import { api } from '../api/client'
import LoadingState from '../components/common/LoadingState'
import Sparkline from '../components/common/Sparkline'
import { TrendingUp, TrendingDown, AlertTriangle, Activity } from 'lucide-react'

export default function HealthDashboard() {
  const [loading, setLoading] = useState(true)
  const [metrics, setMetrics] = useState(null)
  const [healthScore, setHealthScore] = useState(null)
  const [incidents, setIncidents] = useState([])

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    try {
      const [metricsData, healthData, incidentsData] = await Promise.all([
        api.metrics.getDashboard('30d'),
        api.metrics.getHealthScore(),
        api.incidents.list({ limit: 5 }),
      ])

      setMetrics(metricsData)
      setHealthScore(healthData)
      setIncidents(incidentsData)
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <LoadingState message="Loading dashboard..." />
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Business Health Dashboard</h1>
        <p className="text-gray-600 mt-1">Real-time monitoring and reliability metrics</p>
      </div>

      {/* Health Score */}
      {healthScore && (
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-gray-900">Overall Health Score</h2>
              <p className="text-sm text-gray-600 mt-1">Composite reliability metric</p>
            </div>
            <div className="text-right">
              <div className="text-4xl font-bold text-primary-600">
                {healthScore.overall_score.toFixed(1)}
              </div>
              <div className="text-sm text-gray-600 mt-1">{healthScore.trend}</div>
            </div>
          </div>

          <div className="grid grid-cols-4 gap-4 mt-6">
            {Object.entries(healthScore.components).map(([key, value]) => (
              <div key={key} className="text-center">
                <div className="text-2xl font-semibold text-gray-900">{value.toFixed(0)}</div>
                <div className="text-xs text-gray-600 mt-1">
                  {key.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Key Metrics */}
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {metrics.metrics.map((metric) => (
            <div key={metric.metric_name} className="card">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <p className="text-sm text-gray-600">
                    {metric.metric_name.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                  </p>
                  <p className="text-2xl font-bold text-gray-900 mt-2">
                    {metric.current_value.toLocaleString()}
                  </p>
                  <div className="flex items-center space-x-2 mt-2">
                    {metric.trend === 'up' ? (
                      <TrendingUp className="w-4 h-4 text-green-500" />
                    ) : (
                      <TrendingDown className="w-4 h-4 text-red-500" />
                    )}
                    <span
                      className={`text-sm font-medium ${
                        metric.change_percent >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}
                    >
                      {metric.change_percent >= 0 ? '+' : ''}
                      {metric.change_percent.toFixed(1)}%
                    </span>
                  </div>
                </div>
                <div className="w-24 h-16">
                  <Sparkline data={metric.sparkline} />
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Recent Incidents */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">Recent Incidents</h2>
          <a href="/incidents" className="text-sm text-primary-600 hover:text-primary-700">
            View all
          </a>
        </div>

        {incidents.length === 0 ? (
          <div className="text-center py-8">
            <Activity className="w-12 h-12 text-gray-400 mx-auto mb-3" />
            <p className="text-gray-600">No incidents detected</p>
          </div>
        ) : (
          <div className="space-y-3">
            {incidents.map((incident) => (
              <a
                key={incident.incident_id}
                href={`/incidents/${incident.incident_id}`}
                className="block p-4 border border-gray-200 rounded-lg hover:border-primary-300 hover:shadow-sm transition-all"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <AlertTriangle className="w-5 h-5 text-orange-500" />
                    <div>
                      <p className="font-medium text-gray-900">{incident.title}</p>
                      <p className="text-sm text-gray-600 mt-1">
                        {new Date(incident.detected_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium text-gray-900">
                      {incident.affected_entities_count} entities affected
                    </div>
                  </div>
                </div>
              </a>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
