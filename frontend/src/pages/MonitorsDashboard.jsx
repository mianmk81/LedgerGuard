import { useState, useEffect } from 'react'
import { api } from '../api/client'
import LoadingState from '../components/common/LoadingState'
import { Plus, CheckCircle, AlertCircle, XCircle } from 'lucide-react'

const statusIcons = {
  healthy: { icon: CheckCircle, color: 'text-green-500' },
  warning: { icon: AlertCircle, color: 'text-yellow-500' },
  critical: { icon: XCircle, color: 'text-red-500' },
}

export default function MonitorsDashboard() {
  const [loading, setLoading] = useState(true)
  const [monitors, setMonitors] = useState([])

  useEffect(() => {
    fetchMonitors()
  }, [])

  const fetchMonitors = async () => {
    try {
      const data = await api.monitors.list()
      setMonitors(data)
    } catch (error) {
      console.error('Failed to fetch monitors:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <LoadingState message="Loading monitors..." />
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Monitors</h1>
          <p className="text-gray-600 mt-1">Health monitors and SLO tracking</p>
        </div>
        <button className="btn btn-primary flex items-center space-x-2">
          <Plus className="w-4 h-4" />
          <span>Create Monitor</span>
        </button>
      </div>

      {/* Monitors Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {monitors.map((monitor) => {
          const StatusIcon = statusIcons[monitor.status]?.icon || AlertCircle
          const iconColor = statusIcons[monitor.status]?.color || 'text-gray-500'

          return (
            <div key={monitor.monitor_id} className="card">
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <h3 className="font-semibold text-gray-900">{monitor.name}</h3>
                  <p className="text-sm text-gray-600 mt-1">{monitor.metric}</p>
                </div>
                <StatusIcon className={`w-6 h-6 ${iconColor}`} />
              </div>

              <div className="space-y-3">
                <div>
                  <div className="text-2xl font-bold text-gray-900">
                    {monitor.current_value.toFixed(2)}
                  </div>
                  {monitor.threshold && (
                    <div className="text-sm text-gray-600">
                      Threshold: {monitor.threshold.toFixed(2)}
                    </div>
                  )}
                </div>

                <div>
                  <div className="flex items-center justify-between text-sm mb-1">
                    <span className="text-gray-600">SLO Compliance</span>
                    <span className="font-medium text-gray-900">
                      {(monitor.compliance * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-primary-600 h-2 rounded-full transition-all"
                      style={{ width: `${monitor.compliance * 100}%` }}
                    />
                  </div>
                </div>

                <div className="flex items-center justify-between pt-3 border-t border-gray-200">
                  <span
                    className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium ${
                      monitor.enabled ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                    }`}
                  >
                    {monitor.enabled ? 'Enabled' : 'Disabled'}
                  </span>
                  <span className="text-xs text-gray-600">{monitor.type}</span>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
