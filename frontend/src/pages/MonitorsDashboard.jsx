import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/client'
import LoadingState from '../components/common/LoadingState'
import SeverityBadge from '../components/common/SeverityBadge'
import { Plus, CheckCircle, AlertCircle, XCircle, Bell, ToggleLeft, ToggleRight, RefreshCw } from 'lucide-react'
import toast from 'react-hot-toast'

export default function MonitorsDashboard() {
  const [loading, setLoading] = useState(true)
  const [monitors, setMonitors] = useState([])
  const [alerts, setAlerts] = useState([])
  const [evaluating, setEvaluating] = useState(false)
  const [activeTab, setActiveTab] = useState('monitors')

  const fetchData = useCallback(async () => {
    try {
      const [monitorsData, alertsData] = await Promise.all([
        api.monitors.list(),
        api.monitors.alerts(),
      ])
      setMonitors(Array.isArray(monitorsData) ? monitorsData : [])
      setAlerts(Array.isArray(alertsData) ? alertsData : [])
    } catch (error) {
      console.error('Failed to fetch monitors:', error)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  const handleEvaluate = async () => {
    setEvaluating(true)
    try {
      const result = await api.monitors.evaluate()
      toast.success(`Evaluation complete: ${result.alerts_triggered || 0} alerts triggered`)
      fetchData()
    } catch (error) {
      toast.error('Evaluation failed')
    } finally {
      setEvaluating(false)
    }
  }

  const handleToggle = async (monitorId) => {
    try {
      await api.monitors.toggle(monitorId)
      toast.success('Monitor updated')
      fetchData()
    } catch (error) {
      toast.error('Failed to toggle monitor')
    }
  }

  const handleDelete = async (monitorId) => {
    try {
      await api.monitors.delete(monitorId)
      toast.success('Monitor deleted')
      fetchData()
    } catch (error) {
      toast.error('Failed to delete monitor')
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
          <h1 className="text-2xl font-bold text-gray-900">Monitors & Alerts</h1>
          <p className="text-gray-600 mt-1">Health monitors, SLO tracking, and alert management</p>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={handleEvaluate}
            disabled={evaluating}
            className="btn btn-secondary flex items-center space-x-2"
          >
            <RefreshCw className={`w-4 h-4 ${evaluating ? 'animate-spin' : ''}`} />
            <span>{evaluating ? 'Evaluating...' : 'Evaluate All'}</span>
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200" role="tablist" aria-label="Monitors and alerts">
        <div className="flex space-x-8">
          <button
            role="tab"
            aria-selected={activeTab === 'monitors'}
            aria-controls="panel-monitors"
            id="tab-monitors"
            onClick={() => setActiveTab('monitors')}
            className={`py-3 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'monitors'
                ? 'border-primary-600 text-primary-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            Monitors ({monitors.length})
          </button>
          <button
            role="tab"
            aria-selected={activeTab === 'alerts'}
            aria-controls="panel-alerts"
            id="tab-alerts"
            onClick={() => setActiveTab('alerts')}
            className={`py-3 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'alerts'
                ? 'border-primary-600 text-primary-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            <span className="flex items-center space-x-2">
              <span>Alerts</span>
              {alerts.length > 0 && (
                <span className="bg-red-500 text-white text-xs px-1.5 py-0.5 rounded-full">
                  {alerts.length}
                </span>
              )}
            </span>
          </button>
        </div>
      </div>

      {/* Monitors Tab */}
      {activeTab === 'monitors' && (
        <div role="tabpanel" id="panel-monitors" aria-labelledby="tab-monitors">
          {monitors.length === 0 ? (
            <div className="card text-center py-12">
              <Bell className="w-12 h-12 text-gray-300 mx-auto mb-3" />
              <h2 className="text-gray-600 font-medium">No monitors configured yet</h2>
              <p className="text-sm text-gray-500 mt-1">
                Run analysis and generate postmortems to auto-create health monitors for your business metrics.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {monitors.map((monitor) => (
                <div key={monitor.monitor_id} className="card">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-gray-900 truncate">{monitor.name}</h3>
                      <p className="text-xs text-gray-500 mt-0.5 truncate">{monitor.description}</p>
                    </div>
                    <button
                      onClick={() => handleToggle(monitor.monitor_id)}
                      className="ml-2 flex-shrink-0"
                      aria-label={monitor.enabled ? `Disable ${monitor.name}` : `Enable ${monitor.name}`}
                    >
                      {monitor.enabled ? (
                        <ToggleRight className="w-6 h-6 text-green-500" />
                      ) : (
                        <ToggleLeft className="w-6 h-6 text-gray-400" />
                      )}
                    </button>
                  </div>

                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-500">Metric</span>
                      <span className="font-medium text-gray-900">
                        {(monitor.metric_name || '').replace(/_/g, ' ')}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Condition</span>
                      <span className="font-mono text-xs text-gray-700">{monitor.condition}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Frequency</span>
                      <span className="capitalize text-gray-700">{monitor.check_frequency}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-500">Severity</span>
                      <SeverityBadge severity={monitor.severity_if_triggered} />
                    </div>
                  </div>

                  <div className="flex items-center justify-between pt-3 mt-3 border-t border-gray-200">
                    <span className="text-xs text-gray-500">
                      Baseline: {monitor.baseline_window_days}d
                    </span>
                    <button
                      onClick={() => handleDelete(monitor.monitor_id)}
                      className="text-xs text-red-500 hover:text-red-700"
                      aria-label={`Delete ${monitor.name}`}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Alerts Tab */}
      {activeTab === 'alerts' && (
        <div role="tabpanel" id="panel-alerts" aria-labelledby="tab-alerts">
          {alerts.length === 0 ? (
            <div className="card text-center py-12">
              <CheckCircle className="w-12 h-12 text-green-300 mx-auto mb-3" />
              <p className="text-gray-600 font-medium">No active alerts</p>
              <p className="text-sm text-gray-500 mt-1">All monitors are within normal parameters</p>
            </div>
          ) : (
            <div className="space-y-3">
              {alerts.map((alert) => (
                <div
                  key={alert.alert_id}
                  className="card border-l-4 border-l-red-400"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
                        <span className="font-medium text-gray-900">
                          {(alert.metric_name || '').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </span>
                        <SeverityBadge severity={alert.severity} />
                      </div>
                      <p className="text-sm text-gray-700 mt-1">{alert.message}</p>
                      <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                        <span>Current: {alert.current_value?.toFixed(4)}</span>
                        <span>Baseline: {alert.baseline_value?.toFixed(4)}</span>
                        <span>Threshold: {alert.threshold}</span>
                        <span>
                          {alert.triggered_at ? new Date(alert.triggered_at).toLocaleString() : ''}
                        </span>
                      </div>
                    </div>
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-800 capitalize">
                      {alert.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
