import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/client'
import SeverityBadge from '../components/common/SeverityBadge'
import { Play, TrendingUp, BarChart3, GitCompare, Zap } from 'lucide-react'
import toast from 'react-hot-toast'

export default function ComparisonSimulation() {
  const [activeTab, setActiveTab] = useState('comparison')
  const [loading, setLoading] = useState(false)
  const [comparisonResult, setComparisonResult] = useState(null)
  const [simulationResult, setSimulationResult] = useState(null)

  // Comparison form state
  const [incidentAId, setIncidentAId] = useState('')
  const [incidentBId, setIncidentBId] = useState('')
  const [incidents, setIncidents] = useState([])

  // Simulation form state
  const [perturbations, setPerturbations] = useState([
    { metric: 'order_volume', change: '+50%' },
  ])

  const fetchIncidents = useCallback(async () => {
    try {
      const data = await api.incidents.list({ limit: 50 })
      const list = Array.isArray(data) ? data : []
      setIncidents(list)
      if (list.length >= 2) {
        setIncidentAId(list[0].incident_id)
        setIncidentBId(list[1].incident_id)
      }
    } catch (error) {
      console.error('Failed to fetch incidents:', error)
    }
  }, [])

  useEffect(() => {
    fetchIncidents()
  }, [fetchIncidents])

  const handleRunComparison = async () => {
    if (!incidentAId || !incidentBId) {
      toast.error('Select two incidents to compare')
      return
    }
    setLoading(true)
    try {
      const data = await api.comparison.compare({
        incident_a_id: incidentAId,
        incident_b_id: incidentBId,
      })
      setComparisonResult(data)
      toast.success('Comparison complete')
    } catch (error) {
      toast.error('Comparison failed')
      console.error('Failed to run comparison:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleRunSimulation = async () => {
    setLoading(true)
    try {
      const data = await api.comparison.whatIf(perturbations)
      setSimulationResult(data)
      toast.success('Simulation complete')
    } catch (error) {
      toast.error('Simulation failed')
      console.error('Failed to run simulation:', error)
    } finally {
      setLoading(false)
    }
  }

  const addPerturbation = () => {
    setPerturbations([...perturbations, { metric: '', change: '' }])
  }

  const updatePerturbation = (idx, field, value) => {
    const updated = [...perturbations]
    updated[idx][field] = value
    setPerturbations(updated)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Analysis & Simulation</h1>
        <p className="text-gray-600 mt-1">Compare incidents and run what-if scenarios</p>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200" role="tablist" aria-label="Analysis modes">
        <div className="flex space-x-8">
          <button
            role="tab"
            aria-selected={activeTab === 'comparison'}
            aria-controls="panel-comparison"
            id="tab-comparison"
            onClick={() => setActiveTab('comparison')}
            className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'comparison'
                ? 'border-primary-600 text-primary-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <span className="flex items-center space-x-2">
              <GitCompare className="w-4 h-4" />
              <span>Incident Comparison</span>
            </span>
          </button>
          <button
            role="tab"
            aria-selected={activeTab === 'simulation'}
            aria-controls="panel-simulation"
            id="tab-simulation"
            onClick={() => setActiveTab('simulation')}
            className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'simulation'
                ? 'border-primary-600 text-primary-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <span className="flex items-center space-x-2">
              <Zap className="w-4 h-4" />
              <span>What-If Simulation</span>
            </span>
          </button>
        </div>
      </div>

      {/* Comparison Tab */}
      {activeTab === 'comparison' && (
        <div className="space-y-6" role="tabpanel" id="panel-comparison" aria-labelledby="tab-comparison">
          <div className="card">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Compare Two Incidents</h2>
            {incidents.length < 2 ? (
              <div className="text-center py-8">
                <GitCompare className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                <p className="text-gray-600 font-medium">At least 2 incidents needed</p>
                <p className="text-sm text-gray-500 mt-1">Run analysis on the Incidents page to detect anomalies first.</p>
              </div>
            ) : (
              <>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label htmlFor="incident-a" className="block text-sm font-medium text-gray-700 mb-2">Incident A</label>
                    <select
                      id="incident-a"
                      className="input"
                      value={incidentAId}
                      onChange={(e) => setIncidentAId(e.target.value)}
                    >
                      <option value="">Select incident...</option>
                      {incidents.map((inc) => (
                        <option key={inc.incident_id} value={inc.incident_id}>
                          {(inc.incident_type || '').replace(/_/g, ' ')} — {inc.severity} — z:{inc.primary_metric_zscore?.toFixed(1)}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label htmlFor="incident-b" className="block text-sm font-medium text-gray-700 mb-2">Incident B</label>
                    <select
                      id="incident-b"
                      className="input"
                      value={incidentBId}
                      onChange={(e) => setIncidentBId(e.target.value)}
                    >
                      <option value="">Select incident...</option>
                      {incidents.map((inc) => (
                        <option key={inc.incident_id} value={inc.incident_id}>
                          {(inc.incident_type || '').replace(/_/g, ' ')} — {inc.severity} — z:{inc.primary_metric_zscore?.toFixed(1)}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <button
                  onClick={handleRunComparison}
                  disabled={loading || !incidentAId || !incidentBId}
                  className="btn btn-primary flex items-center space-x-2 mt-4"
                >
                  <Play className="w-4 h-4" />
                  <span>{loading ? 'Comparing...' : 'Run Comparison'}</span>
                </button>
              </>
            )}
          </div>

          {comparisonResult && (
            <div className="space-y-4">
              {/* Narrative */}
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Comparison Narrative</h3>
                <p className="text-gray-700 leading-relaxed">{comparisonResult.narrative}</p>
              </div>

              {/* Shared/Unique Root Causes */}
              <div className="grid grid-cols-3 gap-4">
                <div className="card">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Shared Root Causes</h4>
                  {comparisonResult.shared_root_causes?.map((cause, idx) => (
                    <div key={idx} className="text-sm text-gray-600 py-1">
                      {cause.replace(/_/g, ' ')}
                    </div>
                  ))}
                </div>
                <div className="card">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Unique to A</h4>
                  {comparisonResult.unique_to_a?.map((cause, idx) => (
                    <div key={idx} className="text-sm text-gray-600 py-1">
                      {cause.replace(/_/g, ' ')}
                    </div>
                  ))}
                  {(!comparisonResult.unique_to_a || comparisonResult.unique_to_a.length === 0) && (
                    <p className="text-sm text-gray-400">None</p>
                  )}
                </div>
                <div className="card">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Unique to B</h4>
                  {comparisonResult.unique_to_b?.map((cause, idx) => (
                    <div key={idx} className="text-sm text-gray-600 py-1">
                      {cause.replace(/_/g, ' ')}
                    </div>
                  ))}
                  {(!comparisonResult.unique_to_b || comparisonResult.unique_to_b.length === 0) && (
                    <p className="text-sm text-gray-400">None</p>
                  )}
                </div>
              </div>

              {/* Severity Comparison */}
              {comparisonResult.severity_comparison && (
                <div className="card">
                  <h4 className="text-sm font-medium text-gray-700 mb-3">Severity Comparison</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 bg-gray-50 rounded-lg">
                      <p className="text-xs text-gray-500 mb-1">Incident A</p>
                      <div className="flex items-center space-x-2">
                        <SeverityBadge severity={comparisonResult.severity_comparison.incident_a_severity} />
                        <span className="font-mono text-sm">
                          {comparisonResult.severity_comparison.incident_a_zscore}σ
                        </span>
                      </div>
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg">
                      <p className="text-xs text-gray-500 mb-1">Incident B</p>
                      <div className="flex items-center space-x-2">
                        <SeverityBadge severity={comparisonResult.severity_comparison.incident_b_severity} />
                        <span className="font-mono text-sm">
                          {comparisonResult.severity_comparison.incident_b_zscore}σ
                        </span>
                      </div>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 mt-3">
                    {comparisonResult.severity_comparison.severity_relationship}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Simulation Tab */}
      {activeTab === 'simulation' && (
        <div className="space-y-6" role="tabpanel" id="panel-simulation" aria-labelledby="tab-simulation">
          <div className="card">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">What-If Scenario</h2>
            <p className="text-sm text-gray-600 mb-4">
              Specify metric perturbations to simulate downstream impacts
            </p>

            <div className="space-y-3">
              {perturbations.map((p, idx) => (
                <fieldset key={idx} className="flex items-center space-x-3">
                  <legend className="sr-only">Perturbation {idx + 1}</legend>
                  <div className="flex-1">
                    <label htmlFor={`metric-${idx}`} className="sr-only">Metric name</label>
                    <input
                      id={`metric-${idx}`}
                      type="text"
                      className="input w-full"
                      placeholder="Metric name (e.g., order_volume)"
                      value={p.metric}
                      onChange={(e) => updatePerturbation(idx, 'metric', e.target.value)}
                    />
                  </div>
                  <div>
                    <label htmlFor={`change-${idx}`} className="sr-only">Change amount</label>
                    <input
                      id={`change-${idx}`}
                      type="text"
                      className="input w-32"
                      placeholder="+50%"
                      value={p.change}
                      onChange={(e) => updatePerturbation(idx, 'change', e.target.value)}
                    />
                  </div>
                </fieldset>
              ))}
              <button
                onClick={addPerturbation}
                className="text-sm text-primary-600 hover:text-primary-700"
              >
                + Add perturbation
              </button>
            </div>

            <button
              onClick={handleRunSimulation}
              disabled={loading}
              className="btn btn-primary flex items-center space-x-2 mt-4"
            >
              <Zap className="w-4 h-4" />
              <span>{loading ? 'Simulating...' : 'Run Simulation'}</span>
            </button>
          </div>

          {simulationResult && (
            <div className="space-y-4">
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Simulation Narrative</h3>
                <p className="text-gray-700 leading-relaxed">{simulationResult.narrative}</p>
              </div>

              {simulationResult.triggered_incidents?.length > 0 && (
                <div className="card">
                  <h4 className="text-sm font-medium text-gray-700 mb-3">Predicted Incidents</h4>
                  <div className="flex flex-wrap gap-2">
                    {simulationResult.triggered_incidents.map((inc, idx) => (
                      <span
                        key={idx}
                        className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-red-100 text-red-800"
                      >
                        {inc.replace(/_/g, ' ')}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {simulationResult.triggered_cascades?.length > 0 && (
                <div className="card">
                  <h4 className="text-sm font-medium text-gray-700 mb-3">Predicted Cascades</h4>
                  <div className="flex flex-wrap gap-2">
                    {simulationResult.triggered_cascades.map((casc, idx) => (
                      <span
                        key={idx}
                        className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-orange-100 text-orange-800"
                      >
                        {casc}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {simulationResult.ml_insights?.health_score_impact && (
                <div className="card border-l-4 border-primary-500">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Health Score Impact</h4>
                  <div className="flex items-center gap-4">
                    <div className="text-center">
                      <p className="text-xs text-gray-500">Current</p>
                      <p className="text-2xl font-bold text-gray-900">{simulationResult.ml_insights.health_score_impact.current}</p>
                    </div>
                    <span className="text-gray-400 text-xl">→</span>
                    <div className="text-center">
                      <p className="text-xs text-gray-500">Projected</p>
                      <p className={`text-2xl font-bold ${simulationResult.ml_insights.health_score_impact.projected < simulationResult.ml_insights.health_score_impact.current ? 'text-red-600' : 'text-green-600'}`}>
                        {simulationResult.ml_insights.health_score_impact.projected}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {simulationResult.ml_insights?.churn_risk != null && simulationResult.ml_insights.churn_risk > 0 && (
                <div className="card border-l-4 border-amber-500">
                  <h4 className="text-sm font-medium text-gray-700 mb-1">Churn Risk (ML Prediction)</h4>
                  <p className="text-2xl font-bold text-amber-700">
                    {(simulationResult.ml_insights.churn_risk * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-gray-500 mt-1">Percentage of customers at risk of leaving under this scenario</p>
                </div>
              )}

              {simulationResult.simulated_metrics && (
                <div className="card">
                  <h4 className="text-sm font-medium text-gray-700 mb-3">Simulated Metric Values</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {Object.entries(simulationResult.simulated_metrics).map(([key, value]) => (
                      <div key={key} className="p-3 bg-gray-50 rounded-lg">
                        <p className="text-xs text-gray-500">{key.replace(/_/g, ' ')}</p>
                        <p className="text-lg font-bold text-gray-900 mt-1">
                          {typeof value === 'number' ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : value}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {simulationResult.models_used?.length > 0 && (
                <p className="text-xs text-gray-400 text-right">
                  Powered by: {simulationResult.models_used.join(', ')}
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
