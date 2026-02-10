import { useState } from 'react'
import { api } from '../api/client'
import { Play, TrendingUp, BarChart3 } from 'lucide-react'

export default function ComparisonSimulation() {
  const [activeTab, setActiveTab] = useState('comparison')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)

  const handleRunComparison = async () => {
    setLoading(true)
    try {
      const data = await api.comparison.compare({
        baseline_period: '2025-01',
        comparison_period: '2025-02',
        metrics: ['total_revenue', 'invoice_count'],
      })
      setResults(data)
    } catch (error) {
      console.error('Failed to run comparison:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleRunSimulation = async () => {
    setLoading(true)
    try {
      const data = await api.simulation.run({
        simulation_type: 'monte_carlo',
        iterations: 1000,
        parameters: { revenue_volatility: 0.15 },
        time_horizon_days: 90,
      })
      setResults(data)
    } catch (error) {
      console.error('Failed to run simulation:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Analysis & Simulation</h1>
        <p className="text-gray-600 mt-1">Compare periods and run what-if scenarios</p>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex space-x-8">
          <button
            onClick={() => setActiveTab('comparison')}
            className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'comparison'
                ? 'border-primary-600 text-primary-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <div className="flex items-center space-x-2">
              <TrendingUp className="w-4 h-4" />
              <span>Period Comparison</span>
            </div>
          </button>
          <button
            onClick={() => setActiveTab('simulation')}
            className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'simulation'
                ? 'border-primary-600 text-primary-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <div className="flex items-center space-x-2">
              <BarChart3 className="w-4 h-4" />
              <span>Simulation</span>
            </div>
          </button>
        </nav>
      </div>

      {/* Comparison Tab */}
      {activeTab === 'comparison' && (
        <div className="space-y-6">
          <div className="card">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Configure Comparison</h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Baseline Period
                </label>
                <input type="month" className="input" defaultValue="2025-01" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Comparison Period
                </label>
                <input type="month" className="input" defaultValue="2025-02" />
              </div>
            </div>
            <button
              onClick={handleRunComparison}
              disabled={loading}
              className="btn btn-primary flex items-center space-x-2 mt-4"
            >
              <Play className="w-4 h-4" />
              <span>{loading ? 'Running...' : 'Run Comparison'}</span>
            </button>
          </div>

          {results && (
            <div className="card">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Results</h2>
              <div className="space-y-4">
                {results.results?.map((result, idx) => (
                  <div key={idx} className="p-4 bg-gray-50 rounded-lg">
                    <div className="font-medium text-gray-900 mb-2">
                      {result.metric.replace(/_/g, ' ').toUpperCase()}
                    </div>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <div className="text-gray-600">Baseline</div>
                        <div className="font-semibold text-gray-900">
                          {result.baseline_value.toLocaleString()}
                        </div>
                      </div>
                      <div>
                        <div className="text-gray-600">Current</div>
                        <div className="font-semibold text-gray-900">
                          {result.comparison_value.toLocaleString()}
                        </div>
                      </div>
                      <div>
                        <div className="text-gray-600">Change</div>
                        <div
                          className={`font-semibold ${
                            result.change_percent >= 0 ? 'text-green-600' : 'text-red-600'
                          }`}
                        >
                          {result.change_percent >= 0 ? '+' : ''}
                          {result.change_percent.toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Simulation Tab */}
      {activeTab === 'simulation' && (
        <div className="space-y-6">
          <div className="card">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Configure Simulation</h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Simulation Type
                </label>
                <select className="input">
                  <option value="monte_carlo">Monte Carlo</option>
                  <option value="stress_test">Stress Test</option>
                  <option value="scenario">Scenario Analysis</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Iterations</label>
                <input type="number" className="input" defaultValue="1000" />
              </div>
            </div>
            <button
              onClick={handleRunSimulation}
              disabled={loading}
              className="btn btn-primary flex items-center space-x-2 mt-4"
            >
              <Play className="w-4 h-4" />
              <span>{loading ? 'Running...' : 'Run Simulation'}</span>
            </button>
          </div>

          {results && results.results && (
            <div className="card">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Simulation Results</h2>
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-3">Revenue Projection</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Mean</span>
                      <span className="font-semibold">
                        ${results.results.revenue?.mean.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">P50</span>
                      <span className="font-semibold">
                        ${results.results.revenue?.p50.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">P95</span>
                      <span className="font-semibold">
                        ${results.results.revenue?.p95.toLocaleString()}
                      </span>
                    </div>
                  </div>
                </div>
                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-3">Risk Metrics</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">VaR (95%)</span>
                      <span className="font-semibold">
                        ${results.risk_metrics?.var_95.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Max Drawdown</span>
                      <span className="font-semibold text-red-600">
                        -${results.risk_metrics?.max_drawdown.toLocaleString()}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
