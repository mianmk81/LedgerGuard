import axios from 'axios'

const isDev = import.meta.env.DEV

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor - Add auth token and request ID
apiClient.interceptors.request.use(
  (config) => {
    // Add JWT token if available
    const token = localStorage.getItem('ledgerguard_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }

    // Add request ID for tracing
    const requestId = crypto.randomUUID()
    config.headers['X-Request-ID'] = requestId

    if (isDev) {
      console.log(`[API] ${config.method.toUpperCase()} ${config.url}`, { requestId })
    }

    return config
  },
  (error) => {
    if (isDev) console.error('[API Request Error]', error)
    return Promise.reject(error)
  }
)

// Response interceptor - Handle errors and extract data
apiClient.interceptors.response.use(
  (response) => {
    // Unwrap BRE response envelope: {success, data} → data
    const result = response.data
    if (result && typeof result === 'object' && result.success !== undefined && result.data !== undefined) {
      return result.data
    }
    return result
  },
  (error) => {
    const requestId = error.config?.headers['X-Request-ID']

    if (error.response) {
      if (isDev) {
        console.error(`[API Error] ${error.config?.url}`, {
          requestId,
          status: error.response.status,
        })
      }

      if (error.response.status === 401) {
        localStorage.removeItem('ledgerguard_token')
        localStorage.removeItem('ledgerguard_realm_id')
        window.location.href = '/setup'
      }
    } else if (isDev) {
      if (error.request) {
        console.error('[API] No response received', { requestId })
      } else {
        console.error('[API]', error.message)
      }
    }

    return Promise.reject(error)
  }
)

// API methods
export const api = {
  // Auth
  auth: {
    initiateOAuth: () => apiClient.get('/api/v1/auth/authorize'),
    callback: (code, state, realmId) =>
      apiClient.get('/api/v1/auth/callback', { params: { code, state, realmId } }),
    refresh: (realmId) => apiClient.post('/api/v1/auth/refresh', { realm_id: realmId }),
    logout: () => apiClient.post('/api/v1/auth/logout'),
    demoAvailable: () => apiClient.get('/api/v1/auth/demo-available'),
    demoToken: () => apiClient.post('/api/v1/auth/demo-token'),
  },

  // Connection
  connection: {
    getStatus: () => apiClient.get('/api/v1/connection/status'),
    disconnect: () => apiClient.post('/api/v1/connection/disconnect'),
    test: () => apiClient.post('/api/v1/connection/test'),
  },

  // Ingestion
  ingestion: {
    start: (data) => apiClient.post('/api/v1/ingestion/start', data),
    getStatus: (jobId) => apiClient.get(`/api/v1/ingestion/status/${jobId}`),
    getHistory: (limit = 10) => apiClient.get('/api/v1/ingestion/history', { params: { limit } }),
  },

  // Analysis
  analysis: {
    run: (data) => apiClient.post('/api/v1/analysis/run', data),
    getResult: (analysisId) => apiClient.get(`/api/v1/analysis/result/${analysisId}`),
  },

  // Incidents
  incidents: {
    list: (params) => apiClient.get('/api/v1/incidents/', { params }),
    get: (incidentId) => apiClient.get(`/api/v1/incidents/${incidentId}`),
    getPostmortem: (incidentId, format = 'json') =>
      apiClient.get(`/api/v1/incidents/${incidentId}/postmortem`, { params: { format } }),
    analyze: (incidentId) => apiClient.post(`/api/v1/incidents/${incidentId}/analyze`),
  },

  // Cascades
  cascades: {
    getBlastRadius: (incidentId, maxDepth = 3) =>
      apiClient.get(`/api/v1/cascades/${incidentId}`, { params: { max_depth: maxDepth } }),
    getImpact: (incidentId) => apiClient.get(`/api/v1/cascades/${incidentId}/impact`),
  },

  // Monitors
  monitors: {
    list: (params) => apiClient.get('/api/v1/monitors/', { params }),
    create: (data) => apiClient.post('/api/v1/monitors/', data),
    get: (monitorId) => apiClient.get(`/api/v1/monitors/${monitorId}`),
    evaluate: () => apiClient.get('/api/v1/monitors/evaluate'),
    alerts: (params) => apiClient.get('/api/v1/monitors/alerts', { params }),
    toggle: (monitorId) => apiClient.put(`/api/v1/monitors/${monitorId}/toggle`),
    delete: (monitorId) => apiClient.delete(`/api/v1/monitors/${monitorId}`),
  },

  // Comparison
  comparison: {
    compare: (data) => apiClient.post('/api/v1/comparison/compare', data),
    whatIf: (data) => apiClient.post('/api/v1/comparison/whatif', data),
  },

  // Simulation
  simulation: {
    run: (data) => apiClient.post('/api/v1/simulation/run', data),
    getHistory: (limit = 10) => apiClient.get('/api/v1/simulation/history', { params: { limit } }),
  },

  // Credit Pulse (SMB financial health score + explainability)
  creditPulse: {
    get: (lookbackDays = 7) =>
      apiClient.get('/api/v1/credit-pulse', { params: { lookback_days: lookbackDays } }),
  },

  // Warnings (Risk Outlook — forward prediction, early warnings)
  warnings: {
    list: () => apiClient.get('/api/v1/warnings'),
  },

  // Insights (H4-H7 + extended)
  insights: {
    cashRunway: (lookbackDays = 60) =>
      apiClient.get('/api/v1/insights/cash-runway', { params: { lookback_days: lookbackDays } }),
    cashForecastCurve: (projectionMonths = 6, lookbackDays = 60) =>
      apiClient.get('/api/v1/insights/cash-forecast-curve', {
        params: { projection_months: projectionMonths, lookback_days: lookbackDays },
      }),
    invoiceDefaultRisk: (params = {}) =>
      apiClient.get('/api/v1/insights/invoice-default-risk', { params }),
    invoiceFollowUpPriorities: (topN = 5) =>
      apiClient.get('/api/v1/insights/invoice-follow-up-priorities', { params: { top_n: topN } }),
    supportTicketSentiment: (params = {}) =>
      apiClient.get('/api/v1/insights/support-ticket-sentiment', { params }),
    taxLiabilityEstimate: (params = {}) =>
      apiClient.get('/api/v1/insights/tax-liability-estimate', { params }),
    recommendations: (topN = 5) =>
      apiClient.get('/api/v1/insights/recommendations', { params: { top_n: topN } }),
    periodComparison: (period = 'month') =>
      apiClient.get('/api/v1/insights/period-comparison', { params: { period } }),
    scenarioExpense: (expenseCutPct = 10) =>
      apiClient.get('/api/v1/insights/scenario-expense', {
        params: { expense_cut_pct: expenseCutPct },
      }),
    scenarioInvoiceCollection: (topN = 5) =>
      apiClient.get('/api/v1/insights/scenario-invoice-collection', { params: { top_n: topN } }),
    recurringPatterns: (lookbackDays = 365) =>
      apiClient.get('/api/v1/insights/recurring-patterns', { params: { lookback_days: lookbackDays } }),
  },

  // Dashboard (Reports + Future Score)
  dashboard: {
    reports: (limit = 5) =>
      apiClient.get('/api/v1/dashboard/reports', { params: { limit } }),
    futureScore: (projectionDays = 30) =>
      apiClient.get('/api/v1/dashboard/future-score', {
        params: { projection_days: projectionDays },
      }),
  },

  // Metrics
  metrics: {
    getDashboard: (period = '30d') =>
      apiClient.get('/api/v1/metrics/dashboard', { params: { period } }),
    getTimeseries: (metricName, period = '30d', granularity = 'daily') =>
      apiClient.get(`/api/v1/metrics/timeseries/${metricName}`, {
        params: { period, granularity },
      }),
    getHealthScore: () => apiClient.get('/api/v1/metrics/health-score'),
  },

  // System
  system: {
    health: () => apiClient.get('/api/v1/system/health'),
    diagnostics: () => apiClient.get('/api/v1/system/diagnostics'),
    config: () => apiClient.get('/api/v1/system/config'),
    models: () => apiClient.get('/api/v1/system/models'),
    experiments: () => apiClient.get('/api/v1/system/experiments'),
    modelCards: () => apiClient.get('/api/v1/system/model-cards'),
    reportImage: (filename) => `${apiClient.defaults.baseURL}/api/v1/system/reports/${filename}`,
  },
}

export default apiClient
