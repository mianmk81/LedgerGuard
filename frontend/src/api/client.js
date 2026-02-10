import axios from 'axios'

// Create axios instance with base configuration
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

    console.log(`[API Request] ${config.method.toUpperCase()} ${config.url}`, {
      requestId,
      params: config.params,
    })

    return config
  },
  (error) => {
    console.error('[API Request Error]', error)
    return Promise.reject(error)
  }
)

// Response interceptor - Handle errors and extract data
apiClient.interceptors.response.use(
  (response) => {
    const requestId = response.config.headers['X-Request-ID']
    console.log(`[API Response] ${response.config.url}`, {
      requestId,
      status: response.status,
    })
    return response.data
  },
  (error) => {
    const requestId = error.config?.headers['X-Request-ID']

    if (error.response) {
      // Server responded with error status
      console.error(`[API Error] ${error.config.url}`, {
        requestId,
        status: error.response.status,
        data: error.response.data,
      })

      // Handle 401 Unauthorized
      if (error.response.status === 401) {
        localStorage.removeItem('ledgerguard_token')
        localStorage.removeItem('ledgerguard_realm_id')
        window.location.href = '/setup'
      }
    } else if (error.request) {
      // Request made but no response
      console.error('[API Error] No response received', { requestId })
    } else {
      // Error in request setup
      console.error('[API Error]', error.message)
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
  },

  // Cascades
  cascades: {
    getBlastRadius: (incidentId, maxDepth = 3) =>
      apiClient.get(`/api/v1/cascades/${incidentId}`, { params: { max_depth: maxDepth } }),
    getImpact: (incidentId) => apiClient.get(`/api/v1/cascades/${incidentId}/impact`),
  },

  // Monitors
  monitors: {
    list: () => apiClient.get('/api/v1/monitors/'),
    create: (data) => apiClient.post('/api/v1/monitors/', data),
    get: (monitorId) => apiClient.get(`/api/v1/monitors/${monitorId}`),
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
  },
}

export default apiClient
