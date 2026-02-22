import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import { ExternalLink, CheckCircle, Play } from 'lucide-react'

export default function ConnectionSetup({ onLogin }) {
  const [searchParams] = useSearchParams()
  const [loading, setLoading] = useState(false)
  const [demoLoading, setDemoLoading] = useState(false)
  const [error, setError] = useState(null)
  const [demoAvailable, setDemoAvailable] = useState(false)

  useEffect(() => {
    api.auth.demoAvailable().then((r) => setDemoAvailable(r.available)).catch(() => setDemoAvailable(false))
  }, [])

  useEffect(() => {
    // Handle OAuth callback
    const code = searchParams.get('code')
    const state = searchParams.get('state')
    const realmId = searchParams.get('realmId')

    if (code && state && realmId) {
      handleOAuthCallback(code, state, realmId)
    }
  }, [searchParams])

  const handleOAuthCallback = async (code, state, realmId) => {
    setLoading(true)
    setError(null)

    try {
      const response = await api.auth.callback(code, state, realmId)
      onLogin(response.access_token, response.realm_id)
    } catch (err) {
      setError('Failed to complete authentication. Please try again.')
      console.error('OAuth callback error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleConnectQuickBooks = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await api.auth.initiateOAuth()
      window.location.href = response.authorization_url
    } catch (err) {
      setError('Failed to initiate QuickBooks connection.')
      console.error('OAuth init error:', err)
      setLoading(false)
    }
  }

  const handleTryDemo = async () => {
    setDemoLoading(true)
    setError(null)

    try {
      const response = await api.auth.demoToken()
      onLogin(response.access_token, response.realm_id)
    } catch (err) {
      setError('Demo mode is not available. Make sure the API is running with DEV_MODE=true.')
      console.error('Demo token error:', err)
    } finally {
      setDemoLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 to-primary-100 flex items-center justify-center p-6">
      <div className="max-w-md w-full">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-primary-600 rounded-2xl mb-4">
            <span className="text-white font-bold text-2xl">L</span>
          </div>
          <h1 className="text-3xl font-bold text-gray-900">LedgerGuard</h1>
          <p className="text-gray-600 mt-2">Business Reliability Engine</p>
        </div>

        {/* Connection Card */}
        <div className="bg-white rounded-xl shadow-lg p-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Connect to QuickBooks</h2>
          <p className="text-sm text-gray-600 mb-6">
            Connect your QuickBooks Online account to start monitoring business reliability.
          </p>

          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          {demoAvailable && (
            <button
              onClick={handleTryDemo}
              disabled={demoLoading || loading}
              className="w-full mb-4 btn btn-secondary flex items-center justify-center space-x-2 border-2 border-primary-200 bg-primary-50 hover:bg-primary-100"
            >
              {demoLoading ? (
                <>
                  <div className="w-5 h-5 border-2 border-primary-600 border-t-transparent rounded-full animate-spin" />
                  <span>Loading...</span>
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  <span>Try Demo</span>
                </>
              )}
            </button>
          )}

          <button
            onClick={handleConnectQuickBooks}
            disabled={loading}
            className="w-full btn btn-primary flex items-center justify-center space-x-2"
          >
            {loading ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Connecting...</span>
              </>
            ) : (
              <>
                <ExternalLink className="w-5 h-5" />
                <span>Connect QuickBooks Online</span>
              </>
            )}
          </button>

          {demoAvailable && (
            <p className="mt-3 text-xs text-gray-500 text-center">
              No QuickBooks? Use <strong>Try Demo</strong> to explore with sample data.
            </p>
          )}

          {/* Features */}
          <div className="mt-8 space-y-3">
            <h3 className="text-sm font-semibold text-gray-900">What you'll get:</h3>
            {[
              'Automated anomaly detection',
              'Root cause analysis',
              'Business health monitoring',
              'Incident postmortem reports',
            ].map((feature, idx) => (
              <div key={idx} className="flex items-start space-x-2">
                <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                <span className="text-sm text-gray-700">{feature}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="mt-6 text-center">
          <p className="text-xs text-gray-600">
            By connecting, you agree to allow LedgerGuard to access your QuickBooks data.
          </p>
        </div>
      </div>
    </div>
  )
}
