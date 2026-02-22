import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { useState, useEffect, useCallback, lazy, Suspense } from 'react'
import { Toaster } from 'react-hot-toast'

// Layout components
import Sidebar from './components/layout/Sidebar'
import Header from './components/layout/Header'

// React-specialist agent: Error boundaries for production resilience
import ErrorBoundary from './components/ErrorBoundary'
import LoadingFallback from './components/LoadingFallback'

// React-specialist agent: Code splitting with React.lazy for optimized bundle
const ConnectionSetup = lazy(() => import('./pages/ConnectionSetup'))
const HealthDashboard = lazy(() => import('./pages/HealthDashboard'))
const IncidentsList = lazy(() => import('./pages/IncidentsList'))
const IncidentDetail = lazy(() => import('./pages/IncidentDetail'))
const PostmortemView = lazy(() => import('./pages/PostmortemView'))
const MonitorsDashboard = lazy(() => import('./pages/MonitorsDashboard'))
const ComparisonSimulation = lazy(() => import('./pages/ComparisonSimulation'))
const CreditPulse = lazy(() => import('./pages/CreditPulse'))
const Insights = lazy(() => import('./pages/Insights'))
const ModelPerformance = lazy(() => import('./pages/ModelPerformance'))

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)

  // Check for stored auth token on mount
  useEffect(() => {
    const token = localStorage.getItem('ledgerguard_token')
    if (token) {
      setIsAuthenticated(true)
    }
  }, [])

  // React-specialist agent: useCallback to stabilize handler references
  const handleLogin = useCallback((token, realmId) => {
    localStorage.setItem('ledgerguard_token', token)
    localStorage.setItem('ledgerguard_realm_id', realmId)
    setIsAuthenticated(true)
  }, [])

  const handleLogout = useCallback(() => {
    localStorage.removeItem('ledgerguard_token')
    localStorage.removeItem('ledgerguard_realm_id')
    setIsAuthenticated(false)
  }, [])

  const handleSidebarToggle = useCallback(() => {
    setSidebarOpen(prev => !prev)
  }, [])

  const handleMobileSidebarToggle = useCallback(() => {
    setMobileSidebarOpen(prev => !prev)
  }, [])

  const handleMobileSidebarClose = useCallback(() => {
    setMobileSidebarOpen(false)
  }, [])

  return (
    <BrowserRouter>
      <ErrorBoundary name="app-root">
        <div className="min-h-screen bg-surface-50">
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#363636',
                color: '#fff',
              },
              success: {
                duration: 3000,
                iconTheme: {
                  primary: '#10b981',
                  secondary: '#fff',
                },
              },
              error: {
                duration: 5000,
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#fff',
                },
              },
            }}
          />

          {isAuthenticated ? (
            <div className="flex h-screen">
              {/* Skip to content link for keyboard/screen reader users */}
              <a
                href="#main-content"
                className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-[100] focus:px-4 focus:py-2 focus:bg-primary-600 focus:text-white focus:rounded-lg focus:shadow-lg focus:text-sm focus:font-medium"
              >
                Skip to main content
              </a>

              {/* Desktop Sidebar */}
              <div className="hidden md:block">
                <Sidebar isOpen={sidebarOpen} onToggle={handleSidebarToggle} />
              </div>

              {/* Mobile Sidebar Overlay */}
              {mobileSidebarOpen && (
                <div className="fixed inset-0 z-50 md:hidden">
                  <div
                    className="absolute inset-0 bg-black/40"
                    onClick={handleMobileSidebarClose}
                    aria-hidden="true"
                  />
                  <div className="relative w-64 h-full">
                    <Sidebar isOpen={true} onToggle={handleMobileSidebarClose} onNavigate={handleMobileSidebarClose} />
                  </div>
                </div>
              )}

              {/* Main content */}
              <div className="flex-1 flex flex-col overflow-hidden">
                <Header onLogout={handleLogout} onMenuToggle={handleMobileSidebarToggle} />

                <main id="main-content" className="flex-1 overflow-y-auto bg-surface-50 p-6" tabIndex={-1}>
                  {/* React-specialist agent: Suspense boundary for lazy routes */}
                  <ErrorBoundary name="page-content">
                    <Suspense fallback={<LoadingFallback message="Loading page..." />}>
                      <Routes>
                        <Route path="/" element={<Navigate to="/dashboard" replace />} />
                        <Route path="/setup" element={<ConnectionSetup onLogin={handleLogin} />} />
                        <Route path="/dashboard" element={<HealthDashboard />} />
                        <Route path="/models" element={<ModelPerformance />} />
                        <Route path="/credit-pulse" element={<CreditPulse />} />
                        <Route path="/insights" element={<Insights />} />
                        <Route path="/incidents" element={<IncidentsList />} />
                        <Route path="/incidents/:id" element={<IncidentDetail />} />
                        <Route path="/incidents/:id/postmortem" element={<PostmortemView />} />
                        <Route path="/monitors" element={<MonitorsDashboard />} />
                        <Route path="/analysis" element={<ComparisonSimulation />} />
                        <Route path="*" element={<Navigate to="/dashboard" replace />} />
                      </Routes>
                    </Suspense>
                  </ErrorBoundary>
                </main>
              </div>
            </div>
          ) : (
            <Suspense fallback={<LoadingFallback message="Loading..." />}>
              <Routes>
                <Route path="/setup" element={<ConnectionSetup onLogin={handleLogin} />} />
                <Route path="*" element={<Navigate to="/setup" replace />} />
              </Routes>
            </Suspense>
          )}
        </div>
      </ErrorBoundary>
    </BrowserRouter>
  )
}

export default App
